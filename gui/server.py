"""P4.3.2 — the GUI control plane (FastAPI + WebSocket + static dashboard).

The server NEVER trains. It is a thin, disk-backed control plane over
``gui.launcher`` / ``gui.runstate`` (the P4.3.1 protocol), so:

  * the GUI process is disposable — every fact it serves comes from the run
    dir on disk (a restarted server sees the same live training);
  * the only cross-process control is the same one the launcher already uses
    (write intent to disk + the process table for kills);
  * every endpoint is CPU-testable (fastapi TestClient, no GPU, no training
    except the established 1-battle live smoke in the test file).

Routes:

    GET  /                          the static dashboard (gui/static/index.html,
                                    fully offline — no CDN, no external JS)
    GET  /api/status                the launcher's status view (newest run,
                                    env info, metrics tail, checkpoints)
    GET  /api/runs                  the run HISTORY (newest first, last 20)
    POST /api/train/start           {"config": {...}} -> start_trainer.
                                    400 = bad config (every problem listed) or
                                    device=cuda without CUDA; 409 = a run is
                                    already active.
    POST /api/train/stop            {"force": false} graceful (waits for the
                                    in-flight battle, then the trainer exits
                                    clean); {"force": true} kills the tree.
                                    409 = nothing to stop.
    GET  /api/metrics?run_id=&n=    the metrics tail (default: newest run, 50)
    GET  /api/checkpoints?run_id=   checkpoint names + sizes (newest run)
    WS   /ws/stream                 push: a status frame when the STABLE part
                                    of the status changes, a metrics frame
                                    when new round lines appear (file-tail,
                                    no push infra; heartbeat_age_s is volatile
                                    by design and never triggers a frame)
    GET  /files/{run_id}/{path}     checkpoint download (path-traversal
                                    guarded at BOTH levels: run_id and path)

Run it:  ``python -m gui``  (or ``start_gui.bat``) -> http://127.0.0.1:8700
"""
import asyncio
import time
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel

from . import launcher, runstate


# NOTE: request models live at MODULE level on purpose — FastAPI resolves
# annotations by name, and a class defined inside create_app() is invisible to
# that resolver (it silently degrades to a query param; caught in test_p43_2).
class StartRequest(BaseModel):
    config: Dict[str, Any]


class StopRequest(BaseModel):
    force: bool = False

#: how often the WS handler polls the run dir (a file-tail, not a push)
WS_POLL_S = 1.5
#: how many historical round lines a fresh WS connection receives
WS_INITIAL_SNAPSHOT = 200

STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"

#: fields of launcher.status() that are VOLATILE by design (heartbeat age is
#: recomputed from wall-clock each call) — a WS frame must not be triggered by
#: these alone.
_STABLE_STATUS_FIELDS = ("run_id", "state", "pid", "round", "rounds_planned",
                         "error", "stop_requested", "started")


def _status_key(st: Dict[str, Any]) -> tuple:
    return tuple(st.get(k) for k in _STABLE_STATUS_FIELDS) + (
        tuple(st.get("checkpoints") or ()),)


def _run_dir_guarded(runs: Path, run_id: Optional[str]) -> Optional[Path]:
    """Resolve ``runs/run_id`` and refuse anything that escapes ``runs``
    (run_id is user-supplied — ``..`` included) or does not exist."""
    if not run_id or not isinstance(run_id, str):
        return None
    rd = (Path(runs) / run_id).resolve()
    base = Path(runs).resolve()
    # strictly INSIDE runs — a run_id that resolves to runs/ itself (e.g.
    # "<some-run>/..") would otherwise expose other runs' files
    if rd == base or not rd.is_relative_to(base) or not rd.is_dir():
        return None
    return rd


def create_app(runs_dir: Optional[Path] = None) -> FastAPI:
    """Build the control-plane app. ``runs_dir`` defaults to the project's
    ``runs/`` (inject a tmp dir in tests — the app never touches the real
    one)."""
    runs = Path(runs_dir).resolve() if runs_dir is not None else launcher.default_runs_dir()
    app = FastAPI(title="showdown-ai-training-gui", version="0.4.0")
    app.state.runs_dir = runs

    # ── dashboard ──────────────────────────────────────────────────────
    @app.get("/")
    def index() -> FileResponse:
        if not STATIC_INDEX.is_file():
            raise HTTPException(status_code=500, detail="gui/static/index.html is missing")
        return FileResponse(str(STATIC_INDEX))

    # ── status / history ───────────────────────────────────────────────
    @app.get("/api/status")
    def api_status() -> Dict[str, Any]:
        return launcher.status(runs_dir=runs)

    @app.get("/api/runs")
    def api_runs() -> Dict[str, Any]:
        out = []
        for rid in runstate.list_runs(runs)[:20]:
            st = runstate.read_status(runs / rid)
            out.append({"run_id": rid, "state": st.get("state"),
                        "round": st.get("round"), "rounds_planned": st.get("rounds_planned"),
                        "started": st.get("started"), "updated": st.get("updated"),
                        "error": st.get("error")})
        return {"runs": out}

    # ── control ────────────────────────────────────────────────────────
    @app.post("/api/train/start")
    def api_start(body: StartRequest) -> Dict[str, Any]:
        # validate LOUDLY (every problem listed) before anything is spawned
        try:
            tc = launcher.validate_train_config(body.config)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        # device=cuda without CUDA is a config error, not a trainer crash
        if tc.device == "cuda" and not launcher.cuda_available():
            raise HTTPException(status_code=400,
                                detail="device='cuda' was requested but "
                                       "torch.cuda.is_available() is False on this machine")
        try:
            info = launcher.start_trainer(body.config, runs_dir=runs)
        except RuntimeError as exc:  # a run is already active
            raise HTTPException(status_code=409, detail=str(exc))
        return {"started": True, **info}

    @app.post("/api/train/stop")
    def api_stop(body: StopRequest) -> Dict[str, Any]:
        """Synchronous on purpose: the graceful stop waits for the in-flight
        battle to finish (the pinned P4.3.1 contract), so the response
        already carries the final state. Cap: the launcher's 90 s grace, then
        it force-kills."""
        try:
            return launcher.stop_trainer(runs_dir=runs, force=bool(body.force))
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    # ── per-run data ───────────────────────────────────────────────────
    @app.get("/api/metrics")
    def api_metrics(run_id: Optional[str] = None, n: int = 50) -> Dict[str, Any]:
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise HTTPException(status_code=422, detail=f"n must be an int >= 1 (got {n!r})")
        n = min(n, 1000)
        if run_id is None:
            listed = runstate.list_runs(runs)
            if not listed:
                return {"run_id": None, "rows": []}
            run_id = listed[0]
        rd = _run_dir_guarded(runs, run_id)
        if rd is None:
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id!r}")
        return {"run_id": rd.name, "rows": runstate.read_metrics(rd, n)}

    @app.get("/api/checkpoints")
    def api_checkpoints(run_id: Optional[str] = None) -> Dict[str, Any]:
        if run_id is None:
            listed = runstate.list_runs(runs)
            if not listed:
                return {"run_id": None, "checkpoints": []}
            run_id = listed[0]
        rd = _run_dir_guarded(runs, run_id)
        if rd is None:
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id!r}")
        names = runstate.checkpoints(rd)
        ckpts = []
        for name in names:
            p = rd / "checkpoints" / name
            ckpts.append({"name": name, "size_bytes": p.stat().st_size,
                          "modified": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(p.stat().st_mtime))})
        return {"run_id": rd.name, "checkpoints": ckpts}

    # ── live stream (file-tail over the run protocol) ──────────────────
    @app.websocket("/ws/stream")
    async def ws_stream(ws: WebSocket) -> None:
        """Push status frames on stable-status change + metrics frames when
        new round lines land. A fresh connection receives the current status
        plus a bounded snapshot of past round lines. No push infra — this is
        a ~1.5 s poll of the run dir, which is exactly what the protocol is
        built for."""
        await ws.accept()
        seen: Dict[str, int] = {}
        last_key: Optional[tuple] = None
        try:
            while True:
                st = launcher.status(runs_dir=runs)
                key = _status_key(st)
                if key != last_key:
                    await ws.send_json({"type": "status", "status": st})
                    last_key = key

                rid = st.get("run_id")
                if rid:
                    rd = runs / rid
                    rows = runstate.read_metrics(rd)
                    prev = seen.get(rid)
                    if prev is None:
                        snapshot = rows[-WS_INITIAL_SNAPSHOT:]
                        seen[rid] = len(rows)
                        if snapshot:
                            await ws.send_json({"type": "metrics", "run_id": rid,
                                                "rows": snapshot})
                    elif len(rows) > prev:
                        fresh = rows[prev:]
                        seen[rid] = len(rows)
                        await ws.send_json({"type": "metrics", "run_id": rid,
                                            "rows": fresh})
                await asyncio.sleep(WS_POLL_S)
        except WebSocketDisconnect:
            return
        except Exception:  # noqa: BLE001 — client gone / send failed: close clean
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    # ── file download (double traversal guard: run_id AND path) ────────
    @app.get("/files/{run_id}/{file_path:path}")
    def api_file(run_id: str, file_path: str) -> FileResponse:
        rd = _run_dir_guarded(runs, run_id)
        if rd is None:
            raise HTTPException(status_code=404, detail=f"unknown run: {run_id!r}")
        target = (rd / file_path).resolve()
        if not target.is_relative_to(rd) or not target.is_file():
            raise HTTPException(status_code=404, detail="file not found")
        return FileResponse(str(target))

    return app


#: module-level app for ``uvicorn gui.server:app`` / ``python -m gui``
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("gui.server:app", host="127.0.0.1", port=8700)
