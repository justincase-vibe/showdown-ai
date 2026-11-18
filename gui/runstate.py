"""P4.3 — the on-disk run-state protocol.

The GUI and the trainer subprocess communicate EXCLUSIVELY through this
directory protocol (plus the process table for kill), so:

  * a GUI restart never loses a running training (state is on disk);
  * the trainer is observable without any shared memory;
  * the protocol is trivially testable (it's just files).

Layout (under ``runs/<run-id>/``):

    config.json    the TrainConfig as submitted (single source of truth)
    status.json    {state, run_id, pid, started, updated, rounds_planned,
                    round, exit_code, error, stop_requested}
    heartbeat.json {ts, round, battle, note}      <- liveness signal
    metrics.jsonl  ONE JSON object per completed round (append-only)
    stop.flag      presence = "stop gracefully" (checked between battles)
    trainer.log    the subprocess's stdout+stderr (append-only)
    checkpoints/   the run's checkpoints (``selfplay_roundN.pt``)

States: ``starting -> running -> completed | stopped | failed``.

Windows note (caught in review, not assumed): ``os.kill(pid, 0)`` is NOT a
liveness probe on Windows — CPython routes non-CTRL signals to
``TerminateProcess``. Liveness therefore goes through ``tasklist``.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

#: states that mean "a live training should exist for this run"
ACTIVE_STATES = ("starting", "running")
#: states that are final (reconciliation never leaves them alone)
FINAL_STATES = ("completed", "stopped", "failed")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write(path: Path, text: str) -> None:
    """Write-then-rename so a reader never sees a half-written JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ── status.json ──────────────────────────────────────────────────────────────

def write_status(run_dir: Path, **fields: Any) -> Dict[str, Any]:
    """Merge ``fields`` into status.json (create if missing) and return the
    new status."""
    run_dir = Path(run_dir)
    st = read_status(run_dir)
    st.update(fields)
    st["updated"] = now_iso()
    _atomic_write(run_dir / "status.json", json.dumps(st, indent=2))
    return st


def read_status(run_dir: Path) -> Dict[str, Any]:
    p = Path(run_dir) / "status.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}  # a torn read is not a protocol crash — the reconciler re-derives


# ── heartbeat.json ───────────────────────────────────────────────────────────

def write_heartbeat(run_dir: Path, round_i: int, battle: int, note: str = "") -> None:
    _atomic_write(Path(run_dir) / "heartbeat.json", json.dumps({
        "ts": time.time(), "round": int(round_i), "battle": int(battle), "note": note,
    }))


def read_heartbeat(run_dir: Path) -> Dict[str, Any]:
    p = Path(run_dir) / "heartbeat.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def heartbeat_age_s(run_dir: Path) -> Optional[float]:
    ts = read_heartbeat(run_dir).get("ts")
    if not ts:
        return None
    return max(0.0, time.time() - float(ts))


# ── metrics.jsonl ────────────────────────────────────────────────────────────

def append_metric(run_dir: Path, row: Dict[str, Any]) -> None:
    p = Path(run_dir) / "metrics.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def read_metrics(run_dir: Path, n: Optional[int] = None) -> List[Dict[str, Any]]:
    p = Path(run_dir) / "metrics.jsonl"
    if not p.exists():
        return []
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a torn final line mid-write is not a protocol crash
    return rows[-n:] if n else rows


# ── stop flag ────────────────────────────────────────────────────────────────

def request_stop(run_dir: Path) -> None:
    (Path(run_dir) / "stop.flag").write_text(now_iso(), encoding="utf-8")


def stop_requested(run_dir: Path) -> bool:
    return (Path(run_dir) / "stop.flag").exists()


# ── run discovery + reconciliation ───────────────────────────────────────────

def list_runs(runs_dir: Path) -> List[str]:
    """Run ids, newest first (run-id is a UTC timestamp -> lexicographic)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted((p.name for p in runs_dir.iterdir()
                   if p.is_dir() and (p / "status.json").exists()), reverse=True)


def run_dir_for(runs_dir: Path, run_id: str) -> Path:
    return Path(runs_dir) / run_id


def pid_alive(pid: Any) -> bool:
    """Cross-platform liveness. Windows: ``tasklist`` (``os.kill`` is NOT
    safe here — see module docstring). POSIX: signal 0 probe."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        # EXACT CSV field match — a substring test would false-positive
        # (PID 12 is "in" the string "1234").
        r = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                           capture_output=True, text=True)
        for line in r.stdout.splitlines():
            fields = [f.strip().strip('"') for f in line.split(",")]
            if len(fields) >= 2 and fields[1] == str(pid):
                return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def reconcile(run_dir: Path) -> Dict[str, Any]:
    """Backstop for runs whose process died without writing a final status
    (force-kill, crash, power loss). If the run claims to be active but its
    PID is gone, finalize it: stopped if a stop was requested, else failed
    (a clean completion always writes its own final status first).

    Returns the (possibly updated) status. Idempotent for final states.

    D80: a status with ``pid=None`` is NEVER finalized — that means the
    launcher has not finished starting (the pid is the first fact that can
    exist). Finalizing it would mark a live just-started run 'failed'."""
    st = read_status(run_dir)
    if not st or st.get("state") not in ACTIVE_STATES:
        return st
    if st.get("pid") is None:
        return st
    if pid_alive(st.get("pid")):
        return st
    # a dead PID must ALWAYS end in a FINAL state: stopped if a stop was
    # requested (flag on disk or in status), failed otherwise. A pre-existing
    # error is preserved; otherwise a default one is recorded for the failed case.
    stop_req = bool(st.get("stop_requested") or stop_requested(run_dir))
    fields: Dict[str, Any] = {
        "state": "stopped" if stop_req else "failed",
        "stop_requested": stop_req,
        "exit_code": st.get("exit_code"),
    }
    if not stop_req and not st.get("error"):
        fields["error"] = "trainer process died without a final status (killed or crashed)"
    return write_status(run_dir, **fields)


def checkpoints(run_dir: Path) -> List[str]:
    d = Path(run_dir) / "checkpoints"
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.suffix == ".pt")


def kill_tree(pid: int) -> None:
    """Kill the trainer AND its children (fork servers / Node calc procs).
    Windows: ``taskkill /T /F``; POSIX: SIGKILL to the process group (the
    launcher gives the trainer its own session/group)."""
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True)
    else:
        import signal
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
