"""P4.3 — start / stop / inspect the trainer as a DETACHED child process.

The GUI (``gui.server``) is a thin control plane: it writes intent to disk
(config / stop flag) and reads state from disk (``gui.runstate``). The
training itself runs in its OWN python process (``gui.trainer_main``) that:

  * is spawned with a NEW process group and NO console window (Windows), so
    Ctrl+C / kill / restart of the GUI cannot take the training down with it;
  * writes its log, heartbeat, metrics and checkpoints into the run dir;
  * polls ``stop.flag`` between battles for a graceful stop.

All of this is CPU-verifiable (a 1-battle CPU run proves the loop end to
end); the user's GPU training is the same code with ``device="cuda"``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import runstate

#: project root (gui/ -> root); the child runs with this as its cwd so
#: ``python -m gui.trainer_main`` + the ``net``/``belief``/``damage`` packages
#: resolve no matter where the GUI server was started from.
ROOT = Path(__file__).resolve().parents[1]


def python_exe() -> str:
    return sys.executable or "python"


def default_runs_dir() -> Path:
    return ROOT / "runs"


# ── config validation (loud + specific, before anything is spawned) ─────────

def _pos_int(v: Any, name: str, problems: list, minimum: int = 1) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
        problems.append(f"{name} must be an int >= {minimum} (got {v!r})")


def _finite_float(v: Any, name: str, problems: list, lo: float = 0.0,
                  hi: float = None) -> None:
    import math
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
        problems.append(f"{name} must be a finite number (got {v!r})")
        return
    x = float(v)
    if x <= lo or (hi is not None and x > hi):
        problems.append(f"{name} must be in ({lo}, {hi if hi is not None else 'inf'}) (got {x})")


def validate_train_config(d: Optional[Dict[str, Any]]) -> "TrainConfig":
    """Validate a GUI-submitted config into a ``TrainConfig``. Raises
    ``ValueError`` with every problem listed (not just the first)."""
    from net.trainer import TrainConfig
    tc = TrainConfig.from_dict(d or {})
    problems = []
    if not isinstance(tc.battle_format, str) or not tc.battle_format.strip():
        problems.append("battle_format must be a non-empty string (e.g. 'gen9ou')")
    if tc.device not in ("cpu", "cuda"):
        problems.append("device must be 'cpu' or 'cuda' (got %r)" % (tc.device,))
    _pos_int(tc.battles_per_round, "battles_per_round", problems)
    _pos_int(tc.n_rounds, "n_rounds", problems)
    _pos_int(tc.max_steps, "max_steps", problems)
    _pos_int(tc.epochs, "epochs", problems)
    _pos_int(tc.save_every, "save_every", problems)
    _finite_float(tc.lr, "lr", problems, lo=0.0)
    _finite_float(tc.gamma, "gamma", problems, lo=0.0, hi=1.0)
    _finite_float(tc.lam, "lam", problems, lo=0.0, hi=1.0)
    _finite_float(tc.clip_eps, "clip_eps", problems, lo=0.0, hi=1.0)
    if tc.resume_from is not None:
        if not isinstance(tc.resume_from, str) or not tc.resume_from.strip():
            problems.append("resume_from must be a checkpoint path (or null)")
        elif not Path(tc.resume_from).exists():
            problems.append(f"resume_from does not exist: {tc.resume_from!r}")
    if problems:
        raise ValueError("; ".join(problems))
    return tc


# ── process control ──────────────────────────────────────────────────────────

def _popen_kwargs() -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if os.name == "nt":
        # new process group (survives GUI Ctrl+C / console teardown) + no
        # console window popping up for the user.
        kw["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                               | subprocess.CREATE_NO_WINDOW)
    else:
        kw["start_new_session"] = True  # own session -> kill_tree via killpg
    return kw


def start_trainer(cfg: Dict[str, Any], runs_dir: Optional[Path] = None
                  ) -> Dict[str, Any]:
    """Validate the config, refuse if a run is already active, and spawn the
    trainer DETACHED. Returns ``{run_id, pid, run_dir}``."""
    tc = validate_train_config(cfg)
    runs = Path(runs_dir) if runs_dir is not None else default_runs_dir()
    active = find_active_run(runs)
    if active is not None:
        rd, st = active
        raise RuntimeError(
            f"refusing to start: run {rd.name} is already active "
            f"(state={st.get('state')}, pid={st.get('pid')}) — stop it first")

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rd = runs / run_id
    try:
        rd.mkdir(parents=True)
    except FileExistsError as exc:
        raise RuntimeError(f"run dir already exists: {rd}") from exc

    (rd / "config.json").write_text(json.dumps(tc.to_dict(), indent=2), encoding="utf-8")

    log = open(rd / "trainer.log", "ab", buffering=0)
    try:
        cmd = [python_exe(), "-m", "gui.trainer_main",
               "--config", str(rd / "config.json"), "--run-dir", str(rd)]
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, **_popen_kwargs())
    except BaseException:
        log.close()
        # no process was ever created: remove the half-created run dir so it
        # doesn't linger (list_runs would ignore it anyway — no status.json)
        import shutil
        shutil.rmtree(rd, ignore_errors=True)
        raise
    # the child now holds its own reference to trainer.log; our handle may stay
    # open harmlessly (same file, append mode)
    # D80: the status FIRST appears with the LIVE pid — a 'starting' status with
    # pid=None would let a concurrent reconcile() (e.g. the dashboard's poll)
    # finalize the run as 'failed' before the pid lands (narrow race, wrong verdict).
    runstate.write_status(rd, state="starting", run_id=run_id, pid=proc.pid,
                          started=runstate.now_iso(), round=0,
                          rounds_planned=tc.n_rounds, error=None, exit_code=None)
    return {"run_id": run_id, "pid": proc.pid, "run_dir": str(rd)}


def find_active_run(runs_dir: Path) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """The newest run that is (still) active, or None. Reconciles stale
    final-state guesses for dead PIDs on the way (the GUI's backstop)."""
    runs_dir = Path(runs_dir)
    for run_id in runstate.list_runs(runs_dir):
        rd = runs_dir / run_id
        st = runstate.reconcile(rd)
        if st.get("state") in runstate.ACTIVE_STATES and runstate.pid_alive(st.get("pid")):
            return rd, st
    return None


def stop_trainer(runs_dir: Optional[Path] = None, timeout: float = 180.0,
                 force: bool = False) -> Dict[str, Any]:
    """Request a graceful stop (the trainer checks between battles and exits
    cleanly after the current one — D80: the flag is checked BEFORE EVERY
    battle, so a stop lands within one battle + one update). If ``force``
    (or the grace period expires) kill the whole process tree. Finalizes the
    run status either way. D80: grace default 180s — a CPU battle can take
    ~60s + the update, so the old 90s would have force-killed inside that
    window."""
    runs = Path(runs_dir) if runs_dir is not None else default_runs_dir()
    active = find_active_run(runs)
    if active is None:
        raise RuntimeError("no active training run to stop")
    rd, st = active
    pid = int(st.get("pid") or 0)
    runstate.request_stop(rd)
    runstate.write_status(rd, stop_requested=True)

    if not force and pid:
        deadline = time.time() + max(5.0, timeout)
        while runstate.pid_alive(pid) and time.time() < deadline:
            time.sleep(1.0)
    if runstate.pid_alive(pid):
        # graceful stop did not land (or force was asked for): kill the tree
        runstate.kill_tree(pid)
        deadline = time.time() + 15.0
        while runstate.pid_alive(pid) and time.time() < deadline:
            time.sleep(0.5)

    # the trainer usually writes its own final status; this is the backstop
    # (only if it hasn't already recorded a final state — e.g. 'failed' from
    # a crash, which is preserved above rather than overwritten to 'stopped')
    cur = runstate.read_status(rd)
    if cur.get("state") not in runstate.FINAL_STATES:
        runstate.write_status(rd, state="stopped",
                              error=cur.get("error") or "stopped by the GUI")
    return {"run_id": rd.name, "pid": pid, "state": "stopped"}


def status(runs_dir: Optional[Path] = None, metrics_tail: int = 5) -> Dict[str, Any]:
    """The GUI's single status view: the newest run (active or final) plus
    environment info. Everything from disk — safe to call from any process."""
    runs = Path(runs_dir) if runs_dir is not None else default_runs_dir()
    out: Dict[str, Any] = {
        "active": False, "state": None, "run_id": None, "pid": None,
        "round": None, "rounds_planned": None, "heartbeat_age_s": None,
        "error": None, "stop_requested": False,
        "metrics_tail": [], "checkpoints": [], "cuda_available": cuda_available(),
    }
    runs_list = runstate.list_runs(runs)
    if not runs_list:
        return out
    rd = runs / runs_list[0]
    st = runstate.reconcile(rd)
    active = (st.get("state") in runstate.ACTIVE_STATES
              and runstate.pid_alive(st.get("pid")))
    out.update({
        "active": active,
        "state": st.get("state"),
        "run_id": rd.name,
        "pid": st.get("pid"),
        "round": st.get("round"),
        "rounds_planned": st.get("rounds_planned"),
        "heartbeat_age_s": runstate.heartbeat_age_s(rd),
        "error": st.get("error"),
        "stop_requested": bool(st.get("stop_requested")),
        "metrics_tail": runstate.read_metrics(rd, metrics_tail),
        "checkpoints": runstate.checkpoints(rd),
        "started": st.get("started"),
    })
    return out


# ── environment ──────────────────────────────────────────────────────────────

_CUDA: Optional[bool] = None


def cuda_available() -> bool:
    """Cached torch CUDA probe (the server imports torch once; it does NO
    training — this only reports what the user's machine can do)."""
    global _CUDA
    if _CUDA is None:
        try:
            import torch
            _CUDA = bool(torch.cuda.is_available())
        except Exception:
            _CUDA = False
    return _CUDA
