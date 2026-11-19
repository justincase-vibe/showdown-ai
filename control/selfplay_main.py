"""Self-play RUN SUBPROCESS entry (launched detached by the GUI, mirroring
``gui.trainer_main``'s contract).

Usage:
    python -m control.selfplay_main --config <run-dir>/config.json --run-dir <run-dir>

Owns the whole run lifecycle:
  * writes ``pid`` + status into the runstate protocol immediately (a
    reconcile() can then tell a dead run from a live one);
  * drives ``control.selfplay.SelfPlayRun`` (pool start -> N threaded battles
    -> replay capture -> final status);
  * exit codes: 0 = completed OR user-stopped (a stop is a success, not a
    failure), 1 = failed (pool down, N consecutive battle errors, crash);
  * ALWAYS tears the pool down on exit (only the servers this run spawned —
    the pool reuses, never kills, pre-existing ones like the user's :8000).

Everything observable is in the run dir (gui.runstate protocol) — the GUI
never needs shared memory with this process.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="control.selfplay_main")
    ap.add_argument("--config", required=True, help="path to config.json (a SelfPlayConfig)")
    ap.add_argument("--run-dir", required=True, help="the runs/<run-id> dir")
    args = ap.parse_args(argv)

    from gui import runstate
    runstate.harden_stdio()   # a non-ASCII log char must never kill the run on cp1252
    from control.selfplay import SelfPlayConfig, SelfPlayRun

    rd = Path(args.run_dir).resolve()
    cfg_path = Path(args.config).resolve()
    cfg = SelfPlayConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    runstate.write_status(rd, state="starting", pid=os.getpid())
    log(f"start run={rd.name} fmt={cfg.battle_format} battles={cfg.n_battles} "
        f"servers={cfg.n_servers}x{cfg.per_server} port_base={cfg.port_base}")

    run = SelfPlayRun(cfg, run_dir=rd, log=log)
    try:
        run.start()
        summary = run.wait()
    except Exception as exc:  # noqa: BLE001 — any init/start failure is a clean FAIL
        runstate.write_status(rd, state="failed",
                              error=f"{type(exc).__name__}: {exc}", exit_code=1)
        log(f"FATAL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        run.shutdown()

    if summary.get("state") == "failed":
        runstate.write_status(rd, exit_code=1)
        log(f"FAILED: {summary}")
        return 1
    runstate.write_status(rd, exit_code=0)
    log(f"FINAL: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
