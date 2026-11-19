"""League RUN SUBPROCESS entry (launched detached by the GUI, mirroring
``control.selfplay_main``'s contract).

Usage:
    python -m league.league_main --config <run-dir>/config.json --run-dir <run-dir>

Owns the whole run lifecycle:
  * writes ``pid`` + status into the runstate protocol immediately;
  * drives ``league.engine.LeagueRun`` (roster -> pairings -> checkpoint-vs-
    checkpoint battles -> ELO updates -> final status);
  * exit codes: 0 = completed OR user-stopped (a stop is a success, not a
    failure), 1 = failed (pool down, N consecutive battle errors, crash);
  * ALWAYS tears the pool down on exit (only the servers this run spawned —
    the pool reuses, never kills, pre-existing ones like the user's :8000).

The roster (``league/bots.json`` by default) is the league's persistent state:
add bots with ``LeagueRun.add_bot`` (or edit the JSON) before launching — the
config carries no bot list, so the SAME config re-fights the CURRENT roster.
Everything observable is in the run dir (gui.runstate protocol).
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
    ap = argparse.ArgumentParser(prog="league.league_main")
    ap.add_argument("--config", required=True, help="path to config.json (a LeagueConfig)")
    ap.add_argument("--run-dir", required=True, help="the runs/<run-id> dir")
    args = ap.parse_args(argv)

    from gui import runstate
    runstate.harden_stdio()   # a non-ASCII log char must never kill the run on cp1252
    from league.engine import LeagueConfig, LeagueRun

    rd = Path(args.run_dir).resolve()
    cfg_path = Path(args.config).resolve()
    cfg = LeagueConfig.from_dict(json.loads(cfg_path.read_text(encoding="utf-8")))

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    runstate.write_status(rd, state="starting", pid=os.getpid())
    log(f"start run={rd.name} fmt={cfg.battle_format} pairing={cfg.pairing} "
        f"best_of={cfg.best_of} greedy={cfg.greedy} device={cfg.device} "
        f"servers={cfg.n_servers}x{cfg.per_server} port_base={cfg.port_base}")

    run = LeagueRun(cfg, run_dir=rd, log=log)
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
