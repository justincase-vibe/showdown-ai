#!/usr/bin/env python
"""GPU self-play launcher with a watchdog + OS pop-up alerts (Windows).

Runs ``bot/train.py`` (the 1k-battle wide self-play command, PLAN.md \u00a79b) as a
subprocess, streams everything to ``bot/run_gpu.log``, and pops an OS message box
(with beeps) when something needs your attention:

  * ERROR   -- the run exited non-zero (shows the last log lines)
  * STALLED -- no new stats line for --stall-min minutes -> the run is killed
  * STOPPED -- you pressed Ctrl+C
  * DONE    -- clean exit (shows battles finished + last calibration metrics)

The pop-up is the "something went wrong" bell: start the run and walk away; the
desktop grabs your attention when the run ends, dies, or stalls.

Usage (from anywhere):
  python bot/run_gpu_watch.py                  # the 1k-battle GPU run (default)
  python bot/run_gpu_watch.py --no-popup       # print alerts to the console instead
  python bot/run_gpu_watch.py --stall-min 20   # change the stall threshold (default 15)

Extra args are APPENDED to the default train command (argparse last-one-wins), so
quick tests can override any of them, e.g.:
  python bot/run_gpu_watch.py --no-popup --formats gen7ou --num-envs 1 --max-battles 1 --device cpu
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TRAIN_PY = os.path.join(HERE, "train.py")
STATS_PATH = os.path.join(HERE, "checkpoints", "stats.jsonl")
LOG_PATH = os.path.join(HERE, "run_gpu.log")

# 5-server wide self-play from random init (~1 GB net, no imitation warm-start).
# CLI extras are appended after these, so a quick test can override any of them.
TRAIN_ARGS = [
    "--formats", "gen6ou,gen6ru,gen6uu,gen7ou,gen7ru,gen7uu,gen8ou,gen8ru,gen8uu,gen9ou,gen9ru,gen9uu",
    "--num-envs", "10",
    "--opponent", "live",
    "--buffer-size", "2048",
    "--minibatch", "512",
    "--ppo-epochs", "6",
    "--lr", "1e-4",
    "--max-battles", "5000",
    "--device", "cuda",
    "--hidden", "1024,2048,4096,4096,2048,1024",
    "--server-ports", "8000,8001,8002,8003,8004,8005,8006,8007,8008,8009",
    "--team-mode", "random",
    "--log-every", "20",
]


def server_ok(url: str, tries: int = 4, delay: float = 2.0) -> bool:
    """Poll the Showdown server until it answers (or we give up)."""
    import urllib.request
    for _ in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                return r.status == 200
        except Exception:
            time.sleep(delay)
    return False


def all_servers_ok(ports: list, tries: int = 8, delay: float = 2.0) -> list:
    """Check ALL given ports are up. Returns list of (port, ok) tuples."""
    results = []
    for port in ports:
        url = f"http://localhost:{port}/"
        ok = server_ok(url, tries=tries, delay=delay)
        results.append((port, ok))
    return results


def alert(kind: str, title: str, text: str, popup: bool) -> None:
    """Always print the alert; also raise an OS pop-up (Windows) when enabled."""
    print("\n" + "=" * 64)
    print(f"[{kind.upper()}] {title}")
    print(text)
    print("=" * 64, flush=True)
    if not popup or os.name != "nt":
        return
    try:
        import winsound
        for _ in range(3):
            winsound.Beep(660 if kind == "ok" else 440, 300)
    except Exception:
        pass
    try:
        import ctypes
        icons = {"ok": 0x40, "warn": 0x30, "error": 0x10}  # info / warning / error
        ctypes.windll.user32.MessageBoxW(0, f"{title}\n\n{text}", "Showdown AI self-play",
                                        icons.get(kind, 0x10))
    except Exception as e:  # never let pop-up machinery mask the real result
        print(f"(pop-up unavailable: {e})")


def tail(path: str, n_lines: int = 15) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "\n".join(f.readlines()[-n_lines:])
    except OSError:
        return "(no log available)"


def new_records(start_offset: int) -> list:
    """stats.jsonl lines appended AFTER ``start_offset`` (this run's records)."""
    recs = []
    if not os.path.exists(STATS_PATH):
        return recs
    with open(STATS_PATH, "r", encoding="utf-8") as f:
        f.seek(start_offset)
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return recs


def _done_line(path: str) -> str:
    """Last 'done. battles=N elapsed=Xs' line of the run log (fallback summary)."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if "done. battles=" in line:
                    return line.split("INFO", 1)[-1].strip()
    except OSError:
        pass
    return ""


def summarize(recs: list) -> str:
    if not recs:
        done = _done_line(LOG_PATH)
        return (f"{done}" if done
                else "(no stats lines written and no 'done' line in the log)")
    battles = max(int(r.get("battles", 0)) for r in recs)
    updates = [r for r in recs if r.get("tag") == "update"]
    lines = [f"battles finished: {battles}   |   PPO updates: {len(updates)}"]
    last = updates[-1] if updates else None
    if last:
        up = last.get("upstats", {})
        calib = [f"{k}={v:.3f}" for k, v in up.items()
                 if k.startswith("calib_") and k != "calib_n" and isinstance(v, (int, float))]
        if calib:
            lines.append("last calibration: " + "  ".join(calib))
    lines.append(f"log: {LOG_PATH}")
    return "\n".join(lines)


def _kill(proc: subprocess.Popen) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-popup", action="store_true",
                    help="print alerts to the console instead of OS pop-ups")
    ap.add_argument("--stall-min", type=float, default=15.0,
                    help="kill + alert if no new stats line for this many minutes (default 15)")
    ap.add_argument("--server-url", default="http://localhost:8000",
                    help="(legacy, single-server mode) -- see --server-ports")
    args, extra = ap.parse_known_args()
    popup = not args.no_popup

    # Determine which ports to check: use --server-ports from TRAIN_ARGS if present,
    # otherwise fall back to the legacy single --server-url.
    if "--server-ports" in TRAIN_ARGS:
        port_str = TRAIN_ARGS[TRAIN_ARGS.index("--server-ports") + 1]
        ports = [int(p.strip()) for p in port_str.split(",") if p.strip()]
    else:
        ports = [8000]

    results = all_servers_ok(ports)
    down = [(p, r) for p, r in results if not r]
    if down:
        down_str = ", ".join(f"port {p}" for p, _ in down)
        alert("error", f"{len(down)}/{len(ports)} servers are DOWN -- cannot start.",
              f"Down: {down_str}\n"
              f"Start all servers first:\n"
              f"  cd pokemon-showdown && start_servers.bat\n"
              f"or manually: set SHOWDOWN_PORT=<port> && node pokemon-showdown start",
              popup)
        sys.exit(3)
    print(f"All {len(ports)} servers up (ports {ports[0]}-{ports[-1]})")

    cmd = [sys.executable, TRAIN_PY] + TRAIN_ARGS + list(extra)
    stall_s = max(60.0, args.stall_min * 60.0)

    print(f"launching: {' '.join(cmd)}")
    print(f"log: {LOG_PATH} | stall watchdog: {stall_s / 60.0:.0f} min "
          f"| popups: {'off' if not popup else 'ON'}")

    orig_offset = os.path.getsize(STATS_PATH) if os.path.exists(STATS_PATH) else 0
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    started = time.time()
    last_activity = started
    stalled = False
    interrupted = False

    with open(LOG_PATH, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=logf, stderr=subprocess.STDOUT,
                                env=env)
        try:
            while proc.poll() is None:
                time.sleep(5)
                if os.path.exists(STATS_PATH):
                    size = os.path.getsize(STATS_PATH)
                    if size > orig_offset:
                        orig_offset = size
                        last_activity = time.time()
                if time.time() - last_activity > stall_s:
                    stalled = True
                    break
        except KeyboardInterrupt:
            interrupted = True

        if stalled:
            print(f"STALL: no stats progress for {stall_s / 60.0:.0f} min -- killing the run")
            _kill(proc)
        elif interrupted:
            print("Ctrl+C -- stopping the run")
            _kill(proc)
    rc = proc.returncode
    elapsed = time.time() - started
    recs = new_records(orig_offset)

    if interrupted:
        alert("warn", f"STOPPED by user (Ctrl+C) after {elapsed / 60.0:.1f} min",
              summarize(recs), popup)
        sys.exit(130)
    if stalled:
        alert("error", f"STALLED -- no stats progress for {stall_s / 60.0:.0f} min; run killed",
              tail(LOG_PATH), popup)
        sys.exit(2)
    if rc != 0:
        alert("error", f"ERROR -- self-play run failed (exit code {rc})",
              tail(LOG_PATH), popup)
        sys.exit(1)
    alert("ok", f"DONE -- self-play finished ({elapsed / 60.0:.1f} min)",
          summarize(recs), popup)
    sys.exit(0)


if __name__ == "__main__":
    main()
