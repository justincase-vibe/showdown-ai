#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Showdown AI - training progress viewer.

Reads the JSON-lines stats file written by train.py (default: bot/checkpoints/stats.jsonl)
and renders a human-readable picture of how training is going:

  * a per-update trend table (critic calibration + stability)
  * best / worst / last of the main progress signal + a trend verdict
  * an ASCII sparkline of calib_sign_hit across updates
  * any loud warnings (forfeits / degenerate switching / illegal picks) from the run log

The honest caveat baked into the output: in both-sides self-play the WIN RATE is ~50%
by design and is NOT a progress signal. The real signals are the critic's game-read
(calib_sign_hit / calib_corr trending UP) and stability (clip_frac / entropy staying sane).

``render()`` returns the report as a string -- used by both this CLI and the
launcher GUI's "Progress" window.

Usage:
    python bot/progress.py
    python bot/progress.py --ckpt-dir bot/checkpoints --log bot/run_gpu.log --limit 200
"""
from __future__ import annotations

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _num(x):
    try:
        f = float(x)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def _fmt(x, nd=3):
    v = _num(x)
    return "-" if v is None else f"{v:.{nd}f}"


def load_stats(path: str):
    rows = []
    if not os.path.isfile(path):
        return rows
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def sparkline(values, lo=0.0, hi=1.0):
    """Map a sequence of [lo,hi] values to an ASCII bar sparkline (safe on Windows cp1252)."""
    ramp = " .:-=+*#%@"   # low -> high, pure ASCII
    out = []
    for v in values:
        f = _num(v)
        if f is None:
            out.append("?")
            continue
        idx = int(round((f - lo) / max(1e-9, (hi - lo)) * (len(ramp) - 1)))
        out.append(ramp[max(0, min(len(ramp) - 1, idx))])
    return "".join(out) if out else "(no data)"


def render(ckpt_dir: str, log_path: str, limit: int = 60) -> str:
    """Build the full progress report as a string (for the CLI and the GUI window)."""
    out: list[str] = []
    emit = out.append

    stats_path = os.path.join(ckpt_dir, "stats.jsonl")
    rows = load_stats(stats_path)
    if not rows:
        emit(f"No stats found at {stats_path}.")
        emit("Train first, or point the launcher at the right checkpoint dir.")
        return "\n".join(out)

    updates = [r for r in rows if "upstats" in r and r.get("update") is not None]
    updates.sort(key=lambda r: r.get("update", 0))
    emit(f"== Showdown AI progress  ({stats_path}) ==")
    emit(f"  {len(rows)} stat lines | {len(updates)} policy updates")
    emit("")

    if updates:
        shown = updates[-limit:]
        headers = ["upd", "battles", "sign_hit", "corr", "rmse", "entropy", "clip", "vloss", "plo"]
        keys = {
            "sign_hit": "calib_sign_hit", "corr": "calib_corr", "rmse": "calib_rmse",
            "entropy": "entropy", "clip": "clip_frac", "vloss": "value_loss", "plo": "policy_loss",
        }

        def val(r, h):
            if h == "upd":
                return str(r.get("update"))
            if h == "battles":
                return str(r.get("battles"))
            return _fmt((r.get("upstats") or {}).get(keys[h]))

        widths = {h: max(len(h), max(len(val(r, h)) for r in shown)) for h in headers}
        emit("  ".join(h.ljust(widths[h]) for h in headers))
        emit("  ".join("-" * widths[h] for h in headers))
        for r in shown:
            emit("  ".join(val(r, h).ljust(widths[h]) for h in headers))

        # --- summary of the main progress signal (calib_sign_hit) ---
        valid = [(i, _num((updates[i].get("upstats") or {}).get("calib_sign_hit")))
                 for i in range(len(updates))]
        valid = [(i, v) for i, v in valid if v is not None]
        if valid:
            vals = [v for _, v in valid]
            bi, bv = max(valid, key=lambda x: x[1])
            k = max(1, len(vals) // 5)
            early = sum(vals[:k]) / k
            late = sum(vals[-k:]) / k
            if late > early + 0.05:
                verdict = "IMPROVING  (recent reads better than early)"
            elif late < early - 0.05:
                verdict = "REGRESSING  (recent reads WORSE than early) -- investigate"
            else:
                verdict = "FLAT  (not clearly moving yet; needs more updates to judge)"
            emit("")
            emit("  calib_sign_hit -- the main progress signal (want it trending UP):")
            emit(f"    first={vals[0]:.3f}   best={bv:.3f} @ update {updates[bi].get('update')}   "
                 f"last={vals[-1]:.3f}   ({len(vals)} updates)")
            emit(f"    trend: {verdict}")
            emit(f"    sparkline (oldest -> newest):  {sparkline(vals)}")

        # --- quality flags from the stats lines ---
        battles_lines = [r for r in rows if r.get("tag") == "battles"]
        if battles_lines:
            q = battles_lines[-1].get("quality") or {}
            emit("")
            emit(f"  latest quality window: forfeits={q.get('forfeits', 0)}  "
                 f"sw_conc={q.get('sw_conc', 0)}  sw/turn={q.get('avg_sw_per_turn', 0)}  "
                 f"illegal_window={q.get('illegal_window', 0)}")

    # --- loud alarms from the run log ---
    emit(f"\n== Log alarms ({log_path}) ==")
    if os.path.isfile(log_path):
        alarms = []
        with open(log_path, "r", encoding="utf-8", errors="replace") as fh:
            for ln, line in enumerate(fh, 1):
                if "!!!" in line:
                    alarms.append((ln, line.strip()))
        if alarms:
            emit(f"  {len(alarms)} alarm(s) (showing last 40):")
            for ln, line in alarms[-40:]:
                emit(f"    L{ln}: {line}")
        else:
            emit("  none (no '!!!' lines)")
    else:
        emit("  (log file not found)")

    emit("\nNote: win rate is ~50% by design in self-play -- judge progress by "
         "calib_sign_hit / calib_corr, not wins.")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description="Showdown AI - training progress viewer")
    ap.add_argument("--ckpt-dir", default=os.path.join(HERE, "checkpoints"))
    ap.add_argument("--log", default=os.path.join(HERE, "run_gpu.log"))
    ap.add_argument("--limit", type=int, default=60, help="show the last N updates in the table")
    args = ap.parse_args()
    print(render(args.ckpt_dir, args.log, args.limit))


if __name__ == "__main__":
    main()
