"""P4.3.5 — the EVAL (bot vs random) SUBPROCESS entry (D92).

Launched detached by ``gui.launcher`` exactly like the other entries:

    python -m gui.eval_main --config <run-dir>/config.json --run-dir <run-dir>

What it does:
  * loads a PokeNet checkpoint (the user's "fight this model" request);
  * runs N battles of the bot vs a RANDOM opponent (local fork, no online);
  * writes a heartbeat per battle (the dashboard's liveness);
  * writes ONE final metrics row = the eval summary (win/loss, avg turns,
    move/switch counts);
  * exit 0 = completed/stopped, exit 1 = failed.

The config is a FLAT dict:
  ckpt_path, n_battles, device, seed, battle_format.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="gui.eval_main")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    rd = Path(args.run_dir).resolve()
    cfg = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))

    from gui import runstate
    runstate.harden_stdio()   # a non-ASCII log char must never kill the run on cp1252

    ckpt_path = str(cfg.get("ckpt_path", ""))
    n_battles = int(cfg.get("n_battles", 1))
    device = str(cfg.get("device", "cpu") or "cpu")
    seed = int(cfg.get("seed", 0))
    fmt = str(cfg.get("battle_format", "gen9ou") or "gen9ou")

    # ── GPU guard ─────────────────────────────────────────────────────
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            msg = "device='cuda' requested but torch.cuda.is_available() is False"
            runstate.write_status(rd, state="failed", error=msg, exit_code=1)
            _log(f"FATAL: {msg}")
            return 1

    runstate.write_status(rd, state="running", started=runstate.now_iso())
    runstate.write_heartbeat(rd, 0, 0, "eval-start")
    _log(f"EVAL start run={rd.name} ckpt={ckpt_path} n={n_battles} "
         f"device={device} fmt={fmt} seed={seed}")

    # ── load the checkpoint (THE ONE load path, D103 step 1) ──────────
    from net.vocab_db import build_vocabs
    from net.model import PokeNet
    from net.agent import PokeAgent

    try:
        vocabs = build_vocabs()
        _log(f"vocabs: {vocabs.sizes()}")
        net = PokeNet.load_file(ckpt_path, vocabs, device=device)
        net.eval()
        agent = PokeAgent(net, seed=seed, device=device)
        _log(f"agent ready ({sum(p.numel() for p in net.parameters())/1e6:.1f}M params)")
    except Exception as exc:
        runstate.write_status(rd, state="failed",
                              error=f"checkpoint load failed: {type(exc).__name__}: {exc}",
                              exit_code=1)
        _log(f"FATAL: {type(exc).__name__}: {exc}")
        return 1

    # ── run battles ────────────────────────────────────────────────────
    from net.trainer import play_and_collect

    t0 = time.time()
    wins = losses = draws = 0
    total_steps = 0
    total_illegal = 0
    battle_results = []

    try:
        for i in range(n_battles):
            if runstate.stop_requested(rd):
                _log(f"stop requested after {i}/{n_battles} battles")
                break
            runstate.write_heartbeat(rd, 0, i, f"battle {i+1}/{n_battles}")
            _log(f"battle {i+1}/{n_battles} ...")
            res = play_and_collect(agent, fmt=fmt, seed=seed + i)
            won = bool(res.get("a1_won"))
            lost = bool(res.get("a2_won"))
            drawn = bool(res.get("draw"))
            if won: wins += 1
            elif lost: losses += 1
            else: draws += 1
            total_steps += int(res.get("steps", 0))
            total_illegal += int(res.get("n_illegal", 0))
            battle_results.append({"won": won, "lost": lost, "draw": drawn,
                                   "steps": int(res.get("steps", 0))})
            _log(f"  result: {'WIN' if won else 'LOSS' if lost else 'DRAW'} "
                 f"steps={res.get('steps', 0)}")
    except Exception as exc:
        runstate.write_status(rd, state="failed",
                              error=f"battle {i+1} crashed: {type(exc).__name__}: {exc}",
                              exit_code=1)
        _log(f"FATAL: {type(exc).__name__}: {exc}")
        return 1

    # ── final metrics row ──────────────────────────────────────────────
    stopped = runstate.stop_requested(rd)
    n_done = wins + losses + draws
    row = {
        "mode": "eval",
        "wall_s": round(time.time() - t0, 1),
        "n_battles_planned": n_battles,
        "n_battles_done": n_done,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / n_done, 4) if n_done else None,
        "avg_turns": round(total_steps / n_done, 1) if n_done else None,
        "total_illegal": total_illegal,
        "ckpt_path": ckpt_path,
        "battle_results": battle_results,
        "stopped": stopped,
    }
    runstate.append_metric(rd, row)

    state = "stopped" if stopped else "completed"
    runstate.write_status(rd, state=state, exit_code=0)
    runstate.write_heartbeat(rd, 0, 0, state)
    _log(f"{state}: {wins}W/{losses}L/{draws}D avg={row['avg_turns']} turns "
         f"illegal={total_illegal} wall={time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
