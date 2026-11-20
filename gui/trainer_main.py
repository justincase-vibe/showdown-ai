"""P4.3 — the training SUBPROCESS entry (launched detached by gui.launcher).

Usage:
    python -m gui.trainer_main --config <run-dir>/config.json --run-dir <run-dir>

Owns the whole run lifecycle:
  * GPU guard — ``device="cuda"`` without CUDA fails LOUD at start (never a
    half-trained CPU surprise);
  * net init — random (default) or resume via ``PokeNet.load`` (the
    vocab/layout-verified fresh-process load, D77 open item resolved);
  * the round loop with per-battle heartbeats + the graceful-stop check
    (``stop.flag``) BETWEEN battles (D80: the flag is checked before every
    battle, not only at round boundaries — with 50 battles/round a
    round-boundary-only check would turn 'graceful' into 'force-kill after
    the grace period') — a stop never abandons a half-finished battle's
    state mid-update:
  * metrics streaming (one JSON line per round to metrics.jsonl) + a
    non-finite-metric guard (a NaN loss is a FAIL, not a silent checkpoint);
  * checkpointing into the run dir + a loud final status (exit 0 =
    completed/stopped, exit 1 = failed).

It writes EVERYTHING to the run dir (gui.runstate protocol) — the GUI never
needs shared memory with this process.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional


def _finite(row: Dict[str, Any]) -> None:
    """A non-finite loss/metric means the update diverged — fail the run
    LOUDLY (a NaN checkpoint would poison resume)."""
    for k in ("loss", "policy", "value", "entropy", "frac_clipped", "kl_approx"):
        v = row.get(k)
        if v is None:
            continue
        if not math.isfinite(float(v)):
            raise RuntimeError(f"non-finite metric {k}={v!r} in round {row.get('round')!r}")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="gui.trainer_main")
    ap.add_argument("--config", required=True, help="path to config.json (a TrainConfig)")
    ap.add_argument("--run-dir", required=True, help="the runs/<run-id> dir")
    args = ap.parse_args(argv)

    rd = Path(args.run_dir).resolve()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    from gui import runstate
    runstate.harden_stdio()   # a non-ASCII log char must never kill the run on cp1252
    from net.trainer import SelfPlayTrainer, TrainConfig
    import torch

    tc = TrainConfig.from_dict(cfg)
    # every run checkpoints into its own run dir (clean separation; the GUI
    # lists per-run checkpoints) — without rewriting the submitted config.
    tc.checkpoint_dir = str(rd / "checkpoints")

    def log(msg: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log(f"start run={rd.name} fmt={tc.battle_format} device={tc.device} "
        f"battles/round={tc.battles_per_round} rounds={tc.n_rounds} "
        f"resume={tc.resume_from!r}")
    runstate.write_status(rd, state="running", started=runstate.now_iso())

    # ── GPU guard ────────────────────────────────────────────────────────
    if tc.device.startswith("cuda"):
        if not torch.cuda.is_available():
            msg = "device='cuda' requested but torch.cuda.is_available() is False"
            runstate.write_status(rd, state="failed", error=msg, exit_code=1)
            log(f"FATAL: {msg}")
            return 1
    device = tc.device

    # ── net init (random or verified resume) ─────────────────────────────
    try:
        from net.vocab_db import build_vocabs
        from net.agent import PokeAgent, make_random_net

        vocabs = build_vocabs()
        start_round = 0
        if tc.resume_from:
            from net.model import PokeNet
            # THE ONE load path (D103 step 1): format sniffing lives in
            # PokeNet.unwrap_payload; we keep the outer payload for "round".
            payload = torch.load(tc.resume_from, map_location="cpu", weights_only=True)
            net = PokeNet.load(PokeNet.unwrap_payload(payload), vocabs, device=device)  # verifies vocab + layout
            start_round = int(payload.get("round", 0))
            log(f"resumed from {tc.resume_from} (round {start_round})")
        else:
            net = make_random_net(vocabs, device)
        trainer = SelfPlayTrainer(net, tc, agent=PokeAgent(net, seed=tc.seed, device=device))
        trainer.round = start_round
    except Exception as exc:  # noqa: BLE001 — init failures must be a clean FAIL
        runstate.write_status(rd, state="failed",
                              error=f"init: {type(exc).__name__}: {exc}", exit_code=1)
        log(f"FATAL (init): {type(exc).__name__}: {exc}")
        return 1

    rounds_planned = start_round + tc.n_rounds
    runstate.write_status(rd, rounds_planned=rounds_planned, round=start_round)

    def hb(round_i: int, battle: int, note: str = "") -> None:
        runstate.write_heartbeat(rd, round_i, battle, note)

    def battle_hook(battle: int) -> None:
        try:
            hb(trainer.round, int(battle), "battle")
        except Exception:  # noqa: BLE001 — a heartbeat write must never kill the run
            pass

    t0 = time.time()
    state_final = "completed"
    try:
        for _ in range(tc.n_rounds):
            if runstate.stop_requested(rd):
                state_final = "stopped"
                break
            runstate.write_status(rd, round=trainer.round)
            row: Dict[str, Any] = trainer.train_round(
                battle_hook=battle_hook, stop_check=lambda: runstate.stop_requested(rd))
            row["wall_s"] = round(time.time() - t0, 1)
            _finite(row)
            runstate.append_metric(rd, row)
            hb(row.get("round", trainer.round), tc.battles_per_round, "round-done")
            log(f"round {row.get('round')}: battles={row.get('battles')} "
                f"transitions={row.get('transitions')} "
                f"loss={row.get('loss')} wins={row.get('a1_wins')} "
                f"losses={row.get('a1_losses')} draws={row.get('draws')} "
                f"stopped_early={row.get('stopped_early')} "
                f"ckpt={row.get('checkpoint')}")
            if row.get("stopped_early"):
                # graceful stop landed between battles of THIS round: the round
                # is complete (update + checkpoint done) — exit now, clean
                state_final = "stopped"
                break
    except Exception as exc:  # noqa: BLE001
        runstate.write_status(rd, state="failed", round=trainer.round,
                              error=f"{type(exc).__name__}: {exc}", exit_code=1)
        log(f"FATAL: {type(exc).__name__}: {exc}")
        return 1
    finally:
        # C — reap the long-lived Node calc workers on the way out (covers both
        # return paths; atexit in damage.worker is the backstop). Never raises.
        try:
            from damage import worker as _worker
            _worker.shutdown_pool()
        except Exception:  # noqa: BLE001
            pass

    runstate.write_status(rd, state=state_final, round=trainer.round, exit_code=0)
    log(f"{state_final}: rounds={trainer.round} (planned {rounds_planned}) "
        f"wall={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
