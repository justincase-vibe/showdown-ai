"""P4.3 — the BC / imitation (behavioral cloning) SUBPROCESS entry (D87).

Launched detached by ``gui.launcher`` exactly like ``gui.trainer_main``:

    python -m gui.bc_main --config <run-dir>/config.json --run-dir <run-dir>

It speaks the SAME ``gui.runstate`` protocol as the self-play entry, so the
dashboard, the WebSocket stream, the stop button and the run history all work
UNCHANGED for both modes (the "no code written twice" rule — one protocol,
two entries). What it owns:

  * GPU guard — ``device="cuda"`` without CUDA fails LOUD at start (same
    contract as ``trainer_main``: never a half-trained CPU surprise);
  * graceful stop — the D86 hooks: ``stop.flag`` is checked between battle
    replays and between epochs (never mid-battle / mid-epoch);
  * progress — a heartbeat per replayed battle and per epoch (the dashboard's
    liveness + the replay/epoch counter the user was waiting for);
  * the ONE final metrics row = the S4.3 verdict (the "always-switch" gate +
    the D82 first-use/repeat leak gate) with the key held-out numbers — the
    dashboard's metrics tail shows it, no separate "verdict view" needed;
  * the checkpoint into ``<run-dir>/checkpoints`` (``runstate.checkpoints``
    lists it; the file download endpoint serves it);
  * exit 0 = completed/stopped, exit 1 = failed.

The config is a FLAT dict (mirrors ``net.bc._cli`` args): n_battles, fixture,
max_decisions, heldout_frac, epochs, batch_size, lr, device, seed, ckpt_name.
``out_dir`` is NOT a config field — it is ALWAYS the run dir (clean per-run
separation, exactly like the self-play entry forces ``checkpoint_dir``).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _resolve_fixture(cfg: dict, run_dir: Path) -> str:
    """D96 — resolve the BC fixture path. A raw ``fixture`` is used as-is; a
    ``replay_source`` is MATERIALIZED from the replay library into
    ``<run_dir>/fixture.jsonl`` (the "one-click imitation" source — the file
    is drop-in for ``bc.run(fixture=)``). Pure (no torch / no subprocess) so
    it is unit-testable in isolation. Raises ``ValueError`` if neither source
    is usable (the caller turns that into a clean FAIL, not a crash)."""
    fixture = cfg.get("fixture")
    replay_source = cfg.get("replay_source")
    if replay_source is not None:
        from replays import ReplayLibrary
        lib = ReplayLibrary(replay_source.get("library_root"))
        info = lib.materialize_fixture(
            int(replay_source["n"]), key=replay_source.get("key", "elo"),
            out_path=str(run_dir / "fixture.jsonl"))
        _log(f"materialized {info['n']} replay(s) from {info['source']} "
             f"(truncated={info['truncated']}) -> {info['path']}")
        return info["path"]
    if fixture:
        return str(fixture)
    raise ValueError("config has no 'fixture' and no valid 'replay_source'")


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="gui.bc_main")
    ap.add_argument("--config", required=True, help="path to config.json (the BC config)")
    ap.add_argument("--run-dir", required=True, help="the runs/<run-id> dir")
    args = ap.parse_args(argv)

    rd = Path(args.run_dir).resolve()
    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    from gui import runstate
    runstate.harden_stdio()   # a non-ASCII log char must never kill the run on cp1252
    import torch

    device = str(cfg.get("device", "cpu") or "cpu")

    # ── GPU guard (same contract as trainer_main) ─────────────────────
    if device.startswith("cuda") and not torch.cuda.is_available():
        msg = "device='cuda' requested but torch.cuda.is_available() is False"
        runstate.write_status(rd, state="failed", error=msg, exit_code=1)
        _log(f"FATAL: {msg}")
        return 1

    runstate.write_status(rd, state="running", started=runstate.now_iso())
    runstate.write_heartbeat(rd, 0, 0, "bc-start")
    _log(f"BC start run={rd.name} n_battles={cfg.get('n_battles')} "
         f"fixture={cfg.get('fixture')} device={device} epochs={cfg.get('epochs')} "
         f"batch={cfg.get('batch_size')} lr={cfg.get('lr')}")

    from net import bc

    def stop_check() -> bool:
        return runstate.stop_requested(rd)

    def heartbeat(info: Dict) -> None:
        # Map the D86 hook events onto the runstate heartbeat schema
        # {ts, round, battle, note}. round = epoch (0 during replay),
        # battle = the replayed-battle counter.
        try:
            if info.get("phase") == "replay":
                runstate.write_heartbeat(
                    rd, 0, int(info.get("battle", 0)),
                    f"replay {info.get('battle')}/{info.get('total')}")
            else:
                runstate.write_heartbeat(
                    rd, int(info.get("epoch", 0)), 0,
                    f"epoch {info.get('epoch')}/{info.get('total_epochs')} "
                    f"top1={info.get('top1')}")
        except Exception:  # noqa: BLE001 — a heartbeat write must never kill the run
            pass

    # ── D96: resolve the fixture (materialize from the library if requested) ─
    try:
        fixture = _resolve_fixture(cfg, rd)
    except Exception as exc:  # noqa: BLE001 — a bad source is a clean FAIL, not a crash
        runstate.write_status(rd, state="failed",
                              error=f"resolve fixture: {type(exc).__name__}: {exc}",
                              exit_code=1)
        _log(f"FATAL: could not resolve the fixture: {exc}")
        return 1

    t0 = time.time()
    try:
        res = bc.run(
            n_battles=int(cfg.get("n_battles", 50)),
            fixture=str(fixture),
            max_decisions=int(cfg.get("max_decisions", 40)),
            heldout_frac=float(cfg.get("heldout_frac", 0.2)),
            epochs=int(cfg.get("epochs", 15)),
            batch_size=int(cfg.get("batch_size", 24)),
            lr=float(cfg.get("lr", 3e-3)),
            device=device,
            seed=int(cfg.get("seed", 0)),
            out_dir=str(rd / "checkpoints"),
            ckpt_name=cfg.get("ckpt_name") or None,  # None -> bc.run generates date_bc_Nb.pt
            log=_log,
            stop_check=stop_check,
            heartbeat=heartbeat,
        )
    except Exception as exc:  # noqa: BLE001 — any crash must be a clean FAIL
        runstate.write_status(rd, state="failed",
                              error=f"{type(exc).__name__}: {exc}", exit_code=1)
        _log(f"FATAL: {type(exc).__name__}: {exc}")
        return 1

    # ── the ONE final metrics row: the verdict + key numbers ───────────
    stopped = bool(res.get("stopped"))
    verdict = res.get("verdict", {}) or {}
    final = res.get("final_metrics", {}) or {}
    split = res.get("split", {}) or {}
    row: Dict = {
        "mode": "bc",
        "wall_s": round(time.time() - t0, 1),
        "n_train_samples": split.get("n_train_samples"),
        "n_heldout_samples": split.get("n_heldout_samples"),
        "human_switch_rate": split.get("human_switch_rate"),
        "best_top1": res.get("best_top1"),
        "verdict_pass": verdict.get("pass"),
        "verdict_failures": verdict.get("failures"),
        "stopped": stopped,
        "top1_acc": final.get("top1_acc"),
        "move_acc": final.get("move_acc"),
        "switch_acc": final.get("switch_acc"),
        "pred_switch_rate": final.get("pred_switch_rate"),
        "first_use_move_acc": final.get("first_use_move_acc"),
        "repeat_move_acc": final.get("repeat_move_acc"),
    }
    runstate.append_metric(rd, row)

    state = "stopped" if stopped else "completed"
    runstate.write_status(rd, state=state, exit_code=0,
                          error=(str(res.get("error")) if stopped else None))
    runstate.write_heartbeat(rd, 0, 0, state)
    _log(f"{state}: verdict={'PASS' if verdict.get('pass') else 'FAIL/n-a'} "
         f"best_top1={res.get('best_top1')} "
         f"ckpt={res.get('checkpoint', '(none — stopped during replay)')} "
         f"wall={time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
