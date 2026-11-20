"""Reproduce the spurious 'forfeit' bug under the REAL failing conditions
(policy, 4 envs, mixed gen7ou+gen7randombattle, --opponent live), while capturing
the exact stack trace that creates each ForfeitBattleOrder.
"""
from __future__ import annotations

import logging
import sys
import traceback

sys.path.insert(0, "bot")

# ---- apply monkey-patches BEFORE train runs ----
import poke_env.player.battle_order as _bo

_orig_init = _bo.ForfeitBattleOrder.__init__


def _traced_init(self, *a, **k):
    tb = "".join(traceback.format_stack(limit=14)[-14:])
    logging.getLogger("dbg").warning("=== FORFEIT ORDER CREATED ===\n%s", tb)
    _orig_init(self, *a, **k)


_bo.ForfeitBattleOrder.__init__ = _traced_init

from env import ShowdownEnv  # noqa: E402


def _state(env) -> str:
    def sig(b):
        try:
            alive = sum(1 for p in (b.team or {}).values() if not p.fainted)
            return (f"turn={b.turn} alive={alive} fin={bool(b.finished)} "
                    f"tag={getattr(b, 'battle_tag', '?')} id={id(b)}")
        except Exception:
            return "n/a"
    return (f"b1[{sig(env.battle1)}] b2[{sig(env.battle2)}] "
            f"a1_to_move={env.agent1_to_move} a2_to_move={env.agent2_to_move}")


_orig_step = ShowdownEnv.step


def _step(self, actions):
    logging.getLogger("dbg").info("STEP IN  %s | actions=%s", _state(self),
                                  {k: (int(v) if hasattr(v, "item") else v) for k, v in actions.items()})
    out = _orig_step(self, actions)
    logging.getLogger("dbg").info("STEP OUT %s", _state(self))
    return out


ShowdownEnv.step = _step

_orig_reset = ShowdownEnv.reset


def _reset(self, seed=None, options=None):
    logging.getLogger("dbg").info("RESET IN  %s", _state(self))
    out = _orig_reset(self, seed, options)
    logging.getLogger("dbg").info("RESET OUT %s", _state(self))
    return out


ShowdownEnv.reset = _reset

# ---- run the real train.py ----
import train  # noqa: E402

if __name__ == "__main__":
    # failing config: 4 envs, mixed formats, live opponent, small net (fast), 10 battles
    sys.argv = [
        "train.py",
        "--formats", "gen7ou,gen7randombattle",
        "--weights", "0.5,0.5",
        "--num-envs", "4",
        "--opponent", "live",
        "--device", "cpu",
        "--hidden", "128,128",
        "--max-battles", "10",
        "--buffer-size", "128",
        "--server-ports", "8000",
        "--ckpt-dir", "bot/checkpoints/repro",
    ]
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    train.main()
