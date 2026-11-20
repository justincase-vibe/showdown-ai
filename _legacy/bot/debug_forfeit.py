"""Focused reproduction of the spurious 'forfeit' replays.

Monkey-patches poke_env's ForfeitBattleOrder so that the *exact* code path that
creates a forfeit is captured with a full stack trace + the battle state. Then runs
a small batch of self-contained battles (random vs random) and reports every forfeit
it triggers. This isolates the harness bug from policy behaviour.
"""
from __future__ import annotations

import logging
import sys
import traceback

sys.path.insert(0, "bot")

import numpy as np
import poke_env.player.battle_order as _bo
from poke_env.environment.env import PokeEnv
from env import ShowdownEnv, TraineeEnv, UNIFIED_ACTION_SIZE
from train import _server_config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("dbg")

# --------------------------------------------------------------------------- #
# 1) Catch the forfeit at the source
# --------------------------------------------------------------------------- #
_orig_init = _bo.ForfeitBattleOrder.__init__


def _traced_init(self, *a, **k):
    tb = traceback.format_stack(limit=12)
    log.warning("=== FORFEIT ORDER CREATED ===\n%s", "".join(tb[-12:]))
    _orig_init(self, *a, **k)


_bo.ForfeitBattleOrder.__init__ = _traced_init


# --------------------------------------------------------------------------- #
# 2) Instrument the env's step/reset to see agent1_to_move + battle identity
# --------------------------------------------------------------------------- #
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
    log.info("STEP IN  %s | actions=%s", _state(self), actions)
    out = _orig_step(self, actions)
    log.info("STEP OUT %s", _state(self))
    return out


ShowdownEnv.step = _step

_orig_reset = ShowdownEnv.reset


def _reset(self, seed=None, options=None):
    log.info("RESET IN %s", _state(self))
    out = _orig_reset(self, seed, options)
    log.info("RESET OUT %s", _state(self))
    return out


ShowdownEnv.reset = _reset


# --------------------------------------------------------------------------- #
# 3) Run a small batch
# --------------------------------------------------------------------------- #
def make_env(idx: int) -> TraineeEnv:
    env = ShowdownEnv(
        battle_format="gen7randombattle",
        team=None,
        server_configuration=_server_config(8000, "localhost"),
        strict=False,
        save_replays=False,
        log_level=logging.WARNING,
    )
    return TraineeEnv(env, env_id=idx, fmt="gen7randombattle")


def run_batch(num_battles: int, num_envs: int = 4):
    envs = [make_env(i) for i in range(num_envs)]
    skipped = set()
    done_count = 0
    forfeits = 0
    try:
        # initial reset
        for e in envs:
            e.reset()
        while done_count < num_battles:
            for e in envs:
                if e.id in skipped:
                    continue
                if e.done:
                    # re-arm
                    try:
                        e.reset()
                    except Exception as exc:
                        log.error("env %d reset failed: %s", e.id, str(exc)[:160])
                        skipped.add(e.id)
                        continue
                else:
                    import time as _t
                    _t.sleep(0.25)  # simulate slow policy inference -> widen the POKE_LOOP race window
                    try:
                        e.step("random", "random")
                    except Exception as exc:
                        log.error("env %d step crashed: %s", e.id, str(exc)[:200])
                        skipped.add(e.id)
                        continue
                if e.done:
                    b = e.env.battle1
                    team_alive = sum(1 for p in b.team.values() if not p.fainted)
                    opp_alive = sum(1 for p in b.opponent_team.values() if not p.fainted)
                    if (b.won and opp_alive > 0) or (b.lost and team_alive > 0):
                        forfeits += 1
                        log.warning("env %d BATTLE looks like a FORFEIT: team_alive=%d opp_alive=%d won=%s lost=%s tag=%s",
                                    e.id, team_alive, opp_alive, b.won, b.lost, getattr(b, "battle_tag", "?"))
                    done_count += 1
                    log.info("env %d battle #%d done: %s tag=%s", e.id, done_count,
                             "win" if b.won else ("loss" if b.lost else "?"), getattr(b, "battle_tag", "?"))
            if len(skipped) >= len(envs):
                log.error("all envs dropped")
                break
    finally:
        for e in envs:
            try:
                e.env.close()
            except Exception:
                pass
    log.info("DONE. battles=%d suspected_forfeits=%d skipped=%s", num_battles, forfeits, sorted(skipped))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    run_batch(n)
