"""One random-vs-random battle — the shared battle driver.

Two layers (D95):
  * ``play_battle_on(server_cfg, ...)`` — play ONE full battle on a server
    configuration that the CALLER already reserved (e.g. inside
    ``pool.battle_slot()``). This is the layer the self-play run engine
    (``control/selfplay.py``) uses: it owns the slot lifecycle and the
    worker threads, the driver just plays.
  * ``play_random_battle(rp, ...)`` — convenience wrapper: reserve a slot
    on a ``ResourcePool`` and play. This is what the Phase 0 gate and the
    replay-capture live test use.

The legality path is exercised end to end via the real ``ShowdownEnv`` /
``build_action_list`` / ``action_to_pe``. CPU-only (no GPU). ``net`` is
imported lazily so the pool package stays importable (and light) for
GUI/control-plane use without pulling in torch.
"""
from __future__ import annotations

from typing import Any, Callable, Optional, Tuple


def play_battle_on(server_cfg: Any, fmt: str = "gen9ou", seed: int = 0,
                   vocabs: Any = None, max_steps: int = 200,
                   on_finish: Optional[Callable[[Any], None]] = None
                   ) -> Tuple[bool, bool, int]:
    """Play ONE full battle (both sides uniformly random) on an ALREADY
    RESERVED server configuration.

    :param server_cfg: a ``ServerConfiguration`` (as yielded by
                       ``ResourcePool.battle_slot()``).
    :param on_finish: called with ``env.battle1`` right BEFORE teardown —
                      the hook for capture (``replays.make_record``). Must not
                      raise (a capture bug would otherwise kill the battle
                      result); callers wrap their own logic defensively.
    :return: ``(a1_won, a2_won, n_steps)``.
    """
    import logging
    import numpy as np
    from net.env import ShowdownEnv, StrongTeamBuilder, random_action, action_to_pe

    env = ShowdownEnv(battle_format=fmt, team=StrongTeamBuilder(fmt, seed=seed),
                      server_configuration=server_cfg, vocabs=vocabs,
                      log_level=logging.ERROR)
    a1, a2 = env.possible_agents
    n_steps = 0
    try:
        obs, _info = env.reset()
        for _ in range(max_steps):
            if env.battle1 is not None and env.battle1.finished:
                break
            actions = {a1: np.int64(-2), a2: np.int64(-2)}
            if env.agent1_to_move:
                act = random_action(env.battle1)
                if act is not None:
                    actions[a1] = np.int64(action_to_pe(act, env.battle1))
            if env.agent2_to_move:
                act = random_action(env.battle2)
                if act is not None:
                    actions[a2] = np.int64(action_to_pe(act, env.battle2))
            obs, rew, term, trunc, _info = env.step(actions)
            n_steps += 1
            if term[a1] or term[a2] or trunc[a1] or trunc[a2]:
                break
        if on_finish is not None:
            on_finish(env.battle1)
        return bool(env.battle1.won), bool(env.battle2.won), n_steps
    finally:
        env.close()


def play_random_battle(rp: Any, fmt: str = "gen9ou", seed: int = 0,
                       vocabs: Any = None, max_steps: int = 200,
                       on_finish: Optional[Callable[[Any], None]] = None
                       ) -> Tuple[bool, bool, int]:
    """Reserve a slot on a ``ResourcePool`` and play one full random battle
    (the Phase 0 gate / replay-capture test convenience wrapper)."""
    with rp.battle_slot() as server_cfg:
        return play_battle_on(server_cfg, fmt=fmt, seed=seed, vocabs=vocabs,
                              max_steps=max_steps, on_finish=on_finish)
