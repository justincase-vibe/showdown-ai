"""League battle driver — ONE checkpoint-vs-checkpoint battle.

The counterpart of ``pool.battle.play_battle_on`` (random vs random): the SAME
``ShowdownEnv`` step-loop and the SAME battle-driver signature, but each side
is decided by a ``net.agent.PokeAgent`` over a loaded ``PokeNet`` — the exact
live-play decision path (``net.human_play.decide_order``), so what the league
plays is what the checkpoint does:

    build_obs (POV-correct) -> build_action_list -> PokeAgent.act -> action_to_pe

Thread model (D100): the NET is shared across concurrent battles (read-only
forward under ``no_grad`` in eval mode — safe), but each battle gets its OWN
``PokeAgent`` (fresh ``torch.Generator``), so stochastic sampling never shares
RNG state across worker threads. The engine (``league.engine``) builds the
per-battle agents; this module just plays.

CPU-first: import this module without torch (``net`` is pulled lazily, same
discipline as ``pool.battle``).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

#: the decision seam: (agent, battle, fmt, vocabs, greedy) -> action dict | None
#: (``None`` = pass/wait, no legal action this moment — the loop sends -2).
DecideFn = Callable[..., Any]

__all__ = ["play_agent_battle_on", "policy_action", "DecideFn"]


def policy_action(agent: Any, battle: Any, fmt: str, vocabs: Any,
                  greedy: bool = True,
                  log: Optional[Callable[[str], None]] = None) -> Optional[Dict]:
    """The current legal action for ``agent`` from ``battle``'s POV.

    ``None`` = pass/wait (no legal action this moment). Defensive by design:
    a decision failure (malformed log, belief-pipeline hiccup, a momentary
    stale state) degrades to a PASSED turn with a warning — a live league
    battle must never crash on an observation glitch (the same discipline as
    ``net.human_play.CheckpointPlayer.choose_move``).
    """
    from net.env import build_action_list, build_obs, my_side
    try:
        lines = [ln for ln in battle._build_replay_log().split("\n") if ln.strip()]
        side = my_side(battle, lines)
        obs = build_obs(battle, fmt, vocabs, side)
        acts = build_action_list(battle)
        d = agent.act(obs, acts, deterministic=greedy)
        return d.action
    except Exception as exc:  # noqa: BLE001 — see docstring
        if log is not None:
            log(f"[driver] decision error ({type(exc).__name__}: {exc}) - passing the turn")
        return None


def _step_loop(env: Any, decide_a: Callable[[Any], Any],
               decide_b: Callable[[Any], Any], max_steps: int) -> int:
    """The shared battle step-loop over an ALREADY-RESET env (mirrors
    ``pool.battle.play_battle_on``'s loop with policy decisions instead of
    ``random_action``). Extracted so the loop's scheduling is testable with a
    fake env — no server, no net. Returns the number of steps taken."""
    import numpy as np
    from net.env import action_to_pe

    a1, a2 = env.possible_agents
    n_steps = 0
    for _ in range(max_steps):
        if env.battle1 is not None and env.battle1.finished:
            break
        actions: Dict[Any, Any] = {a1: np.int64(-2), a2: np.int64(-2)}
        if env.agent1_to_move:
            act = decide_a(env.battle1)
            if act is not None:
                actions[a1] = np.int64(action_to_pe(act, env.battle1))
        if env.agent2_to_move:
            act = decide_b(env.battle2)
            if act is not None:
                actions[a2] = np.int64(action_to_pe(act, env.battle2))
        _obs, _rew, term, trunc, _info = env.step(actions)
        n_steps += 1
        if term[a1] or trunc[a1] or term[a2] or trunc[a2]:
            break
    return n_steps


def play_agent_battle_on(server_cfg: Any, *, agent_a: Any, agent_b: Any,
                         fmt: str = "gen9ou", seed: int = 0,
                         vocabs: Any = None, max_steps: int = 200,
                         greedy: bool = True,
                         decide: Optional[DecideFn] = None,
                         log: Optional[Callable[[str], None]] = None,
                         on_finish: Optional[Callable[[Any], None]] = None,
                         env_factory: Optional[Callable[[], Any]] = None
                         ) -> Tuple[bool, bool, int]:
    """Play ONE full battle with ``agent_a`` on p1 and ``agent_b`` on p2.

    Same contract as ``pool.battle.play_battle_on`` (the league's
    ``battle_driver_factory`` slot):

    :param server_cfg: a ``ServerConfiguration`` the CALLER already reserved.
    :param agent_a / agent_b: ready ``PokeAgent``s (ONE per side, built by
                              the engine — fresh RNG per battle).
    :param greedy: ``True`` = argmax (deterministic ELO); ``False`` = sample.
    :param decide: OPTIONAL decision-seam override
                   ``(agent, battle, fmt, vocabs, greedy) -> action | None``
                   (tests inject a fake; production uses :func:`policy_action`).
    :param on_finish: called with ``env.battle1`` right BEFORE teardown —
                      the capture hook (``replays.make_record``).
    :param env_factory: OPTIONAL ``() -> env`` override (tests inject a
                        scripted fake env; production builds a ``ShowdownEnv``).
    :return: ``(a1_won, a2_won, n_steps)``.
    """
    import logging
    from net.env import ShowdownEnv, StrongTeamBuilder

    make_env = env_factory or (lambda: ShowdownEnv(
        battle_format=fmt, team=StrongTeamBuilder(fmt, seed=seed),
        server_configuration=server_cfg, vocabs=vocabs, log_level=logging.ERROR))
    env = make_env()
    try:
        _obs, _info = env.reset()

        def da(battle: Any) -> Any:
            if decide is not None:
                return decide(agent_a, battle, fmt, vocabs, greedy)
            return policy_action(agent_a, battle, fmt, vocabs, greedy, log)

        def db(battle: Any) -> Any:
            if decide is not None:
                return decide(agent_b, battle, fmt, vocabs, greedy)
            return policy_action(agent_b, battle, fmt, vocabs, greedy, log)

        n_steps = _step_loop(env, da, db, max_steps)
        if on_finish is not None:
            on_finish(env.battle1)
        return bool(env.battle1.won), bool(env.battle2.won), n_steps
    finally:
        env.close()
