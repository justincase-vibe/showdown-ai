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

__all__ = ["play_agent_battle_on", "policy_action", "DecisionGuard", "DecideFn"]


class DecisionGuard:
    """Counts CONSECUTIVE decision exceptions within ONE battle (D102 guard).

    A single decision failure (a malformed log line, a momentary stale state)
    still degrades to a passed turn — a live battle must never crash on one
    observation hiccup. But ``max_consecutive`` in a row is a SYSTEMATIC
    failure: the D102 device-mismatch bug made 100% of decisions raise, and
    the old ``policy_action`` swallowed every one of them, so a broken net
    silently pass-fested all 3200 battles instead of failing loud. A guard
    with ``max_consecutive=5`` raises on the 5th back-to-back error, the battle
    then FAILS (``control/selfplay`` increments its consecutive-error counter),
    and after ``max_consecutive_errors`` such battles the run is marked
    ``failed`` — a few battles lost, not thousands.

    One guard is shared by BOTH sides of a battle (one battle, one decision
    pipeline). ``on_success`` (a successful ``act``, even a legit pass) resets
    the streak, so a rare glitch on turn 3 does NOT combine with one on turn
    9 — only back-to-back errors count.
    """

    def __init__(self, max_consecutive: int,
                 log: Optional[Callable[[str], None]] = None) -> None:
        self.max_consecutive = max(1, int(max_consecutive))
        self.consecutive = 0
        self._log = log

    def on_success(self) -> None:
        """A decision that did NOT raise (a real action or a legit wait)."""
        self.consecutive = 0

    def on_error(self, exc: BaseException) -> Optional[Dict]:
        """A decision that raised. Degrades to ``None`` (pass this turn) for a
        transient glitch, or RAISES once the streak reaches the threshold."""
        self.consecutive += 1
        if self._log is not None:
            self._log(f"[driver] decision error ({type(exc).__name__}: {exc}) "
                      f"(consecutive={self.consecutive}/{self.max_consecutive})")
        if self.consecutive >= self.max_consecutive:
            raise RuntimeError(
                f"{self.consecutive} consecutive decision errors (last: "
                f"{type(exc).__name__}: {exc}) — the decision path is broken "
                f"(e.g. a net/agent device mismatch); failing the battle loud "
                f"instead of pass-festing") from exc
        return None


def policy_action(agent: Any, battle: Any, fmt: str, vocabs: Any,
                  greedy: bool = True,
                  log: Optional[Callable[[str], None]] = None,
                  guard: Optional["DecisionGuard"] = None,
                  capture: Optional[Callable[[Any, Any], None]] = None) -> Optional[Dict]:
    """The current legal action for ``agent`` from ``battle``'s POV.

    ``None`` = pass/wait (no legal action this moment). Defensive by default:
    a single decision failure (malformed log, belief-pipeline hiccup, a momentary
    stale state) degrades to a PASSED turn with a warning — a live league
    battle must never crash on an observation glitch (the same discipline as
    ``net.human_play.CheckpointPlayer.choose_move``).

    If a ``guard`` (:class:`DecisionGuard`) is supplied it replaces the bare
    log-and-pass: a transient glitch still degrades to a pass, but a SYSTEMATIC
    failure (N back-to-back errors) raises so the run fails loud rather than
    silently pass-festing (the D102 device bug).

    ``capture`` (D103 step 2): optional ``(obs, decision) -> None`` called
    with the EXACT encoded observation + full ``Decision`` (action_index /
    logprob / value) right after a SUCCESSFUL decision with a real action —
    the league's experience-collection seam (a pass/wait is not captured: no
    action taken, no transition to learn from)."""
    from net.env import build_action_list, build_obs, my_side
    try:
        lines = [ln for ln in battle._build_replay_log().split("\n") if ln.strip()]
        side = my_side(battle, lines)
        obs = build_obs(battle, fmt, vocabs, side)
        acts = build_action_list(battle)
        d = agent.act(obs, acts, deterministic=greedy)
    except Exception as exc:  # noqa: BLE001 — see docstring
        if guard is not None:
            return guard.on_error(exc)
        if log is not None:
            log(f"[driver] decision error ({type(exc).__name__}: {exc}) - passing the turn")
        return None
    if guard is not None:
        guard.on_success()
    if capture is not None and d.action is not None:
        capture(obs, d)
    return d.action


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
                         env_factory: Optional[Callable[[], Any]] = None,
                         max_decision_errors: Optional[int] = None,
                         capture_a: Optional[Callable[[Any, Any], None]] = None,
                         capture_b: Optional[Callable[[Any, Any], None]] = None
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
    :param max_decision_errors: OPTIONAL (D102 guard) — the number of
                   CONSECUTIVE decision exceptions tolerated before the battle
                   FAILS LOUD. ``None`` = legacy behaviour (every decision
                   error degrades to a passed turn). When set, a single glitch
                   still degrades to a pass, but ``N`` back-to-back errors
                   raise, so a systematically-broken decision path (e.g. the
                   D102 net/agent device mismatch) fails the run instead of
                   silently pass-festing all ``n_battles``.
    :param capture_a / capture_b: OPTIONAL (D103 step 2) — ``(
                   obs, decision) -> None`` called per real decision of each
                   side: the league's experience-collection seam (the engine
                   wires each side to its bot's :class:`TrajectoryCollector`).
                   Only the PRODUCTION path captures: a ``decide=`` override
                   (test injection) bypasses capture by design.
    :return: ``(a1_won, a2_won, n_steps)``.
    """
    import logging
    from net.env import ShowdownEnv, StrongTeamBuilder

    make_env = env_factory or (lambda: ShowdownEnv(
        battle_format=fmt, team=StrongTeamBuilder(fmt, seed=seed),
        server_configuration=server_cfg, vocabs=vocabs, log_level=logging.ERROR))
    # ONE guard shared by both sides of THIS battle (one decision pipeline).
    guard = (DecisionGuard(max_decision_errors, log)
             if max_decision_errors is not None else None)
    env = make_env()
    try:
        _obs, _info = env.reset()

        def da(battle: Any) -> Any:
            if decide is not None:
                return decide(agent_a, battle, fmt, vocabs, greedy)
            return policy_action(agent_a, battle, fmt, vocabs, greedy, log, guard,
                                 capture=capture_a)

        def db(battle: Any) -> Any:
            if decide is not None:
                return decide(agent_b, battle, fmt, vocabs, greedy)
            return policy_action(agent_b, battle, fmt, vocabs, greedy, log, guard,
                                 capture=capture_b)

        n_steps = _step_loop(env, da, db, max_steps)
        if on_finish is not None:
            on_finish(env.battle1)
        return bool(env.battle1.won), bool(env.battle2.won), n_steps
    finally:
        env.close()
