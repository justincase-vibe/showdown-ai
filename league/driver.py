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

__all__ = ["play_agent_battle_on", "policy_action", "DecisionGuard", "DecideFn",
           "NoProgressTracker", "progress_signature"]


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
                  capture: Optional[Callable[[Any, Any], None]] = None,
                  side: Optional[str] = None) -> Optional[Dict]:
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
    action taken, no transition to learn from).

    ``side`` (2026-11-23 POV fix): the battle's p1/p2 identity of ``battle``,
    KNOWN by the caller. In self-play the env IS the source of truth
    (``ShowdownEnv.embed_battle``: battle1 -> "p1", battle2 -> "p2"), so
    ``play_agent_battle_on`` passes it explicitly and the observation is built
    with the correct POV — no species-guessing. ``None`` = fall back to
    :func:`net.env.my_side` (the human-vs-bot path, where the caller does not
    know the side a priori). A wrong POV here is not a cosmetic glitch: a p2
    agent seen as p1 gets the opponent's team as its own, its active resolves
    to 0 HP, and every rule that reads own state misfires (the 2026-11-23
    switch-standoff / draw bug)."""
    from net.env import build_action_list, build_obs, my_side
    try:
        if getattr(agent, "decides_from_battle", False):
            # a ctx-based brain (fitness.strategy_agent.StrategyAgent): its
            # policy consumes the belief's get_context() — built from the
            # BATTLE (the SAME pipeline build_obs uses), not the encoded obs.
            # Returns an action dict | None (pass/wait). Nothing is captured:
            # a scripted policy has no experience to collect (the D103
            # capture seam is for the learning nets only).
            # The explicit side (when known) is forwarded too — the ctx brain
            # must build its belief from the SAME POV the env uses.
            decide_kwargs: Dict[str, Any] = {}
            if side is not None:
                decide_kwargs["side"] = side
            act = agent.decide(battle=battle, fmt=fmt, vocabs=vocabs,
                               **decide_kwargs)
            if guard is not None:
                guard.on_success()
            return act
        lines = [ln for ln in battle._build_replay_log().split("\n") if ln.strip()]
        if side is None:
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


#: D3 (2026-11-23 ruling): the log tags that constitute GAME-STATE PROGRESS.
#: The ruling: "no HP change from any source (damage, poison/burn, hazards,
#: weather), no status applied, no boost change and no hazard/screen set" is
#: a no-progress turn. The tag vocabulary is built from THIS FORK's own
#: belief adapter (``belief/log_events.py::_map`` — the ground-truth parser,
#: D29), including tags the belief IGNORES but that still change state
#: (``upkeep`` = Life Orb chip, ``-clear*boost`` resets, ``-mustrecharge``
#: locks, forme/item state). Deliberately GENEROUS: over-counting progress
#: only DELAYS the cap (the 200-step cap still ends a stall as a draw — the
#: status quo), while a scanner MISS is caught by the belief-signature
#: confirmation below, so both failure modes degrade to today's outcome.
#: EXCLUDED on purpose (action history, not state): move / -fail / cant /
#: movefail / -miss / -block, switch / drag / replace / leave, turn, the
#: damage-modifier lines (-immune / -resisted / -supereffective / -crit /
#: -ohko), and every battle-meta tag.
_PROGRESS_TAGS = frozenset({
    # HP change, any source (move hit, drain/recoil as context, item upkeep,
    # hazard chip)
    "-damage", "-heal", "-sethp", "-side damage", "upkeep",
    # a KO
    "faint",
    # status applied / removed
    "-status", "-curestatus",
    # stat-boost change (per-stat or whole-reset)
    "-boost", "-unboost", "-layerchange", "-setboost",
    "-clearboost", "-clearpositiveboost", "-clearnegativeboost",
    "-clearallboost",
    # volatiles (Focus Energy, Substitute, Leech Seed, Curse, Taunt, ...)
    "-start", "-end",
    # field: hazards / screens / pseudo-weather / rooms
    "-sidestart", "-sideend", "-fieldstart", "-fieldactivate", "-fieldend",
    "-clearspikes", "-swapsideconditions",
    # weather / terrain
    "weather", "-weather", "terrain",
    # tera / forme / item state
    "terastallize", "-terastallize", "-mega", "-primal", "-formechange",
    "-transform", "-enditem", "-warpplate",
    # recharge lock (Hyper Beam class)
    "-mustrecharge",
})


def progress_signature(ctx: Optional[Dict]) -> Optional[tuple]:
    """D3: the game-state key that the NO-PROGRESS confirmation compares.

    Includes EXACTLY the state the ruling treats as progress: per-slot HP,
    statuses, boosts, volatiles, tera, stockpile, revealed max HP; the
    fainted counts; and the whole field (weather, terrain, rooms, screens,
    hazards). EXCLUDED (not progress under the ruling, and must NOT keep the
    cap from firing): the turn counter, the active slots (switch ping-pong is
    the stall the cap exists to end), ``moves`` (moves_used — Protect spam),
    the opponent's ``posterior`` / ``top_moves`` (inference + revealed moves,
    not game state), and the per-turn flags (``hit_this_turn`` /
    ``used_protect`` / ``last_move_failed`` / ``last_hit``).
    ``None`` ctx (belief unavailable) -> ``None`` (the caller treats that as
    'cannot confirm' — the cap does NOT trip; the step cap backstops).
    """
    if not isinstance(ctx, dict):
        return None

    def _mon_key(b: Optional[Dict]) -> tuple:
        b = b or {}
        return (
            b.get("hp_frac"),
            tuple(sorted(b.get("statuses") or [])),
            tuple(sorted((b.get("boosts") or {}).items())),
            tuple(sorted(b.get("volatiles") or [])),
            b.get("tera") or "",
            b.get("stockpile") or 0,
            b.get("maxhp"),
        )

    def _side_key(side: Optional[Dict]) -> tuple:
        side = side or {}
        return tuple(sorted((slot, _mon_key(side.get(slot))) for slot in side))

    f = ctx.get("field") or {}
    return (
        tuple(sorted((ctx.get("fainted") or {}).items())),
        _side_key(ctx.get("own")),
        _side_key(ctx.get("opponent")),
        (
            f.get("weather"), f.get("terrain"),
            bool(f.get("trickroom")), bool(f.get("wonderroom")),
            f.get("tailwind_side"),
            tuple(sorted((f.get("reflect") or {}).items())),
            tuple(sorted((f.get("lightscreen") or {}).items())),
            tuple(sorted((f.get("auroraveil") or {}).items())),
            tuple(sorted((f.get("spikes") or {}).items())),
            tuple(sorted((f.get("toxicspikes") or {}).items())),
            tuple(sorted((f.get("stealthrock") or {}).items())),
            tuple(sorted((f.get("stickyweb") or {}).items())),
            tuple(sorted((f.get("safeguard") or {}).items())),
        ),
    )


class NoProgressTracker:
    """D3 (2026-11-23 ruling) — a PURE state machine over Showdown log lines.

    Counts CONSECUTIVE ``|turn|`` boundaries with no PROGRESS line between
    them (see :data:`_PROGRESS_TAGS`). Switch spam, Protect spam and move
    spam are NO progress — they are exactly the stalls the cap exists to end.

    The tracker only COUNTS; the caller CONFIRMS via the belief signature
    (:func:`progress_signature`, built from the fork log — D29) before ending
    the battle, so a scanner miss (an unlisted line type that really changed
    state) cannot false-positive the cap: the caller resets the tracker when
    the belief shows a change. Pure — unit-testable with fake lines, no
    battle object needed.
    """

    def __init__(self, limit: int = 20) -> None:
        self.limit = max(1, int(limit))
        self.streak = 0
        self.progress_this_turn = False

    def feed_lines(self, lines) -> None:
        """Consume log lines (append-only; the caller passes only NEW lines).
        Never raises on malformed input — a junk line is simply skipped."""
        for ln in lines:
            s = (ln or "").strip()
            if not s.startswith("|"):
                continue
            parts = s.split("|")
            if len(parts) < 2:
                continue
            tag = parts[1]
            if tag == "turn":
                if self.progress_this_turn:
                    self.streak = 0
                else:
                    self.streak += 1
                self.progress_this_turn = False
            elif tag in _PROGRESS_TAGS:
                self.progress_this_turn = True

    @property
    def tripped(self) -> bool:
        return self.streak >= self.limit

    def reset(self) -> None:
        self.streak = 0
        self.progress_this_turn = False


def _cap_signature(env: Any, fmt: str) -> Optional[tuple]:
    """D3: the belief signature for the no-progress confirmation (p1 POV —
    the shared battle record, so the POV is irrelevant to the state key).
    Never raises (belief failure -> ``None`` -> the cap does NOT trip; the
    step cap backstops the stall as a draw, exactly as today)."""
    try:
        from net.env import build_ctx
        return progress_signature(build_ctx(env.battle1, fmt, "p1"))
    except Exception:  # noqa: BLE001 — see docstring
        return None


def _step_loop(env: Any, decide_a: Callable[[Any], Any],
               decide_b: Callable[[Any], Any], max_steps: int,
               fmt: str = "gen9ou",
               no_progress_limit: int = 20,
               log: Optional[Callable[[str], None]] = None) -> int:
    """The shared battle step-loop over an ALREADY-RESET env (mirrors
    ``pool.battle.play_battle_on``'s loop with policy decisions instead of
    ``random_action``). Extracted so the loop's scheduling is testable with a
    fake env — no server, no net. Returns the number of steps taken.

    D3 (2026-11-23 ruling): after ``no_progress_limit`` consecutive turns
    with NO game-state progress (see :data:`_PROGRESS_TAGS`) the battle ends
    early as a DRAW (no side has won — the harness counts it as a draw; legit
    stall play — hazard chip, toxic — IS progress and keeps the battle alive).
    The confirmation re-checks the belief state at the cap against the
    streak's start (a state change the scanner missed -> the streak resets).
    ``no_progress_limit=0`` disables the cap (the step cap alone governs)."""
    import numpy as np
    from net.env import action_to_pe

    cap = NoProgressTracker(no_progress_limit) if no_progress_limit > 0 else None
    baseline: Optional[tuple] = None  # belief signature at streak==1
    seen_lines = 0

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
        if cap is None or env.battle1 is None:
            continue
        # D3: scan the NEW log lines since the last step (append-only log; the
        # same _build_replay_log() the belief pipeline consumes — D29).
        try:
            lines = env.battle1._build_replay_log().split("\n")
        except Exception:  # noqa: BLE001 — a scan hiccup must never kill the
            continue      # battle (the cap simply stays inert this step)
        if len(lines) > seen_lines:
            cap.feed_lines(lines[seen_lines:])
            seen_lines = len(lines)
        if cap.streak == 1:
            baseline = _cap_signature(env, fmt)
        if cap.tripped:
            sig = _cap_signature(env, fmt)
            if (baseline is not None and sig is not None
                    and sig == baseline):
                if log is not None:
                    log(f"[driver] D3 no-progress cap: {cap.streak} turns "
                        f"without HP/status/boost/field change -> "
                        f"ending the battle as a draw")
                break
            # the belief shows a change the scanner missed -> keep playing
            cap.reset()
            baseline = None
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
                         capture_b: Optional[Callable[[Any, Any], None]] = None,
                         no_progress_limit: int = 20
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
    :param no_progress_limit: D3 (2026-11-23 ruling) — consecutive NO-PROGRESS
                   turns (no HP change / status / boost / field change) after
                   which the battle ends early as a DRAW (default 20;
                   ``0`` = disabled, the step cap alone governs).
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

        # 2026-11-23 POV fix: the env defines the side identity (battle1=p1,
        # battle2=p2 — the same mapping ShowdownEnv.embed_battle uses for the
        # training observation). Pass it explicitly so a decision is NEVER
        # built from the wrong POV (my_side's species guess silently failed on
        # real fork logs — gender suffix — and defaulted p2 players to p1).
        def da(battle: Any) -> Any:
            if decide is not None:
                return decide(agent_a, battle, fmt, vocabs, greedy)
            return policy_action(agent_a, battle, fmt, vocabs, greedy, log, guard,
                                 capture=capture_a, side="p1")

        def db(battle: Any) -> Any:
            if decide is not None:
                return decide(agent_b, battle, fmt, vocabs, greedy)
            return policy_action(agent_b, battle, fmt, vocabs, greedy, log, guard,
                                 capture=capture_b, side="p2")

        n_steps = _step_loop(env, da, db, max_steps, fmt=fmt,
                             no_progress_limit=no_progress_limit, log=log)
        if on_finish is not None:
            on_finish(env.battle1)
        return bool(env.battle1.won), bool(env.battle2.won), n_steps
    finally:
        env.close()
