"""Step 6.6 — the battle-time strategy policy: the ctx brain as a PLAYABLE agent.

``fitness.strategy.choose_action`` (rules 1–4, Steps 6.2–6.5) is a pure
function over the ``get_context()`` snapshot — but the live battle hands an
agent the ENCODED obs, not the ctx. This module closes that gap WITHOUT a
second pipeline (one-workflow):

    battle -> net.env.build_ctx   (the SAME belief pipeline build_obs uses)
           -> net.env.build_action_list
           -> fitness.strategy.choose_action  (the ONE shared damage calc)

It is a ``Reference`` (``fitness.ladder``), so the harness / league / Step-8
can play it like any bot: the driver's ``policy_action`` recognizes the
``decides_from_battle`` marker and routes the decision through
``agent.decide()`` (which sees the battle) instead of ``agent.act(obs, acts)``.

CPU-first: importing this module pulls NO torch and NO poke_env (both are
imported lazily inside ``decide`` — the same discipline as ``fitness.heuristic``).
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

__all__ = ["StrategyAgent", "strategy_reference",
           "strategy_posterior_reference", "recently_switched_after",
           "fallback_stats", "fallback_reset"]

# ── C5 (GATE_CONTROLS_C1_C5.md): silent-fallback counters ──────────────────
# The strategy bot's `decide()` degrades silently in three places (A3.1
# contract). The gate spec requires counting them per 100 battles — they must
# be zero or near zero, or the yardstick is quietly playing a weaker policy
# than we think. Module-level (process-wide): the eval loop runs workers as
# THREADS in one process, so one shared counter is correct. Thread-safe.
_FB_LOCK = threading.Lock()
_FB: Dict[str, int] = {
    "decides": 0,             # total decide() calls (the denominator)
    "ctx_none_acts0": 0,      # build_ctx returned None -> acts[0] fallback
    "choose_exc_acts0": 0,    # choose_action raised -> acts[0] fallback
    "pov_except": 0,          # my_side() POV detection raised -> defaulted p1
    "passed": 0,              # no legal actions -> pass/wait (legit, counted)
}


def fallback_stats() -> Dict[str, int]:
    """A snapshot of the C5 silent-fallback counters."""
    with _FB_LOCK:
        return dict(_FB)


def fallback_reset() -> None:
    """Zero the counters (a fresh eval session)."""
    with _FB_LOCK:
        for k in _FB:
            _FB[k] = 0


def _fb_bump(key: str) -> None:
    with _FB_LOCK:
        _FB[key] = _FB.get(key, 0) + 1


def _gen_from_format(fmt: str) -> int:
    """The @smogon/calc engine key (the ``gen`` argument) for a format id.

    The ONE source of truth is ``legality.formats.engine_key`` (D32):
    champions formats map to **0, NOT 9** — mapping them to 9 would
    silently mis-calc ~25 moves (champions Slash BP=80 vs gen-9 70). The
    fallback covers format ids OUTSIDE the registry (custom formats);
    unrecognised -> 9 (the primary format). Lazy import keeps this module
    cheap at import time. B1: the fallback is LOUD (a warning) via
    ``engine_key_or_warn`` — no silent mis-keying.
    """
    from legality.formats import engine_key_or_warn
    return int(engine_key_or_warn(fmt, default=9))


class StrategyAgent:
    """The ctx-based brain (``choose_action`` rules 1–4) as a battle policy.

    Marker contract (consumed by ``league.driver.policy_action``):
      * ``decides_from_battle = True`` — this policy consumes the BATTLE (the
        belief pipeline), not the encoded obs;
      * ``decide(battle, fmt, vocabs) -> action dict | None`` (``None`` =
        pass/wait, no legal action this moment).

    The decision builds the belief's ``get_context()`` snapshot via
    ``net.env.build_ctx`` — the EXACT pipeline ``build_obs`` uses (one source
    of truth) — and runs ``fitness.strategy.choose_action`` over it with the
    real legal-action list. The damage calc is the ONE shared model
    (``choose_action``'s default = ``damage.calc.calc_damage``).

    Deterministic (``seed`` is accepted for ``Reference.make_agent(seed)``
    contract parity but is unused — like ``HeuristicAgent``). Never raises:
    a belief failure degrades to the first legal action (the A3.1 contract),
    an empty action list degrades to a pass.
    """

    decides_from_battle = True

    def __init__(self, fmt: str = "gen9ou", gen: int = 9,
                 switch_threshold: float = 0.30,
                 seed: Optional[int] = None,
                 posterior: bool = False,
                 no_reswitch_turns: int = 2) -> None:
        self.fmt = fmt
        self.gen = int(gen)
        self.switch_threshold = float(switch_threshold)
        self.seed = seed
        # D2 (2026-11-23 ruling): the hysteresis window N — a switch on turn T
        # blocks a VOLUNTARY can_ko switch on turns T+1..T+N-1 (escape
        # switches are never blocked). The agent is built per battle
        # (Reference.make_agent), so this memory resets itself each battle.
        self.no_reswitch_turns = int(no_reswitch_turns)
        self._last_switch_turn: Optional[int] = None
        # 6.6a: when True, the damage calc uses the opponent's TOP-1 posterior
        # (ability/evs/nature) + tera — the SAME information the net's action
        # features see (a fair M1 comparison). The plain ``strategy`` yardstick
        # keeps posterior=False (known-info only, UNCHANGED).
        self.posterior = bool(posterior)

    def decide(self, battle: Any, fmt: Optional[str] = None,
               vocabs: Optional[Any] = None,
               side: Optional[str] = None) -> Optional[Dict]:
        """ONE decision from the live battle. Never raises (see class doc).

        ``side`` (2026-11-23 POV fix): the caller's KNOWN side for ``battle``
        (the self-play driver passes battle1->"p1" / battle2->"p2" straight
        from the env — the same mapping the training observation uses). When
        ``None`` (human-vs-bot / direct use) it falls back to the ``my_side``
        species guess. A wrong POV here breaks the whole decision: the belief
        would track the opponent's team as "own" (active unresolved, 0 HP),
        and rule 3 would switch every turn — the 2026-11-23 draw bug.
        """
        _fb_bump("decides")
        fmt = fmt or self.fmt
        # POV: the belief pipeline is p1-POV; a p2 player's log is swapped by
        # build_ctx (my_side detects the side — same helper the driver uses).
        if side is None:
            side = "p1"
            try:
                from net.env import my_side
                lines = [ln for ln in battle._build_replay_log().split("\n")
                         if ln.strip()]
                side = my_side(battle, lines)
            except Exception:  # noqa: BLE001 — a POV hiccup defaults to p1
                _fb_bump("pov_except")  # C5: a silent POV fallback
                side = "p1"

        acts: list = []
        try:
            from net.env import build_action_list
            acts = list(build_action_list(battle) or [])
        except Exception:  # noqa: BLE001 — no legal actions -> pass
            acts = []
        if not acts:
            _fb_bump("passed")  # C5: a pass/wait turn
            return None  # a pass/wait turn (no legal action this moment)

        from net.env import build_ctx
        ctx = build_ctx(battle, fmt, side)
        if ctx is None:
            _fb_bump("ctx_none_acts0")  # C5: belief unavailable -> acts[0]
            return acts[0]  # belief unavailable: first legal (the A3.1 degrade)

        from fitness.strategy import choose_action, _posterior_overrides
        # 6.6a: the posterior variant feeds the opponent's top-1 build into the
        # calc (mirroring the net); the plain yardstick passes known-info only.
        ov: Dict = {}
        if self.posterior:
            own_slot = (ctx.get("active") or {}).get("p1")
            opp_slot = (ctx.get("active") or {}).get("p2")
            if own_slot is not None and opp_slot is not None:
                try:
                    ov = _posterior_overrides(ctx, own_slot, opp_slot) or {}
                except Exception:  # noqa: BLE001 — degrade to known-info only
                    ov = {}
        # D2 (2026-11-23): hysteresis — feed the per-battle switch memory into
        # rule 3 (a voluntary can_ko switch is blocked right after a switch;
        # escape switches are never blocked). ``ctx['turn']`` is the belief's
        # turn counter — the same source the observation sees (D29: the fork
        # log is the ground truth the belief tracks).
        turn = ctx.get("turn")
        try:
            act = choose_action(
                ctx, acts, fmt=fmt, gen=self.gen,
                switch_threshold=self.switch_threshold,
                recently_switched=recently_switched_after(
                    self._last_switch_turn, turn, self.no_reswitch_turns),
                **ov)
        except Exception:  # noqa: BLE001 — choose_action should not raise;
            _fb_bump("choose_exc_acts0")  # C5: silent acts[0] degrade
            act = None     # belt-and-braces: degrade to a legal action
        if act is None:
            act = acts[0]
        # D2: record this decision's switch (voluntary OR forced) so the next
        # turn(s) inside the window see ``recently_switched=True``.
        if (isinstance(act, dict) and act.get("kind") == "switch"
                and isinstance(turn, int)):
            self._last_switch_turn = turn
        return act


def recently_switched_after(last_switch_turn: Optional[int],
                            current_turn: Optional[int],
                            no_reswitch_turns: int) -> bool:
    """D2 (2026-11-23): True when ``current_turn`` falls INSIDE the
    no-re-switch window after the last switch.

    A switch on turn T blocks a VOLUNTARY can_ko switch on turns T+1..T+N-1
    (N = ``no_reswitch_turns``); turn T+N is free again. Escape switches
    (guaranteed_ko / HP <= threshold) ignore the window entirely (see
    ``fitness.strategy.choose_action`` rule 3). A missing turn (degrade path)
    or N <= 0 -> False (hysteresis simply inactive). A same-turn re-decision
    (diff 0, defensive) blocks rather than allows. Pure — unit-testable
    without a battle object.
    """
    if last_switch_turn is None or current_turn is None:
        return False
    try:
        last = int(last_switch_turn)
        cur = int(current_turn)
        n = int(no_reswitch_turns)
    except (TypeError, ValueError):
        return False
    if n <= 0:
        return False
    return 0 <= (cur - last) < n


def strategy_reference(fmt: str = "gen9ou",
                       gen: Optional[int] = None) -> "Any":
    """The NEW heuristic (rules 1–4) as a ladder rung — the standing Step-8
    yardstick (the 6.1 design note: "the NEW heuristic becomes the standing
    Step 8 yardstick"). Weight 2.0, the same rung as ``heuristic_reference``.
    Lazy, like ``random_reference`` / ``heuristic_reference``."""
    from .ladder import Reference

    g = int(gen) if gen is not None else _gen_from_format(fmt)

    def make_agent(seed: Optional[int] = None) -> StrategyAgent:
        return StrategyAgent(fmt=fmt, gen=g, seed=seed)

    return Reference(
        name="strategy", make_agent=make_agent, weight=2.0,
        description=("ctx brain (choose_action rules 1-4: hazard/status/"
                     "setup/resist-switch) — the Step-6 teacher heuristic"))


def strategy_posterior_reference(fmt: str = "gen9ou",
                                 gen: Optional[int] = None) -> "Any":
    """6.6a — the strategy bot WITH the opponent's TOP-1 posterior build
    (ability/evs/nature/tera) fed into the damage calc — the SAME information
    the net's action features see. This is the SECOND yardstick for the M1 gate
    (the user's 6.6a: report the net against BOTH ``strategy`` and
    ``strategy_posterior``; the existing ``strategy`` yardstick is unchanged).
    Same weight (2.0) and rules (1-4) as ``strategy_reference`` — the ONLY
    difference is the informed damage calc. Lazy, like the others."""
    from .ladder import Reference

    g = int(gen) if gen is not None else _gen_from_format(fmt)

    def make_agent(seed: Optional[int] = None) -> StrategyAgent:
        return StrategyAgent(fmt=fmt, gen=g, seed=seed, posterior=True)

    return Reference(
        name="strategy_posterior", make_agent=make_agent, weight=2.0,
        description=("ctx brain (rules 1-4) WITH the opponent's top-1 posterior"
                     " build in the damage calc — the fair-comparison yardstick "
                     "for the M1 gate (same info the net sees)"))
