"""Step 3.4 (review 2026-11-22) — per-OWN-MON MATCHUP summary.

The net sees HP/status/hazards + static mon/move facts + the per-ACTION damage
block (3.3, "will THIS move KO the active right now"), but it still had no
BOARD-LEVEL damage landscape: *how does each of my mons fare against the whole
enemy team?* That is exactly the signal the win-condition / switch / threat
planning needs ("my Dragonite 6-0s once the Gengar is dead", "who can I KO, who
KOs me", "is this switch-in safe"). This module emits, for EACH own mon, an
8-dim summary over all enemy mons.

Convention (same as the rest of net/):
  * the OUTPUT is exactly ONE registry field (``own_matchup``) as a numpy array
    — the encoder (:func:`net.encoder.encode_context`) places it (zero-fill when
    absent, so the encoder stays pure / numpy / DB-free);
  * PURE w.r.t. its inputs: same ctx + fmt + gen -> a bitwise-identical array
    (deterministic; the calc is a pure function and the move picks are the
    deterministic highest-power / top-1);
  * DEGRADES, never raises: an unknown species/move or a calc failure yields a
    ZERO row for that mon (the net reads "no matchup signal") — exactly like the
    id channels;
  * lazy calc import so ``import net.matchup_features`` stays cheap and
    torch-free (the same discipline as fitness/ and action_features/).

Reuse, do NOT re-derive: the per-pair damage comes from the ONE
``damage.calc.calc_damage_many`` batch (a SINGLE Node spawn — the
"quick + API-reachable" priority), and the KO verdicts from
``fitness.ko.ko_assessment`` (the same threshold semantics as ``move_ko``: a
defender at ``hp_frac`` has ``hp_frac`` of its max HP; a move KOs when its max
dmg >= that threshold). Opponent hidden info = that mon's TOP-1 posterior
ability (the 3.3 convention). See ``OPEN_DESIGN_CHOICES.md`` ("3.4 DESIGN") for
the locked 8-dim schema and the deliberate DROP of a per-pair speed dim.
``calc_fn`` is injectable so tests pin the interpretation with a fake batch.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .registry import K_OWN_MATCHUP, MAX_SLOTS

__all__ = ["OWN_MATCHUP_NAMES", "build_own_matchup",
           "own_attack_move", "opp_attack_move"]

#: the exact registry field this module fills (the encoder's zero-fill set).
OWN_MATCHUP_NAMES: tuple = ("own_matchup",)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _hp_frac(block: Optional[Dict], default: float = 1.0) -> float:
    """A block's ``hp_frac`` as a clamped 0.0-1.0 float (missing/None -> default)."""
    v = (block or {}).get("hp_frac", default)
    if v is None:
        v = default
    return max(0.0, min(1.0, float(v)))


def _top_ability(block: Optional[Dict]) -> Optional[str]:
    """The mon's TOP-1 posterior ability (deterministic), or None.

    Mirrors ``action_features._top_ability``: the belief's per-slot posterior is
    a list of ``{ability, weight, ...}``; take the highest-weight candidate
    (ties -> the first, stable). This is the hidden-info override for the calc.
    """
    best, best_w = None, -1.0
    for cand in ((block or {}).get("posterior") or []):
        w = float(cand.get("weight", 0.0) or 0.0)
        if w > best_w:
            best_w, best = w, cand
    return (best.get("ability") or None) if best is not None else None


def _move_power(move_name: str, gen: int) -> int:
    """A move's base power from the legality DB (0 for status/unknown)."""
    if not move_name:
        return 0
    try:
        from legality.lookup import get_move
        m = get_move(move_name, gen) or {}
    except Exception:  # noqa: BLE001 — an unknown move degrades to power 0
        return 0
    p = m.get("power")
    return int(p) if isinstance(p, (int, float)) and p > 0 else 0


def own_attack_move(moves: List[str], gen: int) -> Optional[str]:
    """The own mon's STRONGEST damaging move (highest base power, deterministic).

    Falls back to the first move when none has a known power (or there is no
    DB row), so a mon always yields a single representative attack move. A
    status-only moveset degrades to its first move (the calc then reports no
    damage for it — a zero contribution, which is correct).
    """
    moves = list(moves or [])
    if not moves:
        return None
    best, best_p = moves[0], 0
    for m in moves:
        p = _move_power(m, gen)
        if p > best_p:
            best_p, best = p, m
    return best


def opp_attack_move(block: Optional[Dict]) -> Optional[str]:
    """The opponent mon's TOP move (belief posterior), fallback its first move.

    ``top_moves`` is the belief's ranked marginal (``get_context()``); the first
    entry is the deterministic pick (the 3.3 convention). Falls back to the
    first observed move so a belief without a posterior still yields a signal.
    Returns ``None`` when the opponent has revealed NO move yet (turn 1 / an
    untouched bench mon) — the caller then uses the species' best legal damaging
    move (see :func:`best_damaging_move`) so the matchup signal still exists.
    """
    tops = (block or {}).get("top_moves") or []
    if tops:
        return tops[0]
    used = (block or {}).get("moves") or []
    return used[0] if used else None


#: a neutral damaging move that every species can legally carry in every
#: generation (Body Slam, 85 power, 100% accuracy, always hits). It is the
#: representative "attacking" move when a mon has revealed NO move yet (turn-1
#: opponent, an untouched own bench mon in a spec without a full moveset) — a
#: sane, universal, always-hits baseline (a "highest power" scan would pick
#: quirky moves like Explosion / Gigaimpact, which are NOT representative).
#: Public legality data only — using it leaks no hidden info (it is not a
#: species-specific move, just a neutral damage baseline for the summary).
NEUTRAL_DAMAGING_MOVE = "bodyslam"


@lru_cache(maxsize=256)
def best_damaging_move(name: str, fmt: str, gen: int) -> str:
    """A representative damaging move for a mon that has revealed none yet.

    Returns the neutral Body Slam baseline (see :data:`NEUTRAL_DAMAGING_MOVE`).
    Kept as a function (rather than the constant) so a caller can inject a
    different baseline and so the signature is stable if a smarter pick is
    wanted later; the (name, fmt, gen) args are accepted for that extensibility
    and cached so the call is cheap. No hidden-info leak: Body Slam is a
    species-independent, always-legal, always-hits move.
    """
    return NEUTRAL_DAMAGING_MOVE


def _calc_mon(block: Dict, ability: Optional[str]) -> Dict:
    """Map a ``get_context()`` mon block to an @smogon/calc dict (no I/O).

    Uses ``fitness.ko.block_to_calc_mon`` (the ONE block->calc mapping) with the
    top-1 ability override. Also passes the block's known tera type, item, and
    the posterior's top-1 EV spread + nature (the suspected build — makes the
    damage numbers accurate to the expected mon, not just base stats).
    Returns ``{}`` when no species is known (the caller skips the calc for
    that mon).
    """
    from fitness.ko import block_to_calc_mon
    tera = (block.get("tera") or None) or None
    item = (block.get("item") or None) or None
    evs = (block.get("evs") or None) or None
    nature = (block.get("nature") or None) or None
    # For opponent blocks: EVs come from the top-1 posterior candidate
    if evs is None and block.get("posterior"):
        best = max(block["posterior"], key=lambda c: float(c.get("weight", 0)))
        evs = best.get("evs") or None
        nature = nature or best.get("nature") or None
    return block_to_calc_mon(block, ability=ability, tera_type=tera, item=item,
                             evs=evs, nature=nature)


def build_own_matchup(ctx: Dict, fmt: str, gen: int = 9, *,
                      calc_fn: Optional[Callable] = None) -> Dict[str, np.ndarray]:
    """The ONE ``own_matchup`` field (``(MAX_SLOTS, K_OWN_MATCHUP)`` float32).

    For each ALIVE own mon (canonical a..f order) summarises, over all ALIVE
    enemy mons: the strongest hit it can land, the strongest hit it can take,
    how many enemy mons it KOs (guaranteed / can), how many can KO it back,
    whether it survives the strongest hit, and the net KO margin. One
    ``calc_damage_many`` batch (up to 6x6x2 items) in a SINGLE Node spawn.

    :param ctx: the ``belief.state.BeliefState.get_context()`` dict.
    :param fmt / gen: the format id (wins) / engine key for the shared calc.
    :param calc_fn: injectable (tests) — a ``calc_damage_many(items)``-shaped
                    callable; defaults to the shared ``damage.calc.calc_damage_many``.
    """
    out = np.zeros((MAX_SLOTS, K_OWN_MATCHUP), dtype=np.float32)
    own_blocks = ctx.get("own", {}) or {}
    opp_blocks = ctx.get("opponent", {}) or {}
    if not own_blocks or not opp_blocks:
        return {"own_matchup": out}

    # Canonical a..f slot order (deterministic). Own blocks are keyed
    # "p1a".., opponent "p2a".." (the belief's get_context slot ids). A mon is
    # considered iff it is PRESENT and NOT fainted (a gone mon has no matchup;
    # a benched mon still does).
    own_alive: List[Tuple[int, Dict]] = []
    for i in range(MAX_SLOTS):
        b = own_blocks.get(f"p1{chr(ord('a') + i)}")
        if b and not b.get("fainted"):
            own_alive.append((i, b))
    opp_alive: List[Tuple[int, Dict]] = []
    for i in range(MAX_SLOTS):
        b = opp_blocks.get(f"p2{chr(ord('a') + i)}")
        if b and not b.get("fainted"):
            opp_alive.append((i, b))
    if not own_alive or not opp_alive:
        return {"own_matchup": out}

    # Resolve each mon's attack move + top-1 ability ONCE (deterministic).
    own_block = {i: b for i, b in own_alive}
    opp_block = {i: b for i, b in opp_alive}
    # A mon with NO revealed move (own bench in a spec without a full moveset;
    # a turn-1 / untouched opponent) still gets a representative attack move via
    # the species' best legal damaging move — public data, no hidden-info leak.
    def _own_move_name(b: Dict) -> Optional[str]:
        mv = own_attack_move(b.get("moves", []) or [], gen)
        if mv:
            return mv
        nm = b.get("name") or b.get("species") or ""
        return best_damaging_move(nm, fmt, gen) if nm else None

    def _opp_move_name(b: Dict) -> Optional[str]:
        mv = opp_attack_move(b)
        if mv:
            return mv
        nm = b.get("name") or b.get("species") or ""
        return best_damaging_move(nm, fmt, gen) if nm else None

    own_move = {i: _own_move_name(own_block[i]) for i in own_block}
    opp_move = {i: _opp_move_name(opp_block[i]) for i in opp_block}
    opp_ability = {i: _top_ability(opp_block[i]) for i in opp_block}

    # Build the ONE batch: for each (own, opp) pair, BOTH directions.
    items: List[Dict] = []
    pair_idx: List[Tuple[int, int]] = []
    for oi in own_block:
        omon = _calc_mon(own_block[oi], None)          # own mon, own ability (public)
        if not omon or not own_move[oi]:
            continue
        for pi in opp_block:
            pmon = _calc_mon(opp_block[pi], opp_ability[pi])  # opp mon, top-1 ability
            if not pmon or not opp_move[pi]:
                continue
            pair_idx.append((oi, pi))
            items.append({"attacker": omon, "defender": pmon,
                          "move": own_move[oi], "fmt": fmt, "gen": gen})
            items.append({"attacker": pmon, "defender": omon,
                          "move": opp_move[pi], "fmt": fmt, "gen": gen})
    if not items:
        return {"own_matchup": out}

    try:
        from damage import calc as _calc
        batch_fn = calc_fn or _calc.calc_damage_many
        res = batch_fn(items)
    except Exception:  # noqa: BLE001 — a calc failure degrades to zero, never raises
        return {"own_matchup": out}
    if not isinstance(res, list) or len(res) != len(items):
        return {"own_matchup": out}

    # Aggregate per own mon (index oi) over the enemy mons it was paired with.
    from fitness.ko import ko_assessment
    n_opp = len(opp_block)
    for k, (oi, pi) in enumerate(pair_idx):
        own_dmg = res[2 * k]
        opp_dmg = res[2 * k + 1]
        if not isinstance(own_dmg, dict) or "error" in own_dmg:
            own_dmg = {}
        if not isinstance(opp_dmg, dict) or "error" in opp_dmg:
            opp_dmg = {}
        own_exp = float(own_dmg.get("expected_pct", 0.0) or 0.0)
        opp_exp = float(opp_dmg.get("expected_pct", 0.0) or 0.0)
        # KO verdicts: the own mon is the ATTACKER (defender = the opp mon at its
        # HP); the opp mon is the attacker (defender = the own mon at its HP).
        own_assess = ko_assessment(own_dmg, defender_hp_frac=_hp_frac(opp_block[pi]))
        opp_assess = ko_assessment(opp_dmg, defender_hp_frac=_hp_frac(own_block[oi]))
        v = out[oi]
        v[0] = max(v[0], _clamp01(own_exp / 100.0))    # best_hit  (max over opp)
        v[1] = max(v[1], _clamp01(opp_exp / 100.0))    # worst_hit (max over opp)
        if own_assess.get("guaranteed_ko"):
            v[2] += 1.0 / n_opp                          # ko_guar_frac
        if own_assess.get("can_ko"):
            v[3] += 1.0 / n_opp                          # ko_can_frac
        if opp_assess.get("can_ko"):
            v[4] += 1.0 / n_opp                          # taken_ko_frac
    for oi in own_block:
        v = out[oi]
        v[5] = 1.0 if v[1] < 1.0 else 0.0               # survives_strongest
        v[6] = max(-1.0, min(1.0, v[3] - v[4]))         # net_ko_margin
        v[7] = max(-1.0, min(1.0, v[0] - v[1]))         # best_vs_worst
    return {"own_matchup": out}
