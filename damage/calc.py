"""Step 2 — Damage calculator wrapper (build plan: "Damage Calculator Wrapper").

Pythonic API over the official @smogon/calc damage model. The heavy lifting is done
by ``node_calc.js`` (a Node subprocess that calls @smogon/calc directly), so the
numbers always match the library; this module only marshals JSON in/out and adds a
Python-friendly surface plus the speed-comparison helper.

    from damage import calc
    calc.calc_damage(attacker, defender, move, field=None, fmt="gen9bss3v3", gen=9)
        -> {'min','max','expected',            # damage in HP
            'min_pct','max_pct','expected_pct',# % of defender max HP
            'crit': {'min','max','expected'},  # critical-hit range (separate value)
            'immune': bool, 'maxhp': int, 'desc': str,
            # T6 attacker-side blocks (present only when the move has the effect
            # in that engine — absent on immune/miss/no-effect responses):
            'recoil':  {'min','max'},  # self-damage; fraction-of-damage-dealt
                                       # (fork: gens 1-4 floor, gens 5-9/champions round)
                                       # or fixed fraction of the USER's max HP
                                       # (Struggle 1/4 gen 4+, Mind Blown 1/2);
                                       # damage-dealt values are computed on the
                                       # damage CLAMPED to the defender's
                                       # remaining HP (pass defender 'hp'/'curHP')
            'heal':    {'min','max'},  # drain healing (Drain Punch, Giga Drain…);
                                       # net gain is additionally capped by the
                                       # user's missing HP (belief state applies)
            'lifeOrb': {'min','max'},  # Life Orb self-damage (10% of user max HP)
                                       # — set 'item' on the attacker
            'critChance': float | None # fork BASE crit rate for this move/engine
                                       # (1/24 gens 7-9/champions, 1/16 gens <= 6,
                                       # 1/8 critRatio-2 moves); battle-state
                                       # modifiers (Super Luck, Scope Lens, …)
                                       # are belief-state facts, not calc inputs}

    calc.compare_speed(a, b, fmt="gen9bss3v3", gen=9)
        -> {'a': int, 'b': int, 'faster': 'a' | 'b' | 'tie'}

Prefer passing ``fmt`` (a format id from the legality registry): the engine
key is resolved from the format's ``mod`` (``genN`` -> N, ``champions`` -> 0 —
the library's native champions mechanics set, see PLAN.md D32), and each mon's
battle level is resolved through the format's level rules (D33): BSS 3v3,
VGC and champions battle at level 50, LC at level 5, ladder at 100. A mon
without ``level`` gets the format's battle level; an explicit level outside
the format's [minLevel, maxLevel] raises ``ValueError`` (the sim's validator
would reject such a set). ``gen`` is kept as the low-level engine key for
callers that already have it; when both are given, ``fmt`` wins. Without
``fmt`` there are no format level rules (raw engine semantics, library
default level 100).

attacker / defender / a / b are dicts using @smogon/calc option names:
``name`` (required), and any of ``level``, ``nature``, ``evs``, ``ivs``, ``item``,
``ability``, ``status`` ('par'/'brn'/...), ``boosts`` ({'atk':2,...}), ``teraType``,
``moves``, plus the belief-state flags ``abilityOn`` (bool: Unburden/Slow Start
active, QP-eligible) and ``boostedStat`` ('auto' or a stat name: Quark Drive /
Protosynthesis). A defender's ``teraType`` (gen 9 / champions only) replaces its
base types for the special-move immunity check — matching the fork's
getTypes-based runImmunity and the library's own effectiveness (PLAN.md D42).

``move`` is a move-name string (or ``{'name':..., 'isCrit':bool,
``lastHitDamage':int}``). ``lastHitDamage`` is the damage the user last took —
required by the reflector moves (``counter``/``mirrorcoat``/``metalburst``/
``comeuppance``), which scale off that prior hit; without it a reflector deals 0.

``hp`` (optional, AI-facing): the mon's CURRENT HP as a 0.0-1.0 fraction of its
max HP. @smogon/calc gates several damage effects on current HP — attacker-side
Defeatist (<=1/2, Atk&SpA x0.5), Overgrow/Blaze/Torrent/Swarm (<=1/3, x1.5 on the
matching type), Eruption/Water Spout/Flail/Reversal (BP scales with own HP); and
defender-side Multiscale/Shadow Shield/Tera Shell (only AT FULL HP, x0.5), Brine
(<=1/2, x1.5), Crush Grip/Wring Out/Hard Press (BP scales with the defender's HP).
Without ``hp`` the mon is assumed at full HP (unchanged behaviour). ``curHP`` is
accepted as a library-facing alias (absolute HP, clamped to the mon's max HP).
``field`` (optional) may set ``weather`` (capitalised: 'Rain'/'Sun'/'Sand'/'Hail'),
``terrain``, ``attackerSide`` and ``defenderSide`` (e.g. ``{'isReflect': True}``).

Speed model (see PLAN.md D30, D42): FULL mirror of @smogon/calc
``getFinalSpeed`` — base stat -> stat boosts (gen-aware table) -> Tailwind (x2)
-> ability chain (Unburden / Chlorophyll+Sun / Sand Rush+Sand / Swift Swim+Rain /
Slush Rush+Hail|Snow / Surge Surfer+Electric terrain: x2; else Quick Feet+status:
x1.5; else Slow Start: x0.5; else Quark Drive / Protosynthesis: x1.5 when the
boosted stat is Spe) -> item chain (Choice Scarf x1.5; else Iron Ball or an EV
item x0.5; else Quick Powder on Ditto x2) -> paralysis (Quick Feet exempt; the
per-gen factor is FORK-EXACT: gens 1-4 x0.25, gen 5 x0.5, gen 6 x0.25, gens 7-9
and champions x0.5 — @smogon/calc differs only for gen 5, where the fork, the
battle environment, wins) -> cap 999 (gen 1-2) / 10000. Pass ``field`` for the
weather/terrain speed abilities.
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, Optional, Union

from legality import formats as _formats
from legality import lookup as _lookup

_NODE = "node"
_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "node_calc.js")

Mon = Dict[str, Any]
MoveSpec = Union[str, Dict[str, Any]]


def _engine_key(gen: int, fmt: Optional[str]) -> int:
    """Resolve the @smogon/calc engine key: ``fmt`` wins over ``gen``."""
    if fmt is None:
        return gen
    return _formats.engine_key(fmt)


def _resolve_mon(mon: Mon, fmt: Optional[str]) -> Mon:
    """Resolve a mon's battle level through the format's rules (D33).

    No-op without ``fmt`` (raw engine path, library default level). With
    ``fmt``, mirrors the fork's team-validator: unset level -> the format's
    battle level; adjustLevelDown/adjustLevel applied; out-of-range explicit
    level -> ValueError.
    """
    if fmt is None:
        return mon
    m = dict(mon)
    m["level"] = _lookup.battle_level(fmt, mon.get("level"))
    return m


def _node(request: Dict[str, Any]) -> Dict[str, Any]:
    """Send one JSON request to the Node CLI and parse its JSON response."""
    proc = subprocess.run(
        [_NODE, _SCRIPT],
        input=json.dumps(request),
        capture_output=True,
        text=True,
        timeout=30,
    )
    out = proc.stdout.strip()
    if proc.returncode != 0:
        raise RuntimeError(f"node_calc.js failed (exit {proc.returncode}): {out or proc.stderr.strip()}")
    try:
        data = json.loads(out)
    except json.JSONDecodeError as e:  # pragma: no cover - defensive
        raise RuntimeError(f"node_calc.js returned non-JSON: {out!r}") from e
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def _normalise_move(move: MoveSpec) -> Dict[str, Any]:
    if isinstance(move, str):
        return {"name": move}
    if "name" not in move:
        raise ValueError("move must be a name or {'name': ...}")
    out = {"name": move["name"], "isCrit": bool(move.get("isCrit", False))}
    # Reflector moves (counter/mirrorcoat/metalburst/comeuppance) scale off the
    # damage the USER last took. The belief state carries that as an integer; pass
    # it through as-is. Absent -> the reflector fails (0), matching the fork.
    if move.get("lastHitDamage") is not None:
        out["lastHitDamage"] = int(move["lastHitDamage"])
    # T5 state-BP moves (bpCallback) — the transient battle state the belief state
    # reports so the calc can apply the fork's base-power rule (D39). All optional;
    # absent = the no-state baseline the fork would use.
    for key in ("hitThisTurn", "lastMoveFailed", "pledgeCombo"):  # bools
        if move.get(key) is not None:
            out[key] = bool(move[key])
    for key in ("allyFainted", "stockpileLayers"):  # ints
        if move.get(key) is not None:
            out[key] = int(move[key])
    if move.get("beatupAllies") is not None:  # list of mon specs (incl. user)
        out["beatupAllies"] = list(move["beatupAllies"])
    return out


def calc_damage(
    attacker: Mon,
    defender: Mon,
    move: MoveSpec,
    field: Optional[Dict[str, Any]] = None,
    fmt: Optional[str] = None,
    gen: int = 9,
) -> Dict[str, Any]:
    """Damage range (min/max/expected, in HP and % of defender max HP) + crit block.

    ``immune`` is True and all damage is 0 when the move cannot hit the defender —
    including TYPE IMMUNITY, which the fork applies to every move (e.g. Super Fang
    vs Ghost, Night Shade vs Normal, Fissure vs Flying/Levitate, Psywave vs Dark;
    fork-exact per-gen facts in ``special_move_immunity.json``).

    Special-damage moves (no base power) are completed with their official
    fork-verified formulas (@smogon/calc has no formula for these):
      * HP fractions / state: ``superfang``, ``ruination``, ``naturesmadness``,
        ``guardianofalola``, ``finalgambit``, ``endeavor``;
      * fixed level: ``nightshade``, ``seismictoss``;
      * OHKO (``ohko`` flag, guaranteed KO): ``fissure``, ``guillotine``,
        ``horndrill``, ``sheercold`` (immune vs Ice types);
      * random: ``psywave``;
      * reflectors — need ``move={'name':..., 'lastHitDamage': <prior hit>}`` —
        ``counter``, ``mirrorcoat``, ``metalburst``, ``comeuppance``.
    Formula moves respect type immunity but NOT 0.5x/2x multipliers or STAB
    (fork-verified). Explosion moves and faint-support moves
    (``healingwish``/``lunardance``/``memento``) report 0 damage as usual.

    State-dependent base-power moves (``bpCallback``) — the fork re-scales their
    base power from transient battle state; @smogon/calc has no formula for them
    (it would return the static data base power, or 0 for the base-0 ones). We
    complete the calc with the fork's formula (per-engine base powers in
    ``special_bp_moves.json``) and hand the effective base power to the library,
    which applies STAB/typing/boosts/immunity/crits normally (D39). Pass the
    relevant state via ``move={...}`` (all optional; absent = no-state baseline):
      * ``avalanche``            ``hitThisTurn=True``          (x2: user was hit this turn)
      * ``stompingtantrum``      ``lastMoveFailed=True``       (x2: user's previous move failed)
      * ``temperflare``          ``lastMoveFailed=True``       (x2: same)
      * ``firepledge``/``grasspledge``/``waterpledge``
                               ``pledgeCombo=True``            (BP 150: a compatible pledge hit this turn)
      * ``lastrespects``         ``allyFainted=<n>``           (BP 50 + 50*n fainted allies)
      * ``spitup``               ``stockpileLayers=<n>``       (BP 100*n; 0 -> the move fails)
      * ``beatup``               ``beatupAllies=[<mon>,...]``  (each ally: BP 5 + floor(baseAtk/10), own stats)
    Note: the pledges are banned in the champions engine (key 0), so they report 0 there.

    Pass ``fmt`` (a format id, e.g. ``"gen9bss3v3"``) so the correct engine is
    selected per format (champions formats use the library's champions
    mechanics, key 0 — not gen 9). ``gen`` is the raw engine key, used when
    ``fmt`` is not given.
    """
    req = {
        "type": "damage",
        "gen": _engine_key(gen, fmt),
        "attacker": _resolve_mon(attacker, fmt),
        "defender": _resolve_mon(defender, fmt),
        "move": _normalise_move(move),
        "field": field,
    }
    return _node(req)


def effective_speed(
    mon: Mon,
    fmt: Optional[str] = None,
    gen: int = 9,
    tailwind: bool = False,
    field: Optional[Dict[str, Any]] = None,
) -> int:
    """Effective action speed for a mon (full @smogon/calc mirror, see module docstring).

    ``field`` may set ``weather``/``terrain`` for the weather/terrain speed
    abilities (Swift Swim, Chlorophyll, Sand Rush, Slush Rush, Surge Surfer);
    the mon may set ``abilityOn``/``boostedStat`` for the state-gated abilities
    (Unburden, Slow Start, Quark Drive/Protosynthesis).
    """
    req = {"type": "speed", "gen": _engine_key(gen, fmt),
           "pokemon": _resolve_mon(mon, fmt), "tailwind": tailwind, "field": field}
    return int(_node(req)["speed"])


def compare_speed(
    a: Mon,
    b: Mon,
    fmt: Optional[str] = None,
    gen: int = 9,
    a_tailwind: bool = False,
    b_tailwind: bool = False,
    a_field: Optional[Dict[str, Any]] = None,
    b_field: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare two mons' effective speed. ``faster`` is 'a', 'b', or 'tie'."""
    sa = effective_speed(a, fmt, gen, a_tailwind, a_field)
    sb = effective_speed(b, fmt, gen, b_tailwind, b_field)
    faster = "a" if sa > sb else ("b" if sb > sa else "tie")
    return {"a": sa, "b": sb, "faster": faster}
