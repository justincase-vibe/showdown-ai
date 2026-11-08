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
            'immune': bool, 'maxhp': int, 'desc': str}

    calc.compare_speed(a, b, fmt="gen9bss3v3", gen=9)
        -> {'a': int, 'b': int, 'faster': 'a' | 'b' | 'tie'}

Prefer passing ``fmt`` (a format id from the legality registry): the engine
key is resolved from the format's ``mod`` (``genN`` -> N, ``champions`` -> 0 —
the library's native champions mechanics set, see PLAN.md D32). ``gen`` is
kept as the low-level engine key for callers that already have it; when both
are given, ``fmt`` wins.

attacker / defender / a / b are dicts using @smogon/calc option names:
``name`` (required), and any of ``level``, ``nature``, ``evs``, ``ivs``, ``item``,
``ability``, ``status`` ('par'/'brn'/...), ``boosts`` ({'atk':2,...}), ``teraType``,
``moves``. ``move`` is a move-name string (or ``{'name':..., 'isCrit':bool}``).
``field`` (optional) may set ``weather`` (capitalised: 'Rain'/'Sun'/'Sand'/'Hail'),
``terrain``, ``attackerSide`` and ``defenderSide`` (e.g. ``{'isReflect': True}``).

Speed model (see PLAN.md D30): base stat -> stat boosts -> Tailwind (x2) ->
Choice Scarf (x1.5) -> paralysis (x0.5 for gen>=7, matching @smogon/calc and this
project's sim fork).
"""
from __future__ import annotations

import json
import os
import subprocess
from typing import Any, Dict, Optional, Union

from legality import formats as _formats

_NODE = "node"
_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "node_calc.js")

Mon = Dict[str, Any]
MoveSpec = Union[str, Dict[str, Any]]


def _engine_key(gen: int, fmt: Optional[str]) -> int:
    """Resolve the @smogon/calc engine key: ``fmt`` wins over ``gen``."""
    if fmt is None:
        return gen
    return _formats.engine_key(fmt)


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
    return {"name": move["name"], "isCrit": bool(move.get("isCrit", False))}


def calc_damage(
    attacker: Mon,
    defender: Mon,
    move: MoveSpec,
    field: Optional[Dict[str, Any]] = None,
    fmt: Optional[str] = None,
    gen: int = 9,
) -> Dict[str, Any]:
    """Damage range (min/max/expected, in HP and % of defender max HP) + crit block.

    ``immune`` is True and all damage is 0 when the move cannot hit the defender.

    Pass ``fmt`` (a format id, e.g. ``"gen9bss3v3"``) so the correct engine is
    selected per format (champions formats use the library's champions
    mechanics, key 0 — not gen 9). ``gen`` is the raw engine key, used when
    ``fmt`` is not given.
    """
    req = {
        "type": "damage",
        "gen": _engine_key(gen, fmt),
        "attacker": attacker,
        "defender": defender,
        "move": _normalise_move(move),
        "field": field,
    }
    return _node(req)


def effective_speed(
    mon: Mon,
    fmt: Optional[str] = None,
    gen: int = 9,
    tailwind: bool = False,
) -> int:
    """Effective action speed for a mon, applying boosts/item/Tailwind/paralysis."""
    req = {"type": "speed", "gen": _engine_key(gen, fmt), "pokemon": mon, "tailwind": tailwind}
    return int(_node(req)["speed"])


def compare_speed(
    a: Mon,
    b: Mon,
    fmt: Optional[str] = None,
    gen: int = 9,
    a_tailwind: bool = False,
    b_tailwind: bool = False,
) -> Dict[str, Any]:
    """Compare two mons' effective speed. ``faster`` is 'a', 'b', or 'tie'."""
    sa = effective_speed(a, fmt, gen, a_tailwind)
    sb = effective_speed(b, fmt, gen, b_tailwind)
    faster = "a" if sa > sb else ("b" if sb > sa else "tie")
    return {"a": sa, "b": sb, "faster": faster}
