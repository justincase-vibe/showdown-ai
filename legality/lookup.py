"""Step 1 lookup API (build plan: "Move / Species Database").

Read-only query layer over ``legality.db`` for the bot:

    get_move(name, gen=9)           -> move props + parsed effect flags
    get_species(name, gen=9)        -> types, base stats, abilities, tier
    legal_moves(species, fmt)       -> move ids legal in a format
    get_common_sets(species, fmt)   -> Smogon sets w/ normalized probability
    get_type_chart(gen=9)           -> {defending: {attacking: factor}}
    battle_level(fmt, level=None)   -> level a mon fights at in the format (D33)

Names accept display names or ids (normalized with Showdown ``toID``).
``gen`` for ``get_move`` is the @smogon/calc ENGINE KEY (0 = champions, 1-9 =
gen) — move data is mechanics-scoped (champions reuses gen-9 species but
different moves). The gen-9 ``move`` row is the base; a ``move_gen`` override
(extract_move_gen.js, fork-sourced) replaces the per-gen data. ``get_species``
and ``get_type_chart`` still take the raw generation (1-9).

Raises ``KeyError`` for unknown moves/species; returns ``[]`` where a
"no data" answer is meaningful (no sets for a species, …).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from . import api, db

#: side_condition values that mean "hazard setter" (field conditions)
HAZARDS = frozenset({"stealthrock", "spikes", "stickybarbedsteel"})
#: side_condition values that mean "screen setter"
SCREENS = frozenset({"reflect", "lightscreen"})


def to_id(name: str) -> str:
    """Showdown toID: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


# ── moves ────────────────────────────────────────────────────────────
#: data columns a move_gen override may carry (the rest — id/name/desc/
#: side_condition/boosts/drain/recoil — stay on the gen-9 base row).
_MOVE_GEN_OVERRIDES = ("type", "category", "power", "accuracy", "pp", "priority")


def get_move(name: str, gen: int = 9) -> Dict[str, Any]:
    """Move properties + parsed effect flags, PER-GEN.

    ``gen`` is the @smogon/calc ENGINE KEY (0 = champions, 1-9 = gen; see
    ``legality.formats.engine_key``). The gen-9 ``move`` row is the base; a
    ``move_gen`` override (extract_move_gen.js, fork-sourced) replaces the
    per-gen data where the move changed — e.g. Dragon Claw is Special in gen
    1-3, Protect is pri 2 in gen 1-2 / pri 3 in gen 3-4, Slash is 80 BP in
    champions. Effect flags (side_condition/boosts/drain/recoil) stay gen-9 on
    the base row (the calc models those per-gen via its JSON tables).

    Returns a dict with: id, name, type, category, power, accuracy, pp,
    priority, desc, boosts (parsed JSON, e.g. {"atk": 2} or None),
    drain/recoil (integer percent or None), and boolean flags:
    is_hazard, is_screen, is_tailwind (from the server's side_condition).
    """
    mid = to_id(name)
    conn = db.connect()
    try:
        base = conn.execute("SELECT * FROM move WHERE id=?", (mid,)).fetchone()
        if base is None:
            raise KeyError(f"unknown move: {name!r}")
        d = dict(base)
        # A1: per-gen override — replace the data columns where the move changed
        # in this gen. The override row is self-contained (all 6 data columns
        # present, incl. NULL for a status move), so override when the column is
        # in the row (a NULL is a real change, e.g. damage -> status that gen).
        ov = conn.execute(
            "SELECT * FROM move_gen WHERE move_id=? AND gen=?", (mid, gen)).fetchone()
        if ov is not None:
            o = dict(ov)
            for col in _MOVE_GEN_OVERRIDES:
                if col in o:
                    d[col] = o[col]
        d["boosts"] = json.loads(d["boosts"]) if d.get("boosts") else None
        sc = d.get("side_condition")
        d["is_hazard"] = sc in HAZARDS
        d["is_screen"] = sc in SCREENS
        d["is_tailwind"] = sc == "tailwind"
        return d
    finally:
        conn.close()


# ── species ──────────────────────────────────────────────────────────
def get_species(name: str, gen: int = 9) -> Dict[str, Any]:
    """Form row + gen-scoped attributes.

    Returns: id, name, base_id (base species), forme, flags
    (is_mega/is_primal/is_gmax), types [display names], stats
    {hp, atk, def, spa, spd, spe}, abilities [{id, name, hidden}], tier.
    """
    fid = to_id(name)
    conn = db.connect()
    try:
        r = conn.execute("SELECT * FROM form WHERE id=?", (fid,)).fetchone()
        if r is None:
            raise KeyError(f"unknown species: {name!r}")
        d = dict(r)
        d["types"] = [row["type"] for row in conn.execute(
            "SELECT type FROM form_type WHERE form_id=? AND gen=? ORDER BY type",
            (fid, gen))]
        st = conn.execute(
            "SELECT hp, atk, def, spa, spd, spe FROM form_stats WHERE form_id=? AND gen=?",
            (fid, gen)).fetchone()
        d["stats"] = dict(st) if st else None
        d["abilities"] = [
            {"id": a["ability_id"], "name": a["name"], "hidden": bool(a["is_hidden"])}
            for a in conn.execute(
                "SELECT fa.ability_id AS ability_id, a.name AS name, fa.is_hidden AS is_hidden "
                "FROM form_ability fa JOIN ability a ON a.id=fa.ability_id "
                "WHERE fa.form_id=? AND fa.gen=? ORDER BY fa.ability_id",
                (fid, gen))]
        tier = conn.execute(
            "SELECT tier FROM form_tier WHERE form_id=? AND gen=?", (fid, gen)).fetchone()
        d["tier"] = tier["tier"] if tier else None
        return d
    finally:
        conn.close()


def legal_moves(species: str, fmt: str) -> List[str]:
    """Move ids legal for the species in the format (Fissure rule, D7).

    Sorted by move id. Empty list if the species has no legal moves there
    (or is not in the format at all)."""
    api._require_format(fmt)
    fid = to_id(species)
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT move_id FROM format_form_move WHERE format_id=? AND form_id=? "
            "ORDER BY move_id", (fmt, fid)).fetchall()
        return [r["move_id"] for r in rows]
    finally:
        conn.close()


# ── common sets (Smogon) ─────────────────────────────────────────────
def get_common_sets(species: str, fmt: str) -> List[Dict[str, Any]]:
    """Scraped (Smogon) sets for a species in a format.

    Each set: id, title, moves (ordered move ids), item_id, ability_id,
    nature, evs (spread string), tera_type, source_url, and ``probability``
    = 1 / (number of sets for that species in that format) — so probabilities
    for a species sum to exactly 1. Empty list if the species has no sets.
    """
    api._require_format(fmt)
    fid = to_id(species)
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT * FROM strategy WHERE format_id=? AND form_id=? ORDER BY id",
            (fmt, fid)).fetchall()
        if not rows:
            return []
        p = 1.0 / len(rows)
        out = []
        for s in rows:
            d = dict(s)
            d["moves"] = [r["move_id"] for r in conn.execute(
                "SELECT move_id FROM strategy_move WHERE set_id=? ORDER BY ord",
                (s["id"],))]
            d["probability"] = p
            out.append(d)
        return out
    finally:
        conn.close()


def common_set_species(fmt: str) -> List[str]:
    """Species ids (form ids) that have at least one set in the format."""
    api._require_format(fmt)
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT form_id FROM strategy WHERE format_id=? ORDER BY form_id",
            (fmt,)).fetchall()
        return [r["form_id"] for r in rows]
    finally:
        conn.close()


# ── type chart ───────────────────────────────────────────────────────
def get_type_chart(gen: int = 9) -> Dict[str, Dict[str, float]]:
    """{defending_type: {attacking_type: factor}} for the standard 18x18 matrix.

    Keys are display names ("Fire", "Water", …) to match ``move.type`` and
    ``form_type.type`` in the DB — so damage math can do
    ``chart[defender_type][move_type]`` directly without conversion. The
    table stores lowercase ids; display names are the id capitalized (true
    for all 18 real types).

    factor: 0 = immune, 0.5 = resist, 1 = neutral, 2 = super-effective.
    (Values exactly as the local server's sim computes them — this build's
    typechart.ts; see PLAN.md §3 verified facts.)
    """
    conn = db.connect()
    try:
        chart: Dict[str, Dict[str, float]] = {}
        for r in conn.execute(
                "SELECT attacking, defending, factor FROM type_chart WHERE gen=?",
                (gen,)):
            chart.setdefault(r["defending"].capitalize(), {})[r["attacking"].capitalize()] = r["factor"]
        return chart
    finally:
        conn.close()


# ── format level rules (D33) ─────────────────────────────────────────
def battle_level(fmt: str, level: Optional[int] = None) -> int:
    """Level a mon fights at in ``fmt``, mirroring the fork's team-validator.

    Mirrors ``sim/team-validator.ts`` (validateSet, lines 611–615 + 1144):
    unset level -> the format's default; ``adjustLevelDown`` lowers any level
    at/above it to itself; ``adjustLevel`` forces an exact level. So in BSS
    3v3 / VGC (Adjust Level Down 50) a standard level-100 team fights at 50;
    in champions / gen1lc (Adjust Level) the level is always 50 / 5.

    Raises ``ValueError`` for an unknown format or when an explicit ``level``
    is outside the format's [minLevel, maxLevel] — the validator would reject
    such a set, so the calc refuses to invent numbers for it (no silent
    level-100 default in a level-50 format).
    """
    api._require_format(fmt)
    conn = db.connect()
    try:
        r = conn.execute(
            "SELECT min_level, max_level, adjust_level, adjust_level_down, default_level"
            " FROM format WHERE id=?", (fmt,)).fetchone()
    finally:
        conn.close()
    if r is None:
        raise KeyError(f"format '{fmt}' not in DB")
    lvl = r["default_level"] if level is None else int(level)
    if level is not None and not (r["min_level"] <= lvl <= r["max_level"]):
        raise ValueError(
            f"level {level} is outside [{r['min_level']}, {r['max_level']}] for format '{fmt}'")
    adj_down = r["adjust_level_down"]
    if adj_down is not None and lvl >= adj_down:
        lvl = adj_down
    adj = r["adjust_level"]
    if adj is not None:
        lvl = adj
    return lvl


if __name__ == "__main__":
    # tiny smoke demo
    import pprint
    pprint.pprint(get_move("Stealth Rock"))
    pprint.pprint({k: v for k, v in get_species("Garchomp").items()
                   if k not in ("nonstandard_reason",)})
    pprint.pprint(get_common_sets("Garchomp", "gen9ou")[:1])
    pprint.pprint(get_type_chart()["Dragon"])
