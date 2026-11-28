"""S3.4 action-mix metric — classify the learner's sampled actions into
attack / switch / hazard / setup / status / other.

Derived from the MOVE DB fields (``category``, ``side_condition``,
``boosts``) — the same data-driven discipline as the S6.8 evaluator: NO
move-name lists, NO species tags, NO hand-written role lists.

Class definitions (plan S3.4 + S6.8):
  * hazard  — the move sets a FIELD HAZARD (switch-in damage): the DB's
    ``side_condition`` is one of the four hazard conditions. (Screens and
    protection are also ``side_condition`` values in the DB; they are NOT
    hazards here and fall through to ``status``.)
  * setup   — self-boost total >= +2 in one use (S6.8 definition;
    Swords Dance +2, Calm Mind +1/+1, Bulk Up +1/+1; Aromatis Mist +1 is
    NOT setup).
  * status  — Status-category moves that are neither hazard nor setup
    (status infliction, screens, protect, tailwind, ...).
  * attack  — Physical/Special-category damaging moves.
  * switch  — the action's kind is ``switch``.
  * other   — everything else (tera, forfeit, unclassified).

Pure functions + one DB-backed lookup builder, so classification is
unit-testable with fake move rows (no DB, no torch).
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Mapping, Optional

# side_condition values that are field hazards (switch-in damage class).
# The DB marks screens/protection as side_conditions too; those are NOT
# hazards and fall through to the status bucket.
HAZARD_CONDITIONS = frozenset({"spikes", "stealthrock", "toxicspikes",
                               "stickyweb"})

CLASSES = ("attack", "switch", "hazard", "setup", "status", "other")


def _setup_total(boosts: Optional[str]) -> int:
    """Sum of the SELF positive boosts in one use (S6.8: setup = >= +2)."""
    if not boosts:
        return 0
    try:
        b = json.loads(boosts)
    except (ValueError, TypeError):
        return 0
    if not isinstance(b, dict):
        return 0
    return sum(v for v in b.values() if isinstance(v, int) and v > 0)


def classify_move(move: Mapping[str, Any]) -> str:
    """Classify ONE move row (a dict with the DB field names).

    Priority: hazard > setup > attack > status. A damaging move can never
    be a hazard/setup/status move in the data, so the order is safe.
    """
    if move.get("side_condition") in HAZARD_CONDITIONS:
        return "hazard"
    if _setup_total(move.get("boosts")) >= 2:
        return "setup"
    cat = move.get("category")
    if cat in ("Physical", "Special"):
        return "attack"
    if cat == "Status":
        return "status"
    return "other"


def classify_action(action: Optional[Mapping[str, Any]],
                    moves: Mapping[str, Mapping[str, Any]]) -> str:
    """Classify ONE sampled action dict (``{kind, move|species, ...}``)."""
    if not action:
        return "other"
    kind = action.get("kind")
    if kind == "switch":
        return "switch"
    if kind == "move":
        m = moves.get(action.get("move") or "")
        if m is not None:
            return classify_move(m)
        return "other"
    return "other"  # tera, forfeit, unknown kinds


def action_mix(actions: Iterable[Optional[Mapping[str, Any]]],
               moves: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Share of each action class over ``actions`` (every class always
    present, 0.0 when none; all-zero input -> all 0.0)."""
    counts = {c: 0 for c in CLASSES}
    for a in actions:
        counts[classify_action(a, moves)] += 1
    n = sum(counts.values())
    if n == 0:
        return {c: 0.0 for c in CLASSES}
    return {c: counts[c] / n for c in CLASSES}


def build_moves_lookup(db_path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """move id -> {category, side_condition, boosts} (only the 3 fields the
    classifier reads). One DB read; call once per process and cache."""
    from legality import db
    conn = db.connect(db_path) if db_path else db.connect()
    try:
        rows = conn.execute(
            "SELECT id, category, side_condition, boosts FROM move").fetchall()
    finally:
        conn.close()
    return {r[0]: {"category": r[1], "side_condition": r[2], "boosts": r[3]}
            for r in rows}
