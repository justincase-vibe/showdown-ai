"""Canonical "revealed-only" protocol shared by BOTH sides of the pipeline.

The single most important invariant in this project: the observation the policy sees
during **imitation training** (reconstructed from datamined logs by `state_from_log.py`)
must be built with the exact same rules as the observation it sees during **live play**
(masked view in `env.py`). Anything that differs between the two is a train/live
mismatch and silently poisons training.

The canonical rules (see PLAN.md §3.1 "Encoding protocol"):

1. **Species** — always known from the team protocol (`|poke|` line / request details).
   Hashed via ``mon.species`` (poke_env's to_id form), never ``mon.name`` (the player's
   nickname, which is only stable in replay and would leak an untrained signal live).
2. **Moves** — only moves *used so far*, in *order of first use* (per Pokémon).
   Move slot ``i`` of the observation == move ``i`` of this reveal list.
   Gimmick names that have no base Move object (G-Max) occupy a zero-filled slot.
   PP fraction is constant 1.0 on both sides (PP is not recoverable from logs).
3. **Ability** — the exact ability only when it is (a) the species' only ability, or
   (b) revealed by a ``|-ability|`` log line. Otherwise ``None`` (zero hash).
4. **Item** — only when revealed by a ``|-item|`` / ``|-enditem|`` log line;
   otherwise the ``UNKNOWN_ITEM`` default (a consistent "unknown" hash on both sides).
5. **Action mapping** — switch 0-5 = team slot; move slot ``i`` = ``i``-th move of the
   reveal list; unseen moves fill from the moveset's request order. Gimmick bands
   (10-13 mega, 14-17 z, 18-21 dynamax, 22-25 tera) use the same slot ``i``.

This module provides the name maps, the move-name normalizer, the shared log-event
reveal parser (used by the live masked view), and the zero-move placeholder (used by
the replay reconstruction). `datamine/parse.py` imports the maps from here, so there is
one source of truth.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

# ---------------------------------------------------------------------------
# Gimmick move name maps (single source of truth; parse.py imports these)
# ---------------------------------------------------------------------------

# Z-move (gen 7) → base move. The log shows the Z-name; the base move is what exists
# in the moveset and in poke_env's Move data.
ZMOVE_TO_BASE: Dict[str, str] = {
    "Bursting Dreampulse": "Moonblast",
    "Continental Crush": "Rock Slide",
    "Malicious Moonsault": "Play Rough",
    "Sunne Bomb": "Flamethrower",
    "Hydro Vortex": "Water Gun",
    "Hydro Cannon": "Hydro Pump",
    "Supersonic Skystrike": "Air Slash",
    "Cloud Burst": "Rain Dance",
    "10,000,000 Volt Thunderbolt": "Thunderbolt",
    "Bolt Strike": "Thunder",
    "Supernova": "Solar Beam",
    "Healing Wish": "Heal Pulse",
    "Black Luster": "Psychic",
    "Malachar": "Dark Pulse",
    "Let me be the one to judge you.": "Judgment",
    "Colossal Cadence": "Hyper Voice",
    "Soul-Stealing 7-Star Strike": "Giga Drain",
    "Menacing Rising Moon": "Moon Force",
    "Gigavolts": "Wild Charge",
    "Acid Downpour": "Acid",
    "Breakneck Blitz": "Extreme Speed",
    "Cardboard Crash": "Double-Edge",
    "Devastating Drake": "Outrage",
    "Savage Spin-Out": "Twister",
    "All-Out Pummeling": "Close Combat",
    "Genesis Wave": "Bullet Seed",
    "Sinister Slash": "Swords Dance",
    "Raging Fury": "Fire Fang",
    "Gear Grind": "Flash Cannon",
    "Spectral Thief": "Shadow Ball",
    "Befouled Beak": "Fly",
    "Glacial Lance": "Ice Beam",
    "Tectonic Rage": "Earthquake",
    "Black Hole Eclipse": "Psychic",
}

# G-Max moves (gen 8, Gigantamax only). These names have NO base Move object in
# poke_env's data and no recoverable base move in the log — they occupy a
# zero-filled move slot (see `_ZeroMove`) but still count as a revealed move.
GMAX_MOVES = {
    "G-Max Wildfire", "G-Max Volcalith", "G-Max Foam Burst", "G-Max Steelsurge",
    "G-Max Sandblast", "G-Max Finale", "G-Max Centiferno", "G-Max Fly",
    "G-Max Mud Shot", "G-Max Hailstorm", "G-Max Bubblebeam", "G-Max Lightning",
    "G-Max Starfall", "G-Max Vine Lash", "G-Max Meltdown", "G-Max Cannonade",
    "G-Max Stun Shock",
}


def normalize_move_name(move_name: str, gen: int) -> str:
    """Map a log move name to the canonical reveal-list name.

    * gen 7 Z-names → base move name (the base move is what the moveset holds).
    * G-Max names (gen 8) → kept as-is (no Move object; zero slot downstream).
    * everything else → as-is.
    """
    if gen == 7 and move_name in ZMOVE_TO_BASE:
        return ZMOVE_TO_BASE[move_name]
    return move_name


# ---------------------------------------------------------------------------
# Zero-move placeholder: a "revealed move with no properties" (G-Max slots).
# obs.py reads every attribute through _safe(), which turns AttributeErrors into
# defaults (0/None) — exactly the zero vector we want.
# ---------------------------------------------------------------------------
class _ZeroMove:
    """Attribute access intentionally fails so obs._safe() yields zero defaults."""

    id = "unrevealed_gimmick_move"

    def __getattr__(self, name):  # pragma: no cover - exercised via obs._safe
        if name.startswith("__"):
            raise AttributeError(name)
        raise AttributeError(f"_ZeroMove has no {name!r}")


# ---------------------------------------------------------------------------
# Shared log-event reveal parser (used by the LIVE masked view in env.py)
# ---------------------------------------------------------------------------

def _side_ident(field: str) -> Tuple[str, str]:
    """'|move|p1a: Garchomp' → ('p1', 'Garchomp')."""
    role = "p1" if field.startswith("p1") else "p2"
    ident = field.split(": ", 1)[1] if ": " in field else field
    return role, ident


def parse_reveals(
    events: Iterable[str], gen: int
) -> Tuple[
    Dict[str, Dict[str, List[str]]],
    Dict[str, Dict[str, str]],
    Dict[str, Dict[str, str]],
]:
    """Parse Showdown protocol events (strings like ``|move|p1a: X|Scald|...``) into
    per-ident reveal structures, exactly as the replay reconstruction does them:

    * ``moves``: role → ident → [move names in order of first use, normalized]
    * ``abilities``: role → ident → ability name (from ``|-ability|`` lines)
    * ``items``: role → ident → item name (from ``|-item|`` / ``|-enditem|`` lines)
    """
    moves: Dict[str, Dict[str, List[str]]] = {"p1": {}, "p2": {}}
    abilities: Dict[str, Dict[str, str]] = {"p1": {}, "p2": {}}
    items: Dict[str, Dict[str, str]] = {"p1": {}, "p2": {}}

    for ev in events:
        if not isinstance(ev, str) or not ev.startswith("|"):
            continue
        parts = ev.split("|")
        if len(parts) < 3:
            continue
        tag = parts[1]

        if tag == "move" and len(parts) >= 4:
            role, ident = _side_ident(parts[2])
            name = normalize_move_name(parts[3], gen)
            lst = moves[role].setdefault(ident, [])
            if name not in lst:
                lst.append(name)
        elif tag == "-ability" and len(parts) >= 4:
            role, ident = _side_ident(parts[2])
            abilities[role][ident] = parts[3]
        elif tag in ("-item", "-enditem") and len(parts) >= 4:
            role, ident = _side_ident(parts[2])
            items[role][ident] = parts[3]

    return moves, abilities, items
