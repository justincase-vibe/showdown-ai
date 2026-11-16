"""Step 3.1 — belief event model.

A typed event = one observable fact from a battle, expressed the way a player
sees it. ``belief/log_events.py`` maps raw Showdown protocol lines (``|...``)
onto these events; the evidence engine (3.2) consumes them.

Design rules:
  * An event only carries what the log actually reveals at that moment —
    no hidden info, ever (rule 4).
  * Slot ids are Showdown identifiers: ``p1a``-``p1f`` (own side),
    ``p2a``-``p2f`` (opponent side — the belief's focus).
  * ids (moves/items/abilities/teratypes) are Showdown ``toID`` strings;
    statuses are short codes (``brn``/``par``/``skl``/``frz``/``tox``);
    stats are ``hp/atk/def/spa/spd/spe``.
  * ``Unknown`` is a first-class event: the adapter emits it for any line it
    does not map, so unmapped log lines are OBSERVABLE (bug-catcher, 3.7)
    instead of silently dropped.

All events expose ``.type`` (string) and ``.to_dict()`` (JSON-safe) so the
3.7 per-turn belief report and later serialization are trivial.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import ClassVar, Dict, Optional


@dataclass
class Event:
    """Base class. Subclasses set ``type`` and add fields (all defaulted)."""

    type: ClassVar[str] = "event"

    def to_dict(self) -> Dict:
        d: Dict = {"type": self.type}
        d.update(asdict(self))
        return d


@dataclass
class BattleStart(Event):
    type: ClassVar[str] = "battle_start"
    fmt: Optional[str] = None  # format id if the log names one


@dataclass
class Turn(Event):
    type: ClassVar[str] = "turn"
    n: int = 1


@dataclass
class SwitchIn(Event):
    """A mon enters the field. The line is `hp/maxhp`, so max HP (the
    denominator) is always revealed; a fresh mon has hp==max, a re-switching
    mon carries its current HP (e.g. `95/183`)."""

    type: ClassVar[str] = "switch_in"
    side: str = ""
    slot: str = ""
    species: str = ""
    hp: int = 0     # current HP on entry
    maxhp: int = 0  # max HP = the line's denominator (always revealed)"""


@dataclass
class SwitchOut(Event):
    type: ClassVar[str] = "switch_out"
    side: str = ""
    slot: str = ""


@dataclass
class Faint(Event):
    type: ClassVar[str] = "faint"
    side: str = ""
    slot: str = ""


@dataclass
class AbilityRevealed(Event):
    """The ability becomes visible (on entry, or by an effect)."""

    type: ClassVar[str] = "ability_revealed"
    side: str = ""
    slot: str = ""
    ability: str = ""


@dataclass
class MoveUsed(Event):
    """The move is revealed by being used. ``target_*`` may be None (self/field)."""

    type: ClassVar[str] = "move_used"
    side: str = ""
    slot: str = ""
    move: str = ""
    target_side: Optional[str] = None
    target_slot: Optional[str] = None


@dataclass
class MoveFailed(Event):
    type: ClassVar[str] = "move_failed"
    side: str = ""
    slot: str = ""
    move: str = ""


@dataclass
class HitCount(Event):
    """Actual hit count of a multi-hit move (needed later for damage windows)."""

    type: ClassVar[str] = "hit_count"
    side: str = ""
    slot: str = ""
    n: int = 1


@dataclass
class DamageObserved(Event):
    """Defender took damage. ``ko=True`` when the line is the KO line
    (``0 fnt``) — then ``hp_after==0`` and, per D40, the TRUE damage is
    ``>= maxhp`` (the log clips it). ``source_kind``: 'move' | 'residual'
    (burn/poison/recoil, i.e. a ``[from]`` line) | 'field' (hazards)."""

    type: ClassVar[str] = "damage_observed"
    target_side: str = ""
    target_slot: str = ""
    hp_after: int = 0
    maxhp: int = 0
    ko: bool = False
    source_kind: str = "move"
    move: Optional[str] = None
    source_side: Optional[str] = None
    source_slot: Optional[str] = None


@dataclass
class HealObserved(Event):
    type: ClassVar[str] = "heal_observed"
    target_side: str = ""
    target_slot: str = ""
    hp_after: int = 0
    maxhp: int = 0
    source_kind: str = "move"  # 'move' | 'residual'


@dataclass
class StatusApplied(Event):
    type: ClassVar[str] = "status_applied"
    side: str = ""
    slot: str = ""
    status: str = ""


@dataclass
class StatusCured(Event):
    type: ClassVar[str] = "status_cured"
    side: str = ""
    slot: str = ""
    status: str = ""


@dataclass
class BoostChanged(Event):
    type: ClassVar[str] = "boost_changed"
    side: str = ""
    slot: str = ""
    stat: str = ""
    delta: int = 0


@dataclass
class TeraUsed(Event):
    """Terastallization reveals the Tera type (gen 9 / champions only)."""

    type: ClassVar[str] = "tera_used"
    side: str = ""
    slot: str = ""
    tera_type: str = ""


@dataclass
class ItemRevealed(Event):
    """An item becomes visible (used, broken, or ``[from item: ...]`` context)."""

    type: ClassVar[str] = "item_revealed"
    side: str = ""
    slot: str = ""
    item: str = ""


@dataclass
class TailwindChanged(Event):
    """A side's Tailwind condition starts (x2 speed for the rest of its duration).

    Fork log line: ``|-sidestart|<side>|move: Tailwind`` (``side.ts
    addSideCondition`` + the move's ``onSideStart``; verified in the fork,
    2026-11-16). Duration 4 turns (6 with Persistent — an edge the 3.7 fuzz
    reports if it ever matters). The belief expires it on a turn count (see
    ``BeliefState.field``); the exact off-by-one is irrelevant to the evidence
    (Tailwind only ever makes a mon FASTER, and ties are consistent anyway)."""

    type: ClassVar[str] = "tailwind_changed"
    side: str = ""
    active: bool = True


@dataclass
class WeatherChanged(Event):
    type: ClassVar[str] = "weather_changed"
    weather: str = ""  # 'rain' | 'sun' | 'sand' | 'hail' | 'snow' | 'none'


@dataclass
class TerrainChanged(Event):
    type: ClassVar[str] = "terrain_changed"
    terrain: str = ""  # 'electric' | 'grassy' | 'misty' | 'psychic' | 'none'


@dataclass
class BattleEnd(Event):
    type: ClassVar[str] = "battle_end"
    winner_side: Optional[str] = None  # None = tie


@dataclass
class Unknown(Event):
    """A log line the adapter does not map. Observable on purpose: the 3.7
    fuzz reports how many of these appear in real battles, so gaps in this
    model are caught (and extend the mapping) instead of silently lost."""

    type: ClassVar[str] = "unknown"
    raw: str = ""


#: every concrete event class, by tag — handy for tests and the report
EVENT_TYPES = [
    BattleStart, Turn, SwitchIn, SwitchOut, Faint, AbilityRevealed, MoveUsed,
    MoveFailed, HitCount, DamageObserved, HealObserved, StatusApplied,
    StatusCured, BoostChanged, TeraUsed, ItemRevealed, WeatherChanged,
    TerrainChanged, TailwindChanged, BattleEnd, Unknown,
]
