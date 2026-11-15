"""Step 3.1 — mini Showdown-log → belief-event adapter.

Parses raw battle-log lines (the ``|tag|payload...`` protocol) into the typed
events of ``belief/events.py``. Scope for 3.1: map the lines that carry
evidence for the belief (switches, reveals, moves, damage, statuses, boosts,
Tera, items, weather/terrain, outcome) and surface everything else:
known-noise lines are ignored, genuinely unmapped lines become ``Unknown``
events so the 3.7 fuzz can count and report them (observable bug-catcher).

The adapter tracks per-slot context (current species, max HP) so that
evidence events can be attributed correctly — that context is adapter-local
state, never part of the event stream itself.

Two fork log-semantics facts baked in (verified in Step 2 keepers):
  * The sim logs each damage/heal event TWICE — absolute ``hp/maxhp`` and a
    ``%/100`` line. The true line's denominator is the mon's max HP; max HP
    is learned from the slot's switch-in line (always ``max/max``), and any
    line whose denominator doesn't match is dropped. (D34 harness note.)
  * A KO logs ``|-damage|slot|0 fnt`` — the true damage is >= maxhp (D40).
  * The sim logs each switch-in TWICE, back-to-back (real ``hp/maxhp`` plus a
    ``%/100`` duplicate) — identical-consecutive switch lines are deduped, the
    same way as the doubled HP lines.

Ability reveal (verified against the fork, 2026-11-15 — probe_ability_reveal):
  * A mon's ability is NOT revealed at switch-in — the ``|switch|`` line
    carries species + HP only (``pokemon.getFullDetails``). Passive abilities
    (Blaze, Sand Veil, …) never emit a ``|-ability|`` line on entry. So in the
    belief model the ability is a HIDDEN variable, kept in the prior (all of
    the species' abilities) and narrowed only when a reveal fires (3.3).
  * An ability IS revealed contextually, via the fork's ``[from] <kind>: <Name>``
    convention (``[from] ability: Drizzle`` / ``[from] item: Choice Band``):
    weather starts (``|-weather|``), status inflicts (``|-status|``), ability
    changes/boosts (``|-ability|``). The ability HOLDER is the ``[of] <slot>``
    source, NOT the target slot (a -status line targets the victim, but the
    ability is the source's) — so reveals are attributed via ``[of]``.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .events import (
    AbilityRevealed,
    BattleEnd,
    BattleStart,
    BoostChanged,
    DamageObserved,
    Event,
    Faint,
    HealObserved,
    HitCount,
    ItemRevealed,
    MoveFailed,
    MoveUsed,
    StatusApplied,
    StatusCured,
    SwitchIn,
    SwitchOut,
    TeraUsed,
    TerrainChanged,
    Turn,
    Unknown,
    WeatherChanged,
)

#: tags that carry no belief evidence — silently skipped (NOT Unknown)
IGNORED_TAGS = frozenset({
    # battle meta (no set evidence)
    "hint", "-hint", "name", "player", "teamsize", "rating", "gen", "tier",
    "battle_type", "pokestats", "details", "value", "rule", "ruleset",
    "team", "pokeball", "t:", "gametype", "clearpoke", "poke", "teampreview",
    "showteam", "rated", "chat", "message", "-message", "html", "debug",
    "bigerror", "detailschange",
    # visual/ambient effects (no set evidence)
    "-center", "-block", "-activate", "-clearmessage", "-singlemove",
    "-mustrecharge", "-warpplate", "-clearspikes", "-sidestart",
    "split", "upkeep", "fail", "-fail", "-start", "-end", "-miss",
    "-transform", "-primal", "-formechange", "-enditem", "swap", "cant",
    # non-BSS / gen-specific (no 3.1 evidence)
    "-mega", "-zpower", "-zbroken", "-burst", "-candynamax",
    # damage MODIFIERS — not set evidence; promoted to events in 3.4 (type
    # effectiveness, crit, OHKO) so the 3.1 smoke stays clean (Unknown = the
    # genuinely-unexpected signal the 3.7 fuzz reports).
    "-immune", "-resisted", "-supereffective", "-crit", "-ohko",
    # whole-boost resets (the per-stat -boost/-unboost is the evidence)
    "-clearboost", "-clearpositiveboost", "-clearnegativeboost",
})
# NOTE: ``-weather`` is NOT ignored — it is the ability-revealing weather-start
# line (``|-weather|RainDance|[from] ability: Drizzle|[of] p1a``) and maps to
# WeatherChanged (plus an AbilityRevealed via the [from] capture below).

#: [from] <kind>: <Name> — the fork's context convention (conditions.js/battle.js):
#:   [from] ability: Drizzle   (ability holder = the [of] slot, NOT the target)
#:   [from] item: Choice Band
_FROM_KIND = re.compile(r"\[from\]\s*(ability|item):\s*([^|\]]+)")
#: [of] p2a: Charizard — the source/holder slot for a [from]-context reveal
_OF_SLOT = re.compile(r"\[of\]\s*(p[12][a-f])")


def to_id(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _slot_of(payload: str) -> Tuple[str, str]:
    """'p2a: Blastoise' -> ('p2', 'p2a')."""
    if ":" in payload:
        slot = payload.split(":", 1)[0].strip()
    else:
        slot = payload.strip()
    return slot[:2], slot


def _int(s: str) -> int:
    return int(s.strip())


class LogAdapter:
    """Stateful line→event parser.

    Adapter-local context (NOT part of the event stream):
      * ``species_by_slot`` — the mon currently in each slot (switch lines);
      * ``maxhp_by_slot``   — max HP ground truth from switch-in (max/max).
    """

    def __init__(self) -> None:
        self.species_by_slot: Dict[str, str] = {}
        self.maxhp_by_slot: Dict[str, int] = {}
        self._maxhp_species: Dict[str, str] = {}  # which mon owns each slot's maxhp
        self._items_revealed: set = set()
        self._abilities_revealed: set = set()
        # hp-line dedupe (the sim logs each hp change twice, back-to-back):
        self._prev_line_was_hp = False       # was the previous fed line an hp event?
        self._last_hp_sig = None             # ('d'|'h', slot, hp_after, maxhp)
        # switch-in dedupe (the sim logs each switch-in twice, back-to-back):
        self._prev_line_was_switch = False   # was the previous fed line a switch-in?
        self._last_switch_sig = None         # (slot, species) of that switch-in
        self.events: List[Event] = []

    # ── public ─────────────────────────────────────────────────────────
    def feed_line(self, line: str) -> Optional[Event]:
        line = line.strip()
        if not line.startswith("|"):
            self._reset_line_tracking()
            return None
        parts = line.split("|")
        tag = parts[1] if len(parts) > 1 else ""
        if not tag:
            self._reset_line_tracking()
            return None  # empty `|` separator line — skip silently
        try:
            ev: Optional[Event] = self._map(tag, parts, line)
        except Exception as e:  # malformed/unexpected line: surface it, don't abort the battle
            import sys as _sys
            print(f"  [belief] unmapped line {line!r}: {type(e).__name__}: {e}", file=_sys.stderr)
            ev = Unknown(raw=line)
        if ev is not None:
            self.events.append(ev)
        # track hp-line consecutiveness for the back-to-back duplicate dedupe
        is_hp = isinstance(ev, (DamageObserved, HealObserved))
        if is_hp:
            self._last_hp_sig = (("d" if isinstance(ev, DamageObserved) else "h"),
                                 ev.target_slot, ev.hp_after, ev.maxhp)
        self._prev_line_was_hp = is_hp
        # track switch-in consecutiveness (the sim logs each switch-in twice)
        self._prev_line_was_switch = isinstance(ev, SwitchIn)
        if self._prev_line_was_switch:
            self._last_switch_sig = (ev.slot, ev.species)
        # [from] ability:/item: context on ANY line reveals that kind
        # (weather-start, status-inflict, switch-out Choice, consumed berry,
        # ...) — deduped per slot. The holder is the [of] slot (or the primary
        # payload when it is a slot), NOT the target slot.
        m = _FROM_KIND.search(line)
        if m:
            kind, name = m.group(1), m.group(2)
            pair = self._find_slot(line, parts)
            if pair is not None:
                side, slot = pair
                if kind == "item":
                    item = to_id(name)
                    if (slot, item) not in self._items_revealed:
                        self._items_revealed.add((slot, item))
                        self.events.append(ItemRevealed(side=side, slot=slot, item=item))
                elif kind == "ability":
                    ability = to_id(name)
                    if (slot, ability) not in self._abilities_revealed:
                        self._abilities_revealed.add((slot, ability))
                        self.events.append(AbilityRevealed(side=side, slot=slot, ability=ability))
        return ev

    def feed_lines(self, lines: List[str]) -> List[Event]:
        for line in lines:
            self.feed_line(line)
        return self.events

    # ── mapping ────────────────────────────────────────────────────────
    def _map(self, tag: str, parts: List[str], raw: str) -> Optional[Event]:
        p = parts[2:]  # payloads

        if tag == "start":
            return BattleStart(fmt=p[0] if p else None)
        if tag in IGNORED_TAGS:
            return None
        if tag == "turn":
            return Turn(n=_int(p[0]))

        if tag == "switch":
            side, slot = _slot_of(p[0])
            species = to_id(p[1].split(",")[0])
            # the sim logs each switch-in twice, back-to-back — drop the dup
            if self._prev_line_was_switch and self._last_switch_sig == (slot, species):
                return None
            self.species_by_slot[slot] = species
            self._learn_maxhp(slot, p[2])
            return SwitchIn(side=side, slot=slot, species=species)
        if tag == "leave":
            side, slot = _slot_of(p[0])
            return SwitchOut(side=side, slot=slot)
        if tag == "faint":
            side, slot = _slot_of(p[0])
            return Faint(side=side, slot=slot)

        if tag == "-ability" and len(p) >= 2:
            side, slot = _slot_of(p[0])
            return AbilityRevealed(side=side, slot=slot, ability=to_id(p[1]))

        if tag == "move" and len(p) >= 2:
            # |move|source|MoveName|target — battle-actions.js logs
            # addMove("move", pokemon, movename, target): move is p[1].
            side, slot = _slot_of(p[0])
            move = to_id(p[1])
            if len(p) >= 3:
                t_side, t_slot = _slot_of(p[2])
            else:
                t_side = t_slot = None
            return MoveUsed(side=side, slot=slot, move=move,
                            target_side=t_side, target_slot=t_slot)
        if tag == "movefail" and len(p) >= 2:
            # Upstream log format |movefail|slot|MoveName. NOTE: this fork never
            # emits `movefail` (verified in dist/sim + client parser) — a failed
            # move logs the `|move|` announce line first, then `|-fail|<user>`
            # (no move name; stays in IGNORED_TAGS). The move name is already in
            # the announce, and the absence of a following damage line IS the
            # observable failure signal the 3.2 evidence engine consumes.
            side, slot = _slot_of(p[0])
            return MoveFailed(side=side, slot=slot, move=to_id(p[1]))

        if tag == "-hitcount" and len(p) >= 2:
            side, slot = _slot_of(p[0])
            return HitCount(side=side, slot=slot, n=_int(p[1]))

        if tag == "-damage" and len(p) >= 2:
            return self._hp_event(p[0], p[1], damage=True)
        if tag == "-heal" and len(p) >= 2:
            return self._hp_event(p[0], p[1], damage=False)
        if tag == "-side damage" and len(p) >= 2:
            ev = self._hp_event(p[0], p[1], damage=True)
            if ev is not None:
                ev.source_kind = "field"
            return ev

        if tag == "-status" and len(p) >= 2:
            side, slot = _slot_of(p[0])
            return StatusApplied(side=side, slot=slot, status=p[1])
        if tag == "-curestatus" and len(p) >= 2:
            side, slot = _slot_of(p[0])
            return StatusCured(side=side, slot=slot, status=p[1])

        if tag in ("-boost", "-unboost", "-layerchange", "-setboost") and len(p) >= 3:
            side, slot = _slot_of(p[0])
            delta = -abs(_int(p[2])) if tag == "-unboost" else _int(p[2])
            return BoostChanged(side=side, slot=slot, stat=p[1], delta=delta)

        if tag in ("terastallize", "-terastallize") and len(p) >= 2:
            side, slot = _slot_of(p[0])
            return TeraUsed(side=side, slot=slot, tera_type=to_id(p[1]))

        if tag == "-item" and len(p) >= 2:
            side, slot = _slot_of(p[0])
            item = to_id(p[1])
            self._items_revealed.add((slot, item))
            return ItemRevealed(side=side, slot=slot, item=item)

        if tag in ("weather", "-weather"):
            return WeatherChanged(weather=to_id(p[0]) if p else "none")
        if tag == "terrain":
            return TerrainChanged(terrain=to_id(p[0]).removesuffix("terrain") if p else "none")

        if tag == "win":
            return BattleEnd(winner_side=p[0] if p else None)
        if tag == "tie":
            return BattleEnd(winner_side=None)

        return Unknown(raw=raw)

    # ── helpers ────────────────────────────────────────────────────────
    def _reset_line_tracking(self) -> None:
        """A non-event line breaks the back-to-back duplicate runs."""
        self._prev_line_was_hp = False
        self._prev_line_was_switch = False

    def _find_slot(self, line: str, parts: List[str]) -> Optional[Tuple[str, str]]:
        """Slot for a [from]-context reveal: prefer ``[of] <slot>`` (the holder/
        source), else the primary payload when it is itself a slot."""
        m = _OF_SLOT.search(line)
        if m:
            slot = m.group(1)
            return slot[:2], slot
        if len(parts) > 2:
            sm = re.match(r"^(p[12][a-f])", parts[2].strip())
            if sm:
                return sm.group(1)[:2], sm.group(1)
        return None

    def _learn_maxhp(self, slot: str, hp_payload: str) -> None:
        """Switch-in lines are always 'max/max' → ground truth for the slot.

        The sim logs a duplicate '%/100' line right after each real one (see
        module docstring), so '100/100' can arrive after '183/183' for the same
        slot. We key on the SPECIES: the same mon's duplicate is ignored, but a
        DIFFERENT mon taking the active slot (a faint + switch-in, common in
        3v3) updates the slot's max HP. (max() would be wrong for maxhp<100:
        the duplicate reads 100/100.)
        """
        m = re.match(r"(\d+)\s*/\s*(\d+)", hp_payload.strip())
        if not (m and int(m.group(1)) == int(m.group(2))):
            return
        val = int(m.group(2))
        cur = self.species_by_slot.get(slot)
        if self._maxhp_species.get(slot) == cur:
            return  # the same mon's duplicate line — ignore
        self.maxhp_by_slot[slot] = val
        self._maxhp_species[slot] = cur

    def _hp_event(self, slot_payload: str, hp_payload: str,
                  damage: bool) -> Optional[Event]:
        side, slot = _slot_of(slot_payload)
        hp_payload = hp_payload.strip()
        ko = hp_payload.endswith("fnt")
        if ko:
            # KO line: "0 fnt" (no denominator — getHealth() returns "0 fnt" for
            # a fainted mon). Max HP is known from switch-in; per D40 the true
            # damage is >= maxhp (the log clips it to 0).
            m = re.match(r"(\d+)", hp_payload)
            hp = int(m.group(1)) if m else 0
            denom = None
        else:
            m = re.match(r"(\d+)\s*/\s*(\d+)", hp_payload)
            if not m:
                return None
            hp, denom = int(m.group(1)), int(m.group(2))
        known = self.maxhp_by_slot.get(slot)
        # drop the sim's '%/100' duplicate for a mon whose max HP is not 100
        if not ko and known is not None and denom != known:
            return None
        maxhp = known if known is not None else (0 if ko else denom)
        # drop an identical back-to-back line (the maxhp==100 dup / 2nd "0 fnt")
        sig = ("d" if damage else "h", slot, hp, maxhp)
        if self._prev_line_was_hp and self._last_hp_sig == sig:
            return None
        self.maxhp_by_slot[slot] = maxhp
        if damage:
            return DamageObserved(target_side=side, target_slot=slot, hp_after=hp,
                                  maxhp=maxhp, ko=ko, source_kind="move")
        return HealObserved(target_side=side, target_slot=slot, hp_after=hp,
                            maxhp=maxhp, source_kind="move")


def events_from_log(lines: List[str]) -> List[Event]:
    """One-shot parse of a battle log (list of ``|...`` lines)."""
    return LogAdapter().feed_lines(lines)
