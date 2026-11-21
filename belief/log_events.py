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
    FieldEffectChanged,
    HealObserved,
    HitCount,
    ItemRevealed,
    MoveFailed,
    MoveUsed,
    StatusApplied,
    StatusCured,
    SwitchIn,
    SwitchOut,
    TailwindChanged,
    TeamPreview,
    TeraUsed,
    TerrainChanged,
    Turn,
    Unknown,
    VolatileChanged,
    WeatherChanged,
)

#: tags that carry no belief evidence — silently skipped (NOT Unknown)
IGNORED_TAGS = frozenset({
    # battle meta (no set evidence)
    "hint", "-hint", "name", "player", "teamsize", "rating", "gen", "tier",
    "battle_type", "pokestats", "details", "value", "rule", "ruleset",
    # `poke` is NOT here — it maps to TeamPreview (the opponent's team comp);
    # `teampreview` (the phase marker, no payload) stays ignored.
    "team", "pokeball", "t:", "gametype", "clearpoke", "teampreview",
    "showteam", "rated", "chat", "message", "-message", "html", "debug",
    "bigerror", "detailschange",
    # human-ladder ambient noise (S4.0 gate, verified 2026-11-17 against
    # gen9ou.jsonl): real ladder logs are full of lobby/chat/broadcast lines the
    # clean self-play logs never had. j=join, c=chat, l=leave, n=nickname,
    # inactive=battle-timer notice, raw/uhtml/badge=HTML broadcasts, -singleturn=
    # cosmetic "lasts one turn" announce (the real move is the |move| line we map).
    # None carry set evidence. (In the sample battle these were ALL 484 Unknowns.)
    "j", "c", "l", "n", "inactive", "raw", "badge", "uhtml", "-singleturn",
    # visual/ambient effects (no set evidence). `-prepare` (SIM-PROTOCOL.md:594
    # |-prepare|ATTACKER|MOVE) is the client charge/aim animation line — the real
    # move is the |move| line we DO map, so it adds no belief evidence and is
    # ignored (not Unknown). Was surfacing as the 3.7 "benign prepare" Unknowns.
    "-center", "-block", "-activate", "-clearmessage", "-singlemove",
    "-mustrecharge", "-warpplate", "-clearspikes", "-prepare",
    # `-sidestart` is NOT here — it is checked first in _map (the Tailwind
    # start line is the speed-order input for 3.4; other side-condition
    # starts stay ignored).
    # `-fail` / `cant` are NOT here — they map to MoveFailed (the fork's
    # move-attempt-failure lines; see the handlers below).
    # `-start`/`-end` are NOT here — the curated per-mon volatiles (D61)
    # map to VolatileChanged; non-curated ones are ignored explicitly.
    "split", "upkeep", "-miss",
    "-transform", "-primal", "-formechange", "-enditem", "swap",
    # human-ladder extras (S4.0 gate, verified 2026-11-17 across 200 gen9ou
    # battles — 90 had >=1 Unknown, these 4 tags are the no-set-evidence ones):
    #   -anim        (44)  client animation line (battle.js case "-anim"): the
    #                      move's VFX — the real move is the |move| line we map.
    #   inactiveoff   (1)  lobby/timer notice "Battle timer is now OFF."
    #   -clearallboost (42) whole-boost reset (the per-stat -boost/-unboost is the
    #                      evidence; consistent with the already-ignored -clearboost).
    #   -swapsideconditions (13) client swapSideConditions(): swaps the two sides'
    #                      hazards/screens — a rare field-state edge, no SET reveal
    #                      (the fuzz can flag it if it ever matters for the calc).
    # NOTE: `drag`/`replace`/`-sethp` are NOT here — they carry evidence and are
    # mapped (forced SwitchIn / absolute HP set) just above.
    "-anim", "inactiveoff", "-clearallboost", "-swapsideconditions",
    # non-BSS / gen-specific (no 3.1 evidence)
    "-mega", "-zpower", "-zbroken", "-burst", "-candynamax",
    # damage MODIFIERS — the lines themselves carry no set evidence, and none of
    # them NEED to (D48 bug-hunt, verified against the 3.4 machinery): type
    # effectiveness is already inside the calc's window (computed from the
    # candidate's OWN typing), a crit is inside the crit window the likelihood
    # also accepts (damage_likelihood), and an OHKO is the calc's guaranteed-KO
    # formula (consumed via the KO line, D40). Ignoring them keeps Unknown the
    # genuinely-unexpected signal the 3.7 fuzz reports.
    "-immune", "-resisted", "-supereffective", "-crit", "-ohko",
    # whole-boost resets (the per-stat -boost/-unboost is the evidence)
    "-clearboost", "-clearpositiveboost", "-clearnegativeboost",
})
# NOTE: ``-weather`` is NOT ignored — it is the ability-revealing weather-start
# line (``|-weather|RainDance|[from] ability: Drizzle|[of] p1a``) and maps to
# WeatherChanged (plus an AbilityRevealed via the [from] capture below).

#: The seven CURATED per-mon volatiles (D61: "add the 7 curated volatiles to
#: Facts"). Showdown volatile ids as they appear in |-start|/|-end| payloads
#: (toID-normalized). Taunt (4-turn move lock), Encore (move repeat lock),
#: Disguise (Imposter one-hit shield), Trapped (Arena Trap / Shadow Tag),
#: Curse (the Curse-move Atk/Def/Spe volatile), Destiny Bond (retaliation KO),
#: Gastro Acid (move-lock, 5 turns). Stockpile + Protect are already tracked
#: separately (Facts.stockpile_layers / Facts.used_protect).
_CURATED_VOLATILES = frozenset({
    "disguise", "taunt", "encore", "trapped",
    "curse", "destinybond", "gastroacid",
})


#: [from] <kind>: <Name> — the fork's context convention (conditions.js/battle.js):
#:   [from] ability: Drizzle   (ability holder = the [of] slot, NOT the target)
#:   [from] item: Choice Band
_FROM_KIND = re.compile(r"\[from\]\s*(ability|item):\s*([^|\]]+)")
#: [of] p2a: Charizard — the source/holder slot for a [from]-context reveal
_OF_SLOT = re.compile(r"\[of\]\s*(p[12][a-f])")
#:
#: A -damage/-heal 3rd payload that marks the line as a CONTEXT (residual) event
#: rather than a pure move hit. Fork-verified (2026-11-16, scratch/probe_residual.js,
#: all 10 cases: move hit = 2 payloads; burn/poison/toxic/recoil/confusion/powder/
#: weather/drain/leechseed all bracketed) AND proven statically from the fork:
#:
#:   * spreadDamage log switch  sim/battle.ts:2159-2178 — default branch logs NO
#:     3rd payload ONLY when `effect.effectType === 'Move'` (a real move hit) OR
#:     `!name`; every other (non-move) effect logs a bracketed `[from] <name>` /
#:     `[of] <source>` (or the special `[silent]`/`[partiallytrapped]` cases).
#:   * heal log switch          sim/battle.ts:2293-2314 — drain → `[from] drain`,
#:     leechseed/rest → `[silent]`, zpower → `[zeffect]`, a healing Move → no
#:     bracket (a heal move, e.g. Synthesis), every other → `[from] <name>`.
#:   * `!name` edge is UNREACHABLE for real damage: `fullname = data.fullname ||
#:     name` (sim/dex-data.ts:125), and every real damage-source effect carries a
#:     name — brn/psn/tox (Conditions), recoil→'Recoil'/drain→'Drain'
#:     (sim/dex-conditions.ts:700-707), weather (Sandstorm/Hail/Snowscape), the
#:     generic getByID `else` sets `name: id`. The only name-less effect
#:     (EMPTY_CONDITION, id==='') carries no damage. So a non-move damage line
#:     ALWAYS carries a bracketed 3rd payload.
#:
#: Consequence (D48): the presence of a bracketed 3rd payload is the clean
#: move-vs-residual signal; a plain move hit never carries one, so this cannot
#: false-positive on real move damage.
_CONTEXT_TAG = re.compile(r"^\[(from|silent|of|zeffect)\]")


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
        # per-side |poke| count -> preview-order slot (a..f). The side shape
        # ``|poke|p2|Species`` names no slot, so we assign by list order (the
        # fork's format); a slot shape ``|poke| p2a 1 |Species|`` names its own.
        self._poke_order: Dict[str, int] = {}
        self._items_revealed: set = set()
        self._abilities_revealed: set = set()
        # hp-line dedupe (the sim logs each hp change twice, back-to-back):
        self._prev_line_was_hp = False       # was the previous fed line an hp event?
        self._last_hp_sig = None             # ('d'|'h', slot, hp_after, maxhp)
        # switch-in dedupe (the sim logs each switch-in twice, back-to-back):
        self._prev_line_was_switch = False   # was the previous fed line a switch-in?
        self._last_switch_sig = None         # (slot, species) of that switch-in
        # username -> side, from |player|/|name| identity lines: human-ladder
        # logs name the winner by USERNAME in |win| (verified 2026-11-17, all
        # 2000 gen9ou.jsonl battles), so we resolve it to 'p1'/'p2' there.
        self.side_by_user: Dict[str, str] = {}
        # move-source attribution for damage lines (3.4): the most recent
        # `|move|` line as (side, slot, move, target_slot). Damage lines follow
        # their move's announce line (possibly with status/boost lines in
        # between); the target check in _attribute_move_source keeps
        # mis-attribution out.
        self._last_move: Optional[Tuple[str, str, str, Optional[str]]] = None
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
            return BattleStart(fmt=p[0].strip() if p else None)  # "|start| gen9ou" is space-padded
        if tag in ("player", "name") and len(p) >= 2:
            # |player| p1 | Username | avatar | rating — ladder identity. NO
            # belief evidence (still None), but record username -> side so the
            # `|win| <username>` branch resolves the winner to a side.
            side, user = p[0].strip(), p[1].strip()
            if side in ("p1", "p2") and user:
                self.side_by_user[user] = side
                self.side_by_user[to_id(user)] = side
            return None
        if tag in IGNORED_TAGS:
            return None
        if tag == "turn":
            return Turn(n=_int(p[0]))

        if tag in ("switch", "drag", "replace"):
            # `drag` (battle-actions.js:185 `isDrag ? "drag" : "switch"`) and
            # `replace` (client `poke.side.replace`) are the FORCED switch-in
            # variants — a mon is dragged/replaced into the slot by the game, NOT
            # a player decision. Same payload as `switch` (slot, details, hp/max).
            # Marked `forced` so the corpus counts the belief update but NOT a
            # switch decision (keeps the human switch-rate accurate).
            forced = tag != "switch"
            side, slot = _slot_of(p[0])
            species = to_id(p[1].split(",")[0])
            # the sim logs each switch-in twice, back-to-back — drop the dup
            if self._prev_line_was_switch and self._last_switch_sig == (slot, species):
                return None
            self.species_by_slot[slot] = species
            hp_payload = p[2] if len(p) >= 3 else ""
            m = re.match(r"(\d+)\s*/\s*(\d+)", hp_payload.strip())
            cur_hp = int(m.group(1)) if m else 0
            max_hp = int(m.group(2)) if m else 0
            if not forced:
                self._learn_maxhp(slot, hp_payload)  # a real lead/switch is max/max ground truth
            return SwitchIn(side=side, slot=slot, species=species,
                            hp=cur_hp, maxhp=max_hp, forced=forced)
        if tag == "leave":
            side, slot = _slot_of(p[0])
            return SwitchOut(side=side, slot=slot)
        if tag == "poke" and len(p) >= 2:
            # Team preview (|poke|): reveal one team mon. The side shape is
            # ``|poke|p2|Species[, details]`` (the fork's format; battle.js
            # add("poke", side, details)); a slot shape ``|poke| p2a 1 |Species|``
            # also occurs in some logs. The slot is the PREVIEW order (a..f), NOT
            # the field slot — the player picks the lead during the preview
            # phase, so the active species is not necessarily the first previewed.
            # The belief maps the active to its field slot and the rest to bench
            # slots in get_context() (state.py). p1 previews are inert (the own
            # team is fully known from the own_team spec).
            name = p[1].split(",")[0].strip()
            sp = to_id(name)
            if not sp:
                return None
            key = p[0].strip().split()[0]
            side = key[:2]
            if side not in ("p1", "p2"):
                return None
            if len(key) == 3 and key[2] in "abcdef":
                slot = key  # explicit slot shape
            else:
                n = self._poke_order.get(side, 0)
                if n >= 6:
                    return None
                self._poke_order[side] = n + 1
                slot = f"{side}{'abcdef'[n]}"
            return TeamPreview(side=side, slot=slot, species=sp)
        if tag == "faint":
            side, slot = _slot_of(p[0])
            return Faint(side=side, slot=slot)

        if tag == "-ability" and len(p) >= 2:
            side, slot = _slot_of(p[0])
            return AbilityRevealed(side=side, slot=slot, ability=to_id(p[1]))

        if tag in ("-start", "-end") and len(p) >= 2:
            # Per-mon volatile start/end (|-start|p2a|Taunt|, |-end|p2a|Taunt|).
            # Only the D61 curated set is surfaced (VolatileChanged); every other
            # volatile (FocusEnergy, Intimidate-boost echoes, …) is KNOWN ambient
            # state with no set evidence -> silent None (NOT Unknown).
            side, slot = _slot_of(p[0])
            vid = to_id(p[1])
            if vid not in _CURATED_VOLATILES:
                return None
            return VolatileChanged(side=side, slot=slot, volatile=vid,
                                  active=(tag == "-start"))

        if tag == "-sidestart" and len(p) >= 2:
            # Side-condition start (Reflect, Light Screen, Aurora Veil, Spikes,
            # Stealth Rock, Toxic Spikes, Sticky Web, Tailwind, Safeguard, …).
            # Tailwind is special-cased (the 3.4 speed-order evidence); all
            # others emit the generic FieldEffectChanged for the AI's context.
            if "move: Tailwind" in p[1]:
                side, _ = _slot_of(p[0])
                return TailwindChanged(side=side, active=True)
            side, _ = _slot_of(p[0])
            name = to_id(p[1].replace("move: ", "").split(",")[0])
            return FieldEffectChanged(name=name, side=side, active=True)
        if tag == "-sideend" and len(p) >= 2:
            side, _ = _slot_of(p[0])
            name = to_id(p[1].replace("move: ", "").split(",")[0])
            return FieldEffectChanged(name=name, side=side, active=False)
        if tag in ("-fieldstart", "-fieldactivate") and len(p) >= 1:
            # Pseudo-weather: Trick Room, Wonder Room, Gravity, Magic Room, …
            name = to_id(p[0].replace("move: ", "").split(",")[0])
            return FieldEffectChanged(name=name, side="", active=True)
        if tag == "-fieldend" and len(p) >= 1:
            name = to_id(p[0].replace("move: ", "").split(",")[0])
            return FieldEffectChanged(name=name, side="", active=False)
        if tag == "move" and len(p) >= 2:
            # |move|source|MoveName|target — battle-actions.js logs
            # addMove("move", pokemon, movename, target): move is p[1].
            side, slot = _slot_of(p[0])
            move = to_id(p[1])
            if len(p) >= 3:
                t_side, t_slot = _slot_of(p[2])
            else:
                t_side = t_slot = None
            self._last_move = (side, slot, move, t_slot)
            return MoveUsed(side=side, slot=slot, move=move,
                            target_side=t_side, target_slot=t_slot)
        if tag == "-fail" and len(p) >= 1:
            # this fork's "the announced move attempt FAILED" line
            # (battle-actions.ts:656-659): the move is announced (|move|) and
            # then fails (miss, immunity, Protect, blocked) -> `|-fail|<user>`
            # with NO move name in the payload. The fork sets
            # moveThisTurnResult=false on this path, so the NEXT turn's Stomping
            # Tantrum / Temper Flare doubles (moves.ts:18050/19186). The move
            # name comes from the user's own announce line — the fail line
            # always immediately follows it (a move executes atomically).
            side, slot = _slot_of(p[0])
            lm = self._last_move
            move = lm[2] if (lm is not None and lm[1] == slot) else ""
            return MoveFailed(side=side, slot=slot, move=move)
        if tag == "cant" and len(p) >= 2:
            # status/PP move-attempt ABORTs — logged BEFORE any |move| line (the
            # BeforeMove / nopp paths never announce): par (conditions.ts:42),
            # slp (:76), frz (:103), flinch (:203), recharge (:369), nopp
            # (battle-actions.ts:340 — the move name IS in the line). The fork
            # sets moveThisTurnResult=false on ALL of these (runMove line 319 /
            # 331 / 342) -> the next turn's ST / Temper Flare doubles. Other cant
            # reasons (trp, ability blocks, ...) are NOT move-attempt failures
            # and stay ignored (return None below).
            reason = p[1]
            if reason not in ("par", "slp", "frz", "flinch", "recharge", "nopp"):
                return None
            side, slot = _slot_of(p[0])
            move = to_id(p[2]) if (reason == "nopp" and len(p) >= 3) else ""
            return MoveFailed(side=side, slot=slot, move=move)
        if tag == "movefail" and len(p) >= 2:
            # Upstream log format |movefail|slot|MoveName. NOTE: this fork never
            # emits `movefail` (verified in dist/sim + client parser) — this
            # fork's failure lines are `|-fail|<user>` (after the |move| announce)
            # and `|cant|<user>|<reason>` (before it); both map to MoveFailed
            # above. Kept for protocol completeness.
            side, slot = _slot_of(p[0])
            return MoveFailed(side=side, slot=slot, move=to_id(p[1]))

        if tag == "-hitcount" and len(p) >= 2:
            side, slot = _slot_of(p[0])
            return HitCount(side=side, slot=slot, n=_int(p[1]))

        if tag == "-damage" and len(p) >= 2:
            # p[2] is the residual context tag (`[from] brn` / `[from] Recoil` /
            # `[silent]` …) — present on residual lines, absent on a pure move hit.
            return self._hp_event(p[0], p[1], damage=True, context=p[2] if len(p) >= 3 else None)
        if tag in ("-heal", "-sethp") and len(p) >= 2:
            # `-sethp` (pokemon.js sethp / battle.js Pain Split) is an ABSOLUTE
            # HP set ("79/100"), not a delta — the belief just records the new HP
            # (_set_hp), so the heal path is the right fit (no reflector last_hit,
            # no move-source attribution — it is not a move hit).
            return self._hp_event(p[0], p[1], damage=False, context=p[2] if len(p) >= 3 else None)
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
            # payload is the winning SIDE ("p1"/"p2") in sim logs, but the
            # winner's USERNAME in human-ladder logs ("|win| LT111VZ Roman") —
            # resolve via the |player|/|name| mapping. Unresolvable -> keep the
            # raw value (observable in the corpus, NOT silently None).
            raw = p[0].strip() if p else None
            if raw in ("p1", "p2"):
                return BattleEnd(winner_side=raw)
            return BattleEnd(winner_side=(self.side_by_user.get(raw)
                                          or self.side_by_user.get(to_id(raw))
                                          or raw))
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
                  damage: bool, context: Optional[str] = None) -> Optional[Event]:
        side, slot = _slot_of(slot_payload)
        hp_payload = hp_payload.strip()
        residual = self._is_residual_context(context)  # burn/poison/recoil/… not a move
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
            ev = DamageObserved(target_side=side, target_slot=slot, hp_after=hp,
                                maxhp=maxhp, ko=ko,
                                source_kind="residual" if residual else "move")
            if not residual:
                self._attribute_move_source(ev)
            return ev
        return HealObserved(target_side=side, target_slot=slot, hp_after=hp,
                            maxhp=maxhp,
                            source_kind="residual" if residual else "move")

    def _is_residual_context(self, context: Optional[str]) -> bool:
        """A -damage/-heal line whose 3rd payload is a bracketed context tag is a
        RESIDUAL / ability / item / recoil / drain event, NOT a pure move hit. The
        fork (`runDamage`/`heal`/`directDamage`) logs a move hit with no 3rd payload
        and every residual line with one — so a bracketed 3rd payload is the signal."""
        return bool(context) and bool(_CONTEXT_TAG.match(context.strip()))

    def _attribute_move_source(self, ev: DamageObserved) -> None:
        """Attribute move-source damage to the most recent `|move|` line, when
        that line plausibly targeted this slot (or was self-damage/recoil from
        this slot). Residual damage (burn/poison) carries no `|move|` line in
        this fork and is left source-less — the belief still records the HP
        fact and defers that evidence to 3.6."""
        lm = self._last_move
        if lm is None:
            return
        side, slot, move, target = lm
        if target != ev.target_slot and slot != ev.target_slot:
            return
        ev.source_side, ev.source_slot, ev.move = side, slot, move


def events_from_log(lines: List[str]) -> List[Event]:
    """One-shot parse of a battle log (list of ``|...`` lines)."""
    return LogAdapter().feed_lines(lines)
