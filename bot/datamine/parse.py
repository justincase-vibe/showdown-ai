"""Parse Showdown battle logs into structured per-turn decision data.

Takes the JSONL files produced by ``fetch.py`` and turns each raw ``log``
(the ``|``-protocol) into a structured battle object:

  * metadata (format, players, ratings, winner, gen)
  * team composition per player (species in slot order)
  * per-turn event stream (moves, switches, damage, status, faints, hazards)
  * **action mapping**: each player decision → global move-identity action id (0..962)

The action is a move *by identity* (name), not a slot: plain/Z/G-Max moves map to their
move id, switches to the slot id, and mega/dynamax/tera to the gimmick id. This is the
same 963-space the policy and live env use (``action_space.py``), so (state, action)
pairs are self-consistent for behavioral cloning and warm-start PPO.

Usage:
    python bot/datamine/parse.py --input bot/data/replays/gen8ru.jsonl --max 50
    python bot/datamine/parse.py --input bot/data/replays/gen8ru.jsonl --report
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TeamMember:
    slot: int
    species: str
    gender: str = "?"
    shiny: bool = False

@dataclass
class MoveEvent:
    """A single move usage (or Z-move / G-Max move)."""
    actor: str            # "p1" or "p2"
    actor_short: str      # e.g. "p1a: Exploud"
    move_name: str        # as shown in log (Z-moves show the Z-name)
    target: Optional[str]  # e.g. "p2a: Vanilluxe" or None (self/status)
    turn: int
    move_idx: int = -1    # revealed-order index (0-3), set by parser
    gimmick: str = ""     # "", "mega", "z", "dynamax", "tera"

@dataclass
class SwitchEvent:
    actor: str
    actor_short: str
    species: str
    hp_pct: int           # 0-100
    turn: int
    forced: bool = False  # from Volt Switch / U-turn / drag
    slot: int = -1        # team slot index, set by parser

@dataclass
class OutcomeEvent:
    """Damage / heal / status / boost / faint that follows a move."""
    target: str           # "p1a: ..."
    kind: str             # "damage", "heal", "status", "boost", "unboost", "faint",
                          # "super", "resist", "immune", "miss", "crit", "fail"
    detail: str = ""      # e.g. status code, stat name, HP value
    hp_after: Optional[int] = None
    turn: int = 0

@dataclass
class Decision:
    """A single player decision point → unified action index."""
    turn: int
    actor: str            # "p1" or "p2"
    action_idx: int       # 0-25 (unified)
    action_desc: str      # human-readable
    raw: str              # "move:Scald" or "switch:slot3"

@dataclass
class ParsedBattle:
    # metadata
    battle_id: str
    formatid: str
    format: str
    gen: int
    players: List[str] = field(default_factory=list)
    ratings: List[float] = field(default_factory=list)
    winner: Optional[str] = None       # player name
    winner_p: Optional[str] = None     # "p1" or "p2"
    total_turns: int = 0
    # teams
    team_p1: List[TeamMember] = field(default_factory=list)
    team_p2: List[TeamMember] = field(default_factory=list)
    # events (chronological)
    moves: List[MoveEvent] = field(default_factory=list)
    switches: List[SwitchEvent] = field(default_factory=list)
    outcomes: List[OutcomeEvent] = field(default_factory=list)
    # decisions (for imitation)
    decisions: List[Decision] = field(default_factory=list)
    # raw log (kept for debugging)
    raw_log: str = ""

# ---------------------------------------------------------------------------
# Gimmick move maps — imported from reveal.py (single source of truth, shared with
# the live masked view and the replay reconstruction). Previously this module kept
# its own copies (with duplicated/contradictory entries), which is exactly the kind
# of drift that causes a train/live mismatch.
# ---------------------------------------------------------------------------
_BOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # bot/
if _BOT_DIR not in sys.path:
    sys.path.insert(0, _BOT_DIR)
from reveal import ZMOVE_TO_BASE, GMAX_MOVES  # noqa: E402
from action_space import get_action_space  # noqa: E402

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class BattleLogParser:
    """Parses a single Showdown battle log into a ParsedBattle."""

    def __init__(self, battle_id: str = "", formatid: str = "", format: str = "",
                 rating: float = 0.0, players: Optional[List[str]] = None):
        self.b = ParsedBattle(
            battle_id=battle_id,
            formatid=formatid,
            format=format,
            gen=0,
        )
        self._players = players or []
        self._rating = rating
        # Track: nickname/short name → slot index (per player)
        self._name_to_slot: Dict[str, Dict[str, int]] = {"p1": {}, "p2": {}}
        # Track: nickname → species (per player), for move tracking
        self._name_to_species: Dict[str, Dict[str, str]] = {"p1": {}, "p2": {}}
        # Track: (player, species) → list of move names in reveal order
        self._move_reveal: Dict[str, List[str]] = {}
        # Track current turn
        self._turn = 0
        # Track active Pokémon per side (for context)
        self._active: Dict[str, str] = {}  # "p1" -> short name
        # Gimmick flags for current turn
        self._turn_mega: Dict[str, bool] = {}
        self._turn_dynamax: Dict[str, bool] = {}
        self._turn_tera: Dict[str, bool] = {}
        # Last move decision per role this turn. The gimmick log line (|-mega| /
        # |-terastallize| / |-formechange|) arrives AFTER the |move| line it belongs
        # to, so we tag the action retroactively (see _apply_gimmick).
        self._last_move: Dict[str, Optional[dict]] = {"p1": None, "p2": None}
        # Ratings from log
        self._ratings: Dict[str, float] = {}
        # Global move-identity action space (single source of truth for action ids).
        self._as = get_action_space()

    def parse(self, log: str) -> ParsedBattle:
        self.b.raw_log = log
        lines = log.splitlines()
        for line in lines:
            if not line.startswith("|"):
                continue
            self._process_line(line)
        self._finalize()
        return self.b

    def _process_line(self, line: str) -> None:
        # Split on | but be careful: fields can contain | in theory, but in practice
        # the protocol uses | as separator and fields don't contain |.
        parts = line.split("|")
        # parts[0] is "" (before first |), parts[1] is the tag
        if len(parts) < 2:
            return
        tag = parts[1].strip()
        fields = parts[2:]  # remaining fields

        if tag == "":
            return
        elif tag == "player":
            self._handle_player(fields)
        elif tag == "poke":
            self._handle_poke(fields)
        elif tag == "gen":
            self.b.gen = int(fields[0]) if fields else 0
        elif tag == "tier":
            pass  # already have format
        elif tag == "start":
            pass
        elif tag == "turn":
            self._turn = int(fields[0]) if fields else self._turn
            self._turn_mega = {}
            self._turn_dynamax = {}
            self._turn_tera = {}
            self._last_move = {"p1": None, "p2": None}
        elif tag == "switch":
            self._handle_switch(fields)
        elif tag == "move":
            self._handle_move(fields)
        elif tag == "-damage":
            self._handle_outcome(fields, "damage")
        elif tag == "-heal":
            self._handle_outcome(fields, "heal")
        elif tag == "-status":
            self._handle_outcome(fields, "status")
        elif tag == "-boost":
            self._handle_outcome(fields, "boost")
        elif tag == "-unboost":
            self._handle_outcome(fields, "unboost")
        elif tag == "faint":
            self._handle_outcome(fields, "faint")
        elif tag == "-supereffective":
            self._handle_outcome(fields, "super")
        elif tag == "-resisted":
            self._handle_outcome(fields, "resist")
        elif tag == "-immune":
            self._handle_outcome(fields, "immune")
        elif tag == "-miss":
            self._handle_outcome(fields, "miss")
        elif tag == "-crit":
            self._handle_outcome(fields, "crit")
        elif tag == "-fail":
            self._handle_outcome(fields, "fail")
        elif tag == "-mega":
            self._handle_mega(fields)
        elif tag == "-formechange":
            self._handle_formechange(fields)
        elif tag == "-terastallize":
            self._handle_tera(fields)
        elif tag == "-start":
            self._handle_start(fields)
        elif tag == "-sidestart":
            self._handle_outcome(fields, "hazard")
        elif tag == "-sideend":
            self._handle_outcome(fields, "hazard_end")
        elif tag == "-weather":
            self._handle_outcome(fields, "weather")
        elif tag == "-fieldstart":
            self._handle_outcome(fields, "field")
        elif tag == "win":
            self._handle_win(fields)
        elif tag == "raw":
            self._handle_raw(fields)
        # Other tags: ignore (c, j, t:, upkeep, inactive, rule, etc.)

    # ------------------------------------------------------------------ handlers

    def _handle_player(self, fields: List[str]) -> None:
        if len(fields) >= 3:
            role = fields[0]  # "p1" or "p2"
            name = fields[1]
            if len(fields) >= 4 and fields[3].isdigit():
                self._ratings[role] = float(fields[3])
            if name not in self.b.players:
                self.b.players.append(name)

    def _handle_poke(self, fields: List[str]) -> None:
        if len(fields) >= 2:
            role = fields[0]  # "p1" or "p2"
            detail = fields[1]  # "Exploud, M" or "Rotom-Mow"
            # Parse species and gender
            parts = detail.split(",")
            species = parts[0].strip()
            gender = parts[1].strip() if len(parts) > 1 else "?"
            shiny = "shiny" in detail.lower()

            slot = len(self.b.team_p1 if role == "p1" else self.b.team_p2)
            member = TeamMember(slot=slot, species=species, gender=gender, shiny=shiny)
            if role == "p1":
                self.b.team_p1.append(member)
            else:
                self.b.team_p2.append(member)

            # Map short name to slot (and species to species for consistency)
            short = self._short_name(species)
            self._name_to_slot[role][short] = slot
            self._name_to_slot[role][species] = slot
            self._name_to_species[role][short] = species
            self._name_to_species[role][species] = species

    def _handle_switch(self, fields: List[str]) -> None:
        if len(fields) < 3:
            return
        actor_short = fields[0]  # "p1a: Exploud"
        detail = fields[1]       # "Exploud, M" or "Rotom-Mow"
        hp_str = fields[2]       # "100/100" or "0 fnt"
        forced = len(fields) > 3 and any("[from]move" in f.lower() for f in fields[3:])
        # Also check for drag
        if not forced:
            # Check next lines for drag (can't do this here; approximate with [from])
            pass

        role = "p1" if actor_short.startswith("p1") else "p2"
        short_name = actor_short.split(": ")[-1] if ": " in actor_short else actor_short

        # Determine HP%
        hp_pct = self._parse_hp(hp_str)

        # Parse species from the detail field (2nd field) — this is the actual
        # species name even if the player nicknamed the Pokémon.
        parts = detail.split(",")
        species = parts[0].strip()

        # Determine slot: try the nickname first, then the species name
        slot = self._name_to_slot[role].get(short_name, -1)
        if slot == -1:
            slot = self._name_to_slot[role].get(species, -1)
        if slot == -1:
            species_short = self._short_name(species)
            slot = self._name_to_slot[role].get(species_short, -1)
        if slot == -1:
            # Fallback: search team list by species
            team = self.b.team_p1 if role == "p1" else self.b.team_p2
            for m in team:
                if m.species == species:
                    slot = m.slot
                    break

        # Register mappings
        if slot == -1:
            slot = len(self.b.team_p1 if role == "p1" else self.b.team_p2)
            member = TeamMember(slot=slot, species=species)
            if role == "p1":
                self.b.team_p1.append(member)
            else:
                self.b.team_p2.append(member)
        self._name_to_slot[role][short_name] = slot
        self._name_to_slot[role][species] = slot
        self._name_to_species[role][short_name] = species
        self._name_to_species[role][species] = species

        self.b.switches.append(SwitchEvent(
            actor=role,
            actor_short=actor_short,
            species=species,
            hp_pct=hp_pct,
            turn=self._turn,
            forced=forced,
            slot=slot,
        ))
        self._active[role] = short_name

        # Create decision
        self.b.decisions.append(Decision(
            turn=self._turn,
            actor=role,
            action_idx=(self._as.switch_id(slot) if slot is not None else 0),  # switch to slot
            action_desc=f"switch → slot {slot} ({species})",
            raw=f"switch:{slot}",
        ))

    def _handle_move(self, fields: List[str]) -> None:
        if len(fields) < 2:
            return
        actor_short = fields[0]  # "p1a: Exploud"
        move_name = fields[1]
        target = fields[2] if len(fields) > 2 else None

        role = "p1" if actor_short.startswith("p1") else "p2"
        short_name = actor_short.split(": ")[-1] if ": " in actor_short else actor_short

        # Key the reveal list by IDENT (nickname), not species. Two mons of the same
        # species must get independent reveal lists -- keying by species collided for
        # duplicate-species teams and mislabeled both. Showdown disambiguates duplicates
        # as "Garchomp 2", so the ident is unique per mon (matches state_from_log's
        # per-slot tracking, keeping reveal-index == action-slot on both sides).
        key = f"{role}:{short_name}"
        if key not in self._move_reveal:
            self._move_reveal[key] = []

        # Check if this is a Z-move (gen 7 only)
        gimmick = ""
        base_move = move_name
        if self.b.gen == 7 and move_name in ZMOVE_TO_BASE:
            gimmick = "z"
            base_move = ZMOVE_TO_BASE[move_name]
        elif self.b.gen == 8 and move_name in GMAX_MOVES:
            gimmick = "dynamax"
            base_move = move_name

        # Check turn-level gimmicks
        if self._turn_mega.get(role):
            gimmick = "mega"
        if self._turn_dynamax.get(role):
            gimmick = "dynamax"
        if self._turn_tera.get(role):
            gimmick = "tera"

        # Assign move index (cap at 3 — action space has only 4 move slots)
        if base_move not in self._move_reveal[key]:
            self._move_reveal[key].append(base_move)
        move_idx = min(self._move_reveal[key].index(base_move), 3)

        me = MoveEvent(
            actor=role,
            actor_short=actor_short,
            move_name=move_name,
            target=target,
            turn=self._turn,
            move_idx=move_idx,
            gimmick=gimmick,
        )
        self.b.moves.append(me)

        # Create decision → unified action index (gimmick may be patched retroactively
        # when the |-mega| / |-terastallize| line arrives on this same turn).
        decision = Decision(
            turn=self._turn,
            actor=role,
            action_idx=self._move_to_action(move_name, gimmick, self.b.gen),
            action_desc=f"move:{move_name} (slot {move_idx}, gimmick={gimmick or 'none'})",
            raw=f"move:{move_name}",
        )
        self.b.decisions.append(decision)

        # Remember so a later gimmick line this turn can re-label this action.
        self._last_move[role] = {
            "decision": decision,
            "move_event": me,
            "move_idx": move_idx,
            "move_name": move_name,
            "gimmick": gimmick,
            "turn": self._turn,
        }

    def _handle_outcome(self, fields: List[str], kind: str) -> None:
        if not fields:
            return
        target = fields[0]
        detail = fields[1] if len(fields) > 1 else ""
        hp_after = None
        if kind in ("damage", "heal") and "/" in detail:
            try:
                hp_after = int(detail.split("/")[0])
            except ValueError:
                pass
        self.b.outcomes.append(OutcomeEvent(
            target=target,
            kind=kind,
            detail=detail,
            hp_after=hp_after,
            turn=self._turn,
        ))

    def _handle_mega(self, fields: List[str]) -> None:
        if fields:
            role = "p1" if fields[0].startswith("p1") else "p2"
            self._turn_mega[role] = True
            self._apply_gimmick(role, "mega")

    def _handle_formechange(self, fields: List[str]) -> None:
        """Detect Mega Evolution via formechange (gen 6 style)."""
        if len(fields) >= 2:
            new_form = fields[1]  # e.g. "Charizard-Mega-Y", "Venusaur-Mega"
            if "Mega" in new_form:
                role = "p1" if fields[0].startswith("p1") else "p2"
                self._turn_mega[role] = True
                self._apply_gimmick(role, "mega")

    def _handle_tera(self, fields: List[str]) -> None:
        if fields:
            role = "p1" if fields[0].startswith("p1") else "p2"
            self._turn_tera[role] = True
            self._apply_gimmick(role, "tera")

    def _handle_start(self, fields: List[str]) -> None:
        # Check for Dynamax
        if len(fields) >= 2:
            effect = fields[1] if len(fields) > 1 else ""
            if "Dynamax" in effect:
                role = "p1" if fields[0].startswith("p1") else "p2"
                self._turn_dynamax[role] = True
                self._apply_gimmick(role, "dynamax")

    def _apply_gimmick(self, role: str, gimmick: str) -> None:
        """Retroactively tag this role's most recent move (this turn) with a gimmick.

        The Showdown protocol emits the gimmick line *after* the ``|move|`` it belongs
        to, so the action was initially labelled as a plain move. Re-compute the
        unified action index now that we know the gimmick.
        """
        ref = self._last_move.get(role)
        if ref is None or ref["turn"] != self._turn:
            return
        ref["gimmick"] = gimmick
        ref["move_event"].gimmick = gimmick
        ref["decision"].action_idx = self._move_to_action(ref["move_name"], gimmick, self.b.gen)
        ref["decision"].action_desc = (
            f"move:{ref['move_event'].move_name} "
            f"(slot {ref['move_idx']}, gimmick={gimmick})"
        )

    def _handle_win(self, fields: List[str]) -> None:
        if fields:
            winner_name = fields[0]
            self.b.winner = winner_name
            # Determine which player
            for i, name in enumerate(self.b.players):
                if name == winner_name:
                    self.b.winner_p = "p1" if i == 0 else "p2"
                    break

    def _handle_raw(self, fields: List[str]) -> None:
        # Rating changes: "RULT3ro good vibes's rating: 1638 → 1659 (+21 for winning)"
        # Try to extract ratings
        for f in fields:
            m = re.search(r"(\d{3,4})\s*[→>]\s*(\d{3,4})", f)
            if m:
                pass  # Could extract but not critical

    # ------------------------------------------------------------------ helpers

    def _short_name(self, species: str) -> str:
        """Convert full species name to the short form used in logs.
        e.g. 'Slowbro-Galar' → 'Slowbro', 'Rotom-Mow' → 'Rotom'
        """
        # The log uses the base name without form suffix
        # e.g. "Slowbro-Galar" → "Slowbro", "Rotom-Mow" → "Rotom"
        if "-" in species:
            return species.split("-")[0]
        return species

    def _parse_hp(self, hp_str: str) -> int:
        """Parse '100/100' or '0 fnt' → 0-100 percentage."""
        if "fnt" in hp_str:
            return 0
        if "/" in hp_str:
            try:
                cur, max_hp = hp_str.split("/")
                return int(cur)  # HP is shown as percentage in these logs
            except ValueError:
                return -1
        return -1

    def _move_to_action(self, move_name: str, gimmick: str, gen: int) -> int:
        """Map (move name, gimmick) → global move-identity action id (0..N_ACTIONS-1).

        The action is the move by *identity* (name), not a slot -- this removes the
        train/live mismatch where the same slot index meant a different move in reveal
        order vs. moveset order. Gimmicks follow the agreed layout: Z-moves and G-Max
        moves are distinct moves in the MOVE band; mega / tera (and a non-G-Max dynamax)
        are mon-level TRANSFORM actions (the accompanying move is a step-time detail in
        env.py). Falls back to the move's identity id.
        """
        as_ = self._as
        if gimmick == "mega":
            return as_.transform("mega")
        if gimmick == "tera":
            return as_.transform("tera")
        if gimmick == "dynamax" and as_.move_gimmick(move_name) != "dynamax":
            return as_.transform("dynamax")
        # plain move, Z-move, or G-Max move -> the move's identity id.
        mid = as_.move_id(move_name)
        return mid if mid is not None else 0

    def _finalize(self) -> None:
        self.b.total_turns = self._turn
        # Fill in ratings
        self.b.ratings = list(self._ratings.values())

# ---------------------------------------------------------------------------
# Batch parsing + quality report
# ---------------------------------------------------------------------------

def parse_jsonl(path: str, max_battles: Optional[int] = None) -> List[ParsedBattle]:
    """Parse all battles in a JSONL file."""
    battles = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_battles and i >= max_battles:
                break
            rec = json.loads(line)
            parser = BattleLogParser(
                battle_id=rec.get("id", f"row_{i}"),
                formatid=rec.get("formatid", ""),
                format=rec.get("format", ""),
                rating=rec.get("rating", 0.0),
                players=rec.get("players"),
            )
            battle = parser.parse(rec["log"])
            battles.append(battle)
    return battles


def quality_report(battles: List[ParsedBattle], top_n: int = 10) -> str:
    """Generate a quality report string."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"QUALITY REPORT: {len(battles)} battles parsed")
    lines.append("=" * 70)

    # Format distribution
    fmts = {}
    for b in battles:
        fmts[b.formatid] = fmts.get(b.formatid, 0) + 1
    lines.append(f"\nFormats: {fmts}")

    # ELO stats
    ratings = [b.ratings[0] for b in battles if b.ratings]
    if ratings:
        ratings.sort()
        n = len(ratings)
        lines.append(f"\nELO: min={ratings[0]:.0f}  med={ratings[n//2]:.0f}  max={ratings[-1]:.0f}")

    # Turn stats
    turns = [b.total_turns for b in battles]
    if turns:
        turns.sort()
        n = len(turns)
        lines.append(f"Turns: min={turns[0]}  med={turns[n//2]}  max={turns[-1]}  avg={sum(turns)/n:.1f}")

    # Win distribution
    p1_wins = sum(1 for b in battles if b.winner_p == "p1")
    p2_wins = sum(1 for b in battles if b.winner_p == "p2")
    unknown = len(battles) - p1_wins - p2_wins
    lines.append(f"\nWins: p1={p1_wins}  p2={p2_wins}  unknown={unknown}")

    # Action type distribution
    action_types = {"switch": 0, "move": 0, "mega": 0, "z": 0, "dynamax": 0, "tera": 0}
    for b in battles:
        for d in b.decisions:
            if d.raw.startswith("switch"):
                action_types["switch"] += 1
            elif d.raw.startswith("move"):
                if "gimmick=mega" in d.action_desc:
                    action_types["mega"] += 1
                elif "gimmick=z" in d.action_desc:
                    action_types["z"] += 1
                elif "gimmick=dynamax" in d.action_desc:
                    action_types["dynamax"] += 1
                elif "gimmick=tera" in d.action_desc:
                    action_types["tera"] += 1
                else:
                    action_types["move"] += 1
    total_actions = sum(action_types.values())
    lines.append(f"\nAction distribution ({total_actions} total decisions):")
    for k, v in action_types.items():
        pct = 100 * v / total_actions if total_actions else 0
        lines.append(f"  {k:12s}: {v:6d}  ({pct:.1f}%)")

    # Decision count per battle
    dec_counts = [len(b.decisions) for b in battles]
    if dec_counts:
        lines.append(f"\nDecisions/battle: min={min(dec_counts)}  med={sorted(dec_counts)[len(dec_counts)//2]}  max={max(dec_counts)}")

    # Move diversity
    all_moves = set()
    for b in battles:
        for m in b.moves:
            all_moves.add(m.move_name)
    lines.append(f"Unique moves seen: {len(all_moves)}")

    # Species diversity
    all_species = set()
    for b in battles:
        for m in b.team_p1:
            all_species.add(m.species)
        for m in b.team_p2:
            all_species.add(m.species)
    lines.append(f"Unique species seen: {len(all_species)}")

    # Sample battle
    if battles:
        b = battles[0]
        lines.append(f"\n{'─' * 70}")
        lines.append(f"SAMPLE: {b.battle_id} ({b.format}, ELO {b.ratings[0] if b.ratings else '?'}→{b.ratings[1] if len(b.ratings)>1 else '?'})")
        lines.append(f"  P1: {', '.join(m.species for m in b.team_p1)}")
        lines.append(f"  P2: {', '.join(m.species for m in b.team_p2)}")
        lines.append(f"  Winner: {b.winner} ({b.winner_p})  Turns: {b.total_turns}")
        lines.append(f"  Decisions: {len(b.decisions)}  Moves: {len(b.moves)}  Switches: {len(b.switches)}")
        # First 5 decisions
        lines.append(f"  First 5 decisions:")
        for d in b.decisions[:5]:
            lines.append(f"    turn {d.turn:2d}  {d.actor}  action={d.action_idx:2d}  {d.action_desc}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="Path to JSONL file from fetch.py")
    p.add_argument("--max", type=int, default=None, help="Max battles to parse")
    p.add_argument("--report", action="store_true", help="Print quality report")
    p.add_argument("--show", type=int, default=0, help="Show first N battles in detail")
    p.add_argument("--out", default=None, help="Write parsed battles to JSON")
    args = p.parse_args()

    print(f"Parsing {args.input} ...", file=sys.stderr)
    battles = parse_jsonl(args.input, args.max)
    print(f"Parsed {len(battles)} battles.", file=sys.stderr)

    if args.report:
        print(quality_report(battles))

    if args.show > 0:
        for i, b in enumerate(battles[:args.show]):
            print(f"\n{'=' * 70}")
            print(f"Battle {i}: {b.battle_id} ({b.format})")
            print(f"  Gen {b.gen}  ELO: {b.ratings}")
            print(f"  P1: {', '.join(m.species for m in b.team_p1)}")
            print(f"  P2: {', '.join(m.species for m in b.team_p2)}")
            print(f"  Winner: {b.winner} ({b.winner_p})  Turns: {b.total_turns}")
            print(f"  Decisions ({len(b.decisions)}):")
            for d in b.decisions[:20]:
                print(f"    t{d.turn:2d} {d.actor} a={d.action_idx:2d} {d.action_desc}")
            if len(b.decisions) > 20:
                print(f"    ... {len(b.decisions) - 20} more")

    if args.out:
        # Serialize to JSON
        def battle_to_dict(b: ParsedBattle) -> dict:
            return {
                "id": b.battle_id,
                "formatid": b.formatid,
                "format": b.format,
                "gen": b.gen,
                "players": b.players,
                "ratings": b.ratings,
                "winner": b.winner,
                "winner_p": b.winner_p,
                "total_turns": b.total_turns,
                "team_p1": [{"slot": m.slot, "species": m.species, "gender": m.gender} for m in b.team_p1],
                "team_p2": [{"slot": m.slot, "species": m.species, "gender": m.gender} for m in b.team_p2],
                "decisions": [{"turn": d.turn, "actor": d.actor, "action": d.action_idx, "desc": d.action_desc} for d in b.decisions],
                "moves": [{"turn": m.turn, "actor": m.actor, "move": m.move_name, "idx": m.move_idx, "gimmick": m.gimmick} for m in b.moves],
                "switches": [{"turn": s.turn, "actor": s.actor, "species": s.species, "slot": s.slot, "hp": s.hp_pct} for s in b.switches],
            }
        out_data = [battle_to_dict(b) for b in battles]
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=1)
        print(f"Wrote {len(out_data)} parsed battles to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
