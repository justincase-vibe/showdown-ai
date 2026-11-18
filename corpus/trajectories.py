"""S4.0 decision-point walker.

Replay one battle log through the EXISTING belief stack, in order::

    log lines -> LogAdapter.feed_line -> events
              -> BeliefState.update   (per event)
              -> BeliefState.get_context()   (snapshot at turn start)

At each DECISION point — a ``MoveUsed`` (any turn) or an in-battle ``SwitchIn``
(a switch seen after the first ``|turn|``; the pre-turn-1 lead selections are
forced and excluded) — emit a decision carrying the turn-START observation
snapshot. The turn-start snapshot avoids leaking the opponent's same-turn action
(both players choose simultaneously; the observation is the state BEFORE the
turn resolves). Both POVs are captured: ``get_context()`` already exposes both
sides' slot blocks, and a decision's ``pov`` (``p1``/``p2``) tells the S4.1
encoder which side is "self". (A full two-sided posterior is an S4.1 refinement;
S4.0 only verifies the pipeline yields valid contexts for both sides.)

The walker doubles as the bug-catcher (3.7): it counts and samples ``Unknown``
(unmapped) log lines so a human-ladder tag the fork-fuzz never saw is CAUGHT
(the S4.0 gate asserts 0) instead of silently dropped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from belief.events import BattleEnd, MoveUsed, SwitchIn, Turn, Unknown
from belief.log_events import LogAdapter
from belief.state import BeliefState, build_prior
from legality import lookup as _lookup


@dataclass
class Decision:
    """One human decision: a move or a switch, with its legality verdict."""
    turn: int
    pov: str          # 'p1' | 'p2' — the side that made this decision
    kind: str         # 'move' | 'switch'
    action: str       # move id (move) or species id (switch)
    species: str      # the acting mon's species (the legality reference)
    legal: bool = False

    def to_dict(self) -> Dict:
        return {"turn": self.turn, "pov": self.pov, "kind": self.kind,
                "action": self.action, "species": self.species, "legal": self.legal}

    @classmethod
    def from_dict(cls, d: Dict) -> "Decision":
        """Inverse of :meth:`to_dict` (the corpus-trace cache round-trip)."""
        return cls(turn=d["turn"], pov=d["pov"], kind=d["kind"], action=d["action"],
                   species=d["species"], legal=d.get("legal", False))


@dataclass
class BattleTrace:
    """The corpus entry for one battle: decisions + turn-start observation history.

    ``context_seq[i]`` is the turn-start ``get_context()`` snapshot for turn
    ``i + 1`` (log turns are 1-indexed). A training sample for a decision at
    turn ``t`` = ``(context_seq[:t], decision)`` — the observation history up
    to that turn (the GRU input the S4.2 net consumes).
    """
    battle_id: str
    formatid: str
    rating: float
    winner_side: Optional[str] = None   # 'p1' | 'p2' | None (tie / unknown)
    n_turns: int = 0
    skipped: Optional[str] = None       # non-None => this battle was not replayed
    unknown_count: int = 0
    unknown_tags: Dict[str, int] = field(default_factory=dict)   # tag -> count
    unknown_samples: List[str] = field(default_factory=list)     # capped raw lines
    decisions: List[Decision] = field(default_factory=list)
    context_seq: List[Dict] = field(default_factory=list)

    @property
    def n_decisions(self) -> int:
        return len(self.decisions)

    @property
    def n_legal(self) -> int:
        return sum(1 for d in self.decisions if d.legal)

    @property
    def switch_rate(self) -> float:
        n = len(self.decisions)
        return (sum(1 for d in self.decisions if d.kind == "switch") / n) if n else 0.0

    @property
    def has_both_povs(self) -> bool:
        return {"p1", "p2"} <= {d.pov for d in self.decisions}

    def to_dict(self) -> Dict:
        return {
            "battle_id": self.battle_id, "formatid": self.formatid,
            "rating": self.rating, "winner_side": self.winner_side,
            "n_turns": self.n_turns, "skipped": self.skipped,
            "unknown_count": self.unknown_count, "unknown_tags": self.unknown_tags,
            "decisions": [d.to_dict() for d in self.decisions],
            "context_seq": self.context_seq,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "BattleTrace":
        """Inverse of :meth:`to_dict`. Used to load a battle back from the
        on-disk trace cache (``corpus.cache``) without re-replaying it through
        the belief stack (the ~tens-of-seconds/battle node-spawn cost)."""
        return cls(
            battle_id=d["battle_id"], formatid=d["formatid"], rating=d["rating"],
            winner_side=d.get("winner_side"), n_turns=d.get("n_turns", 0),
            skipped=d.get("skipped"), unknown_count=d.get("unknown_count", 0),
            unknown_tags=dict(d.get("unknown_tags", {})),
            unknown_samples=list(d.get("unknown_samples", [])),
            decisions=[Decision.from_dict(x) for x in d.get("decisions", [])],
            context_seq=list(d.get("context_seq", [])),
        )


# ── helpers ────────────────────────────────────────────────────────────
def _fmt_of(battle: dict) -> Optional[str]:
    """The format id (a ``legality`` registry key) this battle is in."""
    fid = battle.get("formatid")
    return str(fid) if fid else None


def _tag_of(raw: str) -> str:
    """The Showdown tag of a raw ``|...`` line ('' if unparseable)."""
    parts = raw.split("|", 2)
    return parts[1] if len(parts) > 1 else "?"


def _team_species(lines: List[str]) -> List[str]:
    """Every species on either side (``|poke|`` team lines, ``|switch|`` fallback).

    Order-preserving, de-duplicated display names (the belief/legality stack
    normalizes with ``toID`` itself)."""
    seen: List[str] = []
    seen_set = set()
    for line in lines:
        if not isinstance(line, str) or not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 4:
            continue
        if parts[1] in ("poke", "switch"):
            sp = parts[3].split(",")[0].strip()
            if sp and sp not in seen_set:
                seen_set.add(sp)
                seen.append(sp)
    return seen


def _own_team(lines: List[str]) -> Dict[str, Dict]:
    """p1's own-team spec for the belief (D68): slot -> {'name': species}.

    The corpus walker creates ``BeliefState(fid)`` with NO own team, so
    ``get_context()['own']`` is empty and the encoder emits all-zero
    ``own_*`` tensors — the net would train blind to its own bench (which
    mons benched, their HP/status). This parses the team out of the log and
    hands it to the belief; the belief/encoder/registry stay unchanged.

    ``|poke|`` team-preview lines come in two shapes (both occur in the
    corpus; the fork emits the side shape — ``battle.js``:
    ``this.add("poke", pokemon.side.id, details, "")``):
      * side shape: ``|poke|p1|Lokix, M|``       — p1 slots a..f in list order
      * slot shape: ``|poke| p1a 1 | Kyurem |``   — explicit slot in the first
        payload field
    ``|switch|`` / ``|drag|`` / ``|replace|`` fill any slot the preview did
    not name (a log without a team preview stays covered). Only ``name`` is
    recoverable from the log: ability/item are hidden until a reveal, and a
    reveal updates the belief's posterior, not this static spec. The p2 side
    needs no spec: its slots already carry species + facts via
    ``add_opponent`` (a p2-POV sample reads them through the same channels).
    """
    order = "abcdef"
    team: Dict[str, Dict] = {}
    n_side = 0
    for line in lines:
        if not isinstance(line, str):
            continue
        parts = line.split("|")
        if len(parts) < 4 or parts[1] not in ("poke", "switch", "drag", "replace"):
            continue
        key = (parts[2].strip().split() or [""])[0]
        name = parts[3].split(",")[0].strip()
        if not name:
            continue
        slot: Optional[str] = None
        if key == "p1":
            if n_side < 6:
                slot = f"p1{order[n_side]}"
                n_side += 1
        elif key.startswith("p1") and len(key) == 3 and key[2] in "abcdef":
            slot = key
        if slot is None or slot in team:
            continue
        team[slot] = {"name": name}
    return team


def _is_move_legal(move: str, species: str, fid: Optional[str],
                   pool_cache: Optional[Dict[Tuple[str, str], set]] = None) -> bool:
    """Static legality: is ``move`` a legal move for ``species`` in the format?

    This is the S4.0 "yields a legal action" check — a move-pool test (it ignores
    dynamic state the log never reveals: PP, status locks, Choice, …).

    ``pool_cache`` memoizes the (species, format) -> legal-move-set lookup:
    it is static within a battle, and one battle has ~60 move decisions but
    only ~12 distinct (species, format) pairs — without the cache every
    decision opens a fresh SQLite connection (legal_moves does connect/query/
    close per call)."""
    if not (fid and species and move):
        return False
    if pool_cache is not None:
        key = (species, fid)
        pool = pool_cache.get(key)
        if pool is None:
            try:
                pool = set(_lookup.legal_moves(species, fid))
            except Exception:
                pool = set()  # species with no pool -> no legal move (same as before)
            pool_cache[key] = pool
        return move in pool
    try:
        return move in _lookup.legal_moves(species, fid)
    except Exception:
        return False


# ── the walker ─────────────────────────────────────────────────────────
def replay_battle(battle: dict, max_unknown_samples: int = 30) -> BattleTrace:
    """Replay one battle log through the belief adapter.

    Returns a ``BattleTrace``. If the battle cannot be replayed cleanly (a
    species with no prior in the format, an unsupported format, …) then
    ``trace.skipped`` holds the reason and the decisions/contexts are empty —
    the corpus stays clean rather than carrying a corrupt belief.
    """
    fid = _fmt_of(battle)
    trace = BattleTrace(
        battle_id=str(battle.get("id", "")),
        formatid=fid or "",
        rating=float(battle.get("rating") or 0.0),
    )
    log = battle.get("log")
    lines = log.split("\n") if isinstance(log, str) else list(log or [])

    # (1) Prior pre-scan (DB) — can a usable prior be built for every species?
    #     A no-prior species would raise inside add_opponent MID-replay and corrupt
    #     the belief, so we mark the battle skipped (clean corpus) with a precise,
    #     reportable reason. This is a DATA-coverage check, independent of the
    #     adapter (the adapter gate below runs regardless).
    prior_ok = True
    if not fid:
        trace.skipped = "no formatid in battle record"
        prior_ok = False
    else:
        for sp in _team_species(lines):
            try:
                build_prior(sp, fid)
            except ValueError as e:
                trace.skipped = f"no prior: {e}"
                prior_ok = False
                break
            except Exception as e:  # format not in registry, DB error, …
                trace.skipped = f"{type(e).__name__}: {e}"
                prior_ok = False
                break

    # (2) Adapter pass — count Unknown on EVERY line (the S4.0 "0 unknown-tag"
    #     gate is DB-independent: it must hold even for skipped battles). Build
    #     the belief + emit decisions only when the prior is valid.
    adapter = LogAdapter()
    # D68: give the belief its own-team spec (p1) so get_context()['own'] is
    # populated — the corpus previously left it empty (the net would train
    # blind to its own bench). The belief API already supports this;
    # belief/encoder/registry are unchanged.
    own_team = _own_team(lines)
    belief = BeliefState(fid, own_team=own_team) if prior_ok else None
    current_turn = 0
    seen_first_turn = False
    pool_cache: Dict[Tuple[str, str], set] = {}   # (species, format) -> legal moves

    try:
        for line in lines:
            ev = adapter.feed_line(line)
            if ev is None:
                continue
            if isinstance(ev, Unknown):
                trace.unknown_count += 1
                tag = _tag_of(ev.raw)
                trace.unknown_tags[tag] = trace.unknown_tags.get(tag, 0) + 1
                if len(trace.unknown_samples) < max_unknown_samples:
                    trace.unknown_samples.append(ev.raw)
                continue
            if belief is None:
                continue  # prior-invalid battle — we only counted unknowns
            if isinstance(ev, Turn):
                belief.update(ev)  # turn bookkeeping (speed-flush + per-turn reset)
                current_turn = ev.n
                trace.n_turns = max(trace.n_turns, ev.n)
                seen_first_turn = True
                # turn-START snapshot: after the turn reset, before this turn's
                # actions -> no same-turn opponent leak.
                trace.context_seq.append(belief.get_context())
                continue
            belief.update(ev)
            if isinstance(ev, MoveUsed):
                species = adapter.species_by_slot.get(ev.slot, "")
                trace.decisions.append(Decision(
                    turn=current_turn, pov=ev.side, kind="move",
                    action=ev.move, species=species,
                    legal=_is_move_legal(ev.move, species, fid, pool_cache)))
            elif isinstance(ev, SwitchIn) and seen_first_turn and not ev.forced:
                # a player-chosen in-battle switch (a |drag|/|replace| is a game
                # FORCED entry, not a decision — excluded from the switch-rate).
                trace.decisions.append(Decision(
                    turn=current_turn, pov=ev.side, kind="switch",
                    action=ev.species, species=ev.species, legal=True))
            elif isinstance(ev, BattleEnd):
                trace.winner_side = ev.winner_side
    except ValueError as e:
        # a no-prior species slipped past the pre-scan (revealed only mid-battle)
        trace.skipped = f"no prior mid-replay: {e}"
        trace.decisions = []
        trace.context_seq = []
    except Exception as e:  # noqa: BLE001 — one bad battle must not kill the corpus run
        # a genuinely unexpected adapter/belief crash: mark skipped with a precise
        # reason (surfaces in the corpus report's skipped_reasons, stays counted)
        # instead of aborting the whole replay_many on one malformed battle.
        trace.skipped = f"replay error: {type(e).__name__}: {e}"
        trace.decisions = []
        trace.context_seq = []

    return trace


def replay_many(battles: List[dict], max_unknown_samples: int = 30) -> List[BattleTrace]:
    """Replay a list of battles; one trace each (skipped battles are marked)."""
    return [replay_battle(b, max_unknown_samples) for b in battles]
