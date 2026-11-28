"""S3.1 POV symmetry — the p2 observation path must be a true MIRROR of the
p1 path (same state, mirrored, -> equivalent facts).

3.1 will let the learner play BOTH seats (flag-gated, off by default), so a
broken p2-POV path (the opponent's team read as own, hazards on the wrong
side, fainted counts swapped, boosts attributed to the wrong mon) would make
the encoder seat-dependent — and a long run would train on the asymmetry.
These pin the mirror invariants through the REAL belief pipeline
(``build_ctx``) on a real-shaped protocol log: no fork, no network.

Mirror invariants (``ctx1`` = p1 POV, ``ctx2`` = p2 POV of the SAME log):
  1. fainted:  ctx1.fainted[p1] == ctx2.fainted[p2]  (and p2 <-> p1)
  2. side-keyed field effects: for every hazard/screen X,
     ctx1.X["p1"] == ctx2.X["p2"]  and  ctx1.X["p2"] == ctx2.X["p1"]
     (the physical p1 side is "own" in ctx1 and reads as p2 in the swapped
     log that feeds ctx2)
  3. public mon facts: for each species S, the block in ctx1's opponent
     view of S equals the block in ctx2's own view of S (hp_frac, statuses,
     boosts, fainted, active) — and vice versa for S on the other team
  4. active: the active species on each side agrees across the two POVs
"""
from __future__ import annotations

from legality.lookup import to_id
from net.env import build_ctx


LOG = "\n".join([
    "|gen|9",
    "|tier|OU",
    "|poke|p1a|Garchomp, M|",
    "|poke|p1b|Tornadus, M|",
    "|poke|p2a|Landorus, M|",
    "|poke|p2b|Talonflame, F|",
    "|start",
    # turn 1: Landorus (p2) sets up + hazards, Garchomp (p1) attacks
    "|turn|1",
    "|switch|p1a|Garchomp|300/300|",
    "|switch|p2a|Landorus|280/280|",
    "|move|p2a: Landorus|Swords Dance",
    "|-boost|p2a: Landorus|atk|2",
    "|move|p2a: Landorus|Stealth Rock",
    "|-sidestart|p1: Garchomp|Stealth Rock|[from] move: Stealth Rock",
    "|move|p1a: Garchomp|Outrage|p2a: Landorus",
    "|-damage|p2a: Landorus|200/280|",
    "|-status|p2a: Landorus|brn",
    # turn 2: Landorus KOs Garchomp, Tornadus comes in
    "|turn|2",
    "|move|p2a: Landorus|Earth Power|p1a: Garchomp",
    "|-damage|p1a: Garchomp|0/300|",
    "|faint|p1a: Garchomp",
    "|switch|p1a|Tornadus|250/250|",
])

T1 = ("garchomp", "tornadus")   # p1's team
T2 = ("landorus", "talonflame")  # p2's team

# the side-keyed field conditions (hazards + screens/protection)
SIDE_KEYED = ("stealthrock", "spikes", "toxicspikes", "stickyweb",
              "reflect", "lightscreen", "auroraveil", "safeguard")


class _FakeMon:
    def __init__(self, species: str):
        self.species = species
        self.base_species = species
        self.moves = {}


class _FakeBattle:
    """Just enough of a poke-env battle for build_ctx: the shared protocol
    log + THIS side's team (the own-team spec source)."""

    def __init__(self, team_slots):
        self._log = LOG
        self.team = {slot: _FakeMon(sp) for slot, sp in team_slots}

    def _build_replay_log(self):
        return self._log


def _ctxs():
    b1 = _FakeBattle([("p1a", "garchomp"), ("p1b", "tornadus")])
    b2 = _FakeBattle([("p2a", "landorus"), ("p2b", "talonflame")])
    ctx1 = build_ctx(b1, "gen9ou", "p1")
    ctx2 = build_ctx(b2, "gen9ou", "p2")
    assert ctx1 is not None and ctx2 is not None, "fixture log must parse"
    return ctx1, ctx2


def _by_species(blocks, key):
    out = {}
    for blk in blocks.values():
        name = blk.get(key)
        if name:
            out[to_id(name)] = blk
    return out


def _own_species(ctx):
    return _by_species(ctx["own"], "name")


def _opp_species(ctx):
    return _by_species(ctx["opponent"], "species")


# ── sanity: the fixture is non-degenerate (both sides have real state) ──────

def test_fixture_is_non_degenerate():
    ctx1, ctx2 = _ctxs()
    own1, own2 = _own_species(ctx1), _own_species(ctx2)
    opp1, opp2 = _opp_species(ctx1), _opp_species(ctx2)
    assert set(T1) <= set(own1) and set(T2) <= set(own2)
    assert set(T1) <= set(opp1.keys()) | set(own1)   # T1 visible in ctx1
    assert set(T2) <= set(opp1.keys())
    # the state we assert on exists: a KO, a hazard, a burn, a boost
    assert any(b["fainted"] for b in own1.values())
    assert ctx1["field"]["stealthrock"].get("p1")
    assert "brn" in opp1["landorus"]["statuses"]
    assert opp1["landorus"]["boosts"].get("atk") == 2


# ── invariant 1: fainted counts mirror ──────────────────────────────────────

def test_fainted_counts_mirror():
    ctx1, ctx2 = _ctxs()
    assert ctx1["fainted"]["p1"] == ctx2["fainted"]["p2"]
    assert ctx1["fainted"]["p2"] == ctx2["fainted"]["p1"]
    # and the fixture actually has one KO on p1's side
    assert ctx1["fainted"]["p1"] == 1
    assert ctx1["fainted"]["p2"] == 0


# ── invariant 2: side-keyed field effects mirror ────────────────────────────

def test_side_keyed_field_effects_mirror():
    ctx1, ctx2 = _ctxs()
    f1, f2 = ctx1["field"], ctx2["field"]
    for k in SIDE_KEYED:
        assert f2[k].get("p2") == f1[k].get("p1"), f"{k}: p1 side mismatch"
        assert f2[k].get("p1") == f1[k].get("p2"), f"{k}: p2 side mismatch"
    # global (non-side) field state is identical
    for k in ("weather", "terrain", "trickroom", "wonderroom"):
        assert f1[k] == f2[k]
    # the fixture hazard landed on p1's side in ctx1 and reads as OWN (p2)
    # side in ctx2 — the physical fact is the same
    assert f1["stealthrock"].get("p1")
    assert f2["stealthrock"].get("p2")


# ── invariant 3: public mon facts mirror per species ────────────────────────

# the fields present on BOTH own- and opponent-side mon blocks (opponent
# blocks carry no ``active`` flag — the active is the ``ctx['active']`` slot)
PUBLIC = ("hp_frac", "statuses", "boosts", "fainted")


def _facts(blk):
    return {k: blk.get(k) for k in PUBLIC}


def test_public_facts_mirror_per_species():
    ctx1, ctx2 = _ctxs()
    own1, own2 = _own_species(ctx1), _own_species(ctx2)
    opp1, opp2 = _opp_species(ctx1), _opp_species(ctx2)
    # T2 species: opponent-view in ctx1 == own-view in ctx2
    for sp in T2:
        assert _facts(opp1[sp]) == _facts(own2[sp]), f"{sp}: ctx1.opp != ctx2.own"
    # T1 species still on the field: own-view in ctx1 == opponent-view in ctx2
    for sp in T1:
        if sp in opp2:
            assert _facts(own1[sp]) == _facts(opp2[sp]), \
                f"{sp}: ctx1.own != ctx2.opp"
        else:
            # schema fact: a FAINED opponent mon drops out of the opponent
            # blocks (it survives in the fainted count + the own-side flag)
            assert own1[sp]["fainted"], f"{sp} absent from ctx2.opp but not fainted in ctx1.own"


def test_fainted_opp_mon_drops_out_of_blocks():
    """Schema fact the mirror relies on: the KO'd Garchomp is gone from the
    p2-POV opponent blocks, and its absence is exactly the fainted count
    (1 on p1's side in both POVs)."""
    ctx1, ctx2 = _ctxs()
    opp2 = _opp_species(ctx2)
    assert "garchomp" not in opp2
    # in ctx2 the physical p1 side reads as "p2" (the log was side-swapped)
    assert ctx2["fainted"]["p2"] == 1
    assert ctx1["fainted"]["p1"] == 1


# ── invariant 4: the active species agrees across POVs ──────────────────────

def _active_species(ctx, side, blocks):
    """The species on the active slot for ``side`` (ctx['active'] is the
    active source of truth; opponent blocks carry no active flag)."""
    slot = ctx["active"].get(side)
    if slot is None:
        return None
    blk = blocks.get(slot) or {}
    name = blk.get("name") or blk.get("species")
    return to_id(name) if name else None


def test_active_species_agrees_across_povs():
    ctx1, ctx2 = _ctxs()
    own1, own2 = _own_species(ctx1), _own_species(ctx2)
    # p1's active (tornadus after the KO) is the opponent-side active in ctx2
    assert _active_species(ctx1, "p1", ctx1["own"]) == \
        _active_species(ctx2, "p2", ctx2["opponent"]) == "tornadus"
    # p2's active (landorus) is the opponent-side active in ctx1
    assert _active_species(ctx1, "p2", ctx1["opponent"]) == \
        _active_species(ctx2, "p1", ctx2["own"]) == "landorus"
    # own-side active flags agree with the active slot (own blocks carry it)
    assert own1[_active_species(ctx1, "p1", ctx1["own"])]["active"] is True
    assert own2[_active_species(ctx2, "p1", ctx2["own"])]["active"] is True
