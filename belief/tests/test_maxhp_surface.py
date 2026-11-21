"""M2b data-flow — the ``maxhp`` surface in ``get_context()``.

The MCTS ``step`` turns a predicted-damage delta into an HP-fraction update
(``hp_frac - damage / maxhp``). For that it needs each mon's max HP in the
CONTEXT (not the observation). This locks in that surface:

  * the ACTIVE mon's max HP (own + opponent) is exposed from the belief's
    ground-truth ``Facts.maxhp`` (the switch-in line reveals hp/maxhp);
  * bench placeholders (own bench / opponent preview bench) carry
    ``maxhp=None`` — but the KEY is always present, so every mon block shares
    one schema and the MCTS step reads ``block["maxhp"]`` uniformly without a
    KeyError on a bench slot;
  * a mon whose max HP is not yet revealed exposes ``maxhp=None`` (step must
    tolerate that and fall back, never crash);
  * ``maxhp`` is CONTEXT, not OBSERVATION — the encoder reads specific fields
    and ignores it, so the registry checksum is unchanged and existing
    checkpoints stay loadable (the net suite verifies the encoder still runs).

Public info (no leak): the switch-in line reveals hp/maxhp to both players, so
exposing it is not a hidden-information leak — it is the same value the
opponent already sees.

Decision-independent: this is the shared DATA every viable M2b option needs;
it encodes none of the OC-6 (bookkeeping fidelity) or OC-9 (posterior
semantics) choices, so it is safe to build before those are decided.

Run from the project root:

    CUDA_VISIBLE_DEVICES=-1 python -m pytest belief/tests/test_maxhp_surface.py -v
"""
from __future__ import annotations

from belief import events as E
from belief.state import BeliefState, Facts, SlotBelief


# ── helpers (mirror test_volatiles_fainted.py) ───────────────────────────────

def cand(**kw):
    base = {"moves": ["outrage", "a", "b", "c"], "item": "", "ability": "x",
            "nature": "Hardy", "evs": {}, "tera": None, "weight": 1.0}
    base.update(kw)
    return base


def install(bs: BeliefState, slot: str, species: str, n: int = 2) -> SlotBelief:
    """White-box slot (no DB prior needed for Facts-level tests)."""
    cands = [cand(moves=["m%d" % i], ability=chr(97 + i), weight=1.0 / n)
             for i in range(n)]
    sb = SlotBelief(slot, species, cands)
    bs._slots[slot] = sb
    return sb


# ── opponent active mon: maxhp from the belief's ground truth ───────────────

def test_opp_active_maxhp_exposed():
    bs = BeliefState("gen9ou")
    sb = install(bs, "p2a", "garchomp")
    sb.facts.maxhp = 183
    sb.facts.hp = 183
    ctx = bs.get_context()
    assert ctx["opponent"]["p2a"]["maxhp"] == 183


def test_opp_active_maxhp_via_switchin_log():
    """The switch-in line reveals hp/maxhp — the belief records it as ground
    truth and get_context exposes it (the path the calc/step actually rely on)."""
    bs = BeliefState("gen9ou")
    install(bs, "p2a", "garchomp")
    bs.update(E.SwitchIn(side="p2", slot="p2a", species="garchomp", hp=183, maxhp=183))
    ctx = bs.get_context()
    assert ctx["opponent"]["p2a"]["maxhp"] == 183


# ── own active mon: maxhp from the facts (team is fully known) ───────────────

def test_own_active_maxhp_exposed():
    bs = BeliefState("gen9ou", own_team={"p1a": {"name": "garchomp"}})
    bs.active["p1"] = "p1a"
    f = Facts()
    f.hp = 120
    f.maxhp = 183
    bs.own_facts["p1a"] = f
    ctx = bs.get_context()
    assert ctx["own"]["p1a"]["maxhp"] == 183


# ── bench placeholders: maxhp key present but None (uniform schema) ──────────

def test_opp_bench_placeholder_maxhp_is_none_not_missing():
    """A bench mon with no observed max HP must carry maxhp=None (not a missing
    key) so every mon block shares one schema — the MCTS step reads
    block["maxhp"] uniformly and must never KeyError on a bench slot."""
    bs = BeliefState("gen9ou")
    install(bs, "p2a", "garchomp")     # the active belief at p2a
    bs.opp_team.append("blastoise")    # a bench species -> placeholder at p2b
    ctx = bs.get_context()
    assert "maxhp" in ctx["opponent"]["p2b"]
    assert ctx["opponent"]["p2b"]["maxhp"] is None


def test_own_bench_maxhp_is_none_not_missing():
    """Own bench mon (no observed facts yet) -> maxhp=None, key present."""
    bs = BeliefState("gen9ou", own_team={
        "p1a": {"name": "garchomp"},    # active
        "p1b": {"name": "blastoise"},   # bench (no facts)
    })
    bs.active["p1"] = "p1a"
    f = Facts()
    f.hp = 183
    f.maxhp = 183
    bs.own_facts["p1a"] = f
    ctx = bs.get_context()
    assert ctx["own"]["p1a"]["maxhp"] == 183      # active carries its real maxhp
    assert "maxhp" in ctx["own"]["p1b"]           # bench: key present
    assert ctx["own"]["p1b"]["maxhp"] is None     # bench: no observed maxhp yet


# ── unknown maxhp: facts.maxhp None -> context maxhp None (no crash) ─────────

def test_unknown_maxhp_exposes_none():
    bs = BeliefState("gen9ou")
    sb = install(bs, "p2a", "garchomp")
    sb.facts.maxhp = None   # not yet revealed (pre-first-switch, etc.)
    ctx = bs.get_context()
    assert ctx["opponent"]["p2a"]["maxhp"] is None
