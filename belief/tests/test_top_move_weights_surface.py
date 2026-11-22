"""6.5b surface — the ``top_move_weights`` marginal in ``get_context()``.

OC-6.5b (2026-11-22, user-ruled): expose the per-move posterior WEIGHTS (the
marginal over the candidate posterior) in the SAME ORDER as ``top_moves``.
This is CONTEXT, not OBSERVATION — the encoder reads ``top_moves`` (the id
channel) and IGNORES this key, so the registry checksum is unchanged and
existing checkpoints stay loadable (the net suite's ``test_checksum_pin``
guards that). It un-locks fitness.strategy's rule-4 resist weighting, which
already reads ``top_move_weights`` when present and degrades to uniform
otherwise — so exposing it upgrades rule 4 from uniform to belief-weighted.

Run from the project root:

    CUDA_VISIBLE_DEVICES=-1 python -m pytest belief/tests/test_top_move_weights_surface.py -v
"""
from __future__ import annotations

from belief.state import BeliefState, SlotBelief
from fitness.strategy import _opp_move_type_weights


def _install(bs: BeliefState, slot: str, species: str, cands: list) -> SlotBelief:
    """White-box slot (no DB prior needed for the marginal math)."""
    sb = SlotBelief(slot, species, cands)
    bs._slots[slot] = sb
    return sb


def _cand(moves, weight):
    return {"moves": list(moves), "item": "", "ability": "x",
            "nature": "Hardy", "evs": {}, "tera": None, "weight": weight}


def test_top_move_weights_same_order_as_top_moves():
    """The weights are the marginal over the posterior, in the SAME order as
    ``top_moves`` (top-4 by weight), rounded to 4dp."""
    bs = BeliefState("gen9ou")
    # a: 0.8 + 0.2 = 1.0 ; b: 0.8 ; c: 0.2  -> order a, b, c
    cands = [_cand(["a", "b"], 0.8), _cand(["a", "c"], 0.2)]
    _install(bs, "p2a", "garchomp", cands)
    blk = bs.get_context()["opponent"]["p2a"]
    assert blk["top_moves"] == ["a", "b", "c"]
    assert blk["top_move_weights"] == [1.0, 0.8, 0.2]
    # invariant the consumer relies on: 1:1 with top_moves, same length
    assert len(blk["top_move_weights"]) == len(blk["top_moves"])


def test_top_move_weights_capped_at_top4():
    """A 6-move marginal is truncated to the top 4 (registry MAX_TOP_MOVES)."""
    bs = BeliefState("gen9ou")
    cands = [_cand([f"m{i}"], 0.9 - 0.1 * i) for i in range(6)]  # all positive
    _install(bs, "p2a", "garchomp", cands)
    blk = bs.get_context()["opponent"]["p2a"]
    assert len(blk["top_moves"]) == 4
    assert len(blk["top_move_weights"]) == 4
    # descending (the sort is by weight, descending)
    assert blk["top_move_weights"] == sorted(blk["top_move_weights"], reverse=True)


def test_bench_placeholder_carries_the_key():
    """Bench placeholders share one schema: the key is present (empty) so a
    consumer never KeyErrors on a bench slot — the same contract as ``maxhp``."""
    bs = BeliefState("gen9ou")
    bs.opp_team.append("blastoise")   # a bench species -> placeholder at p2b
    bs.active["p2"] = "p2a"
    _install(bs, "p2a", "garchomp", [_cand(["a"], 1.0)])
    ctx = bs.get_context()
    assert "top_move_weights" in ctx["opponent"]["p2b"]
    assert ctx["opponent"]["p2b"]["top_move_weights"] == []
    assert ctx["opponent"]["p2b"]["top_moves"] == []


def test_rule4_consumes_the_weights_not_uniform():
    """Once the belief exposes ``top_move_weights``, rule-4's resist input is
    belief-weighted (not the uniform degrade). A higher-weighted move type must
    contribute more to the (type, weight) list than a lower-weighted one."""
    bs = BeliefState("gen9ou")
    # candidate weights are NORMALIZED to sum to 1.0 by SlotBelief, so use
    # inputs that already do: a carries 0.8, b carries 0.2. If rule-4 were
    # running the UNIFORM degrade both would be 0.5 — so 0.8/0.2 proves the
    # belief weights are consumed.
    cands = [_cand(["a"], 0.8), _cand(["b"], 0.2)]
    _install(bs, "p2a", "garchomp", cands)
    ctx = bs.get_context()
    ctx["active"]["p2"] = "p2a"
    # move_db is a CALLABLE (name, gen) -> {"type": ...} (the get_move shape);
    # inject it so the test is hermetic (no DB) and never asserts a real type.
    def move_db(n, gen):
        return {"a": {"type": "Fire"}, "b": {"type": "Water"}}.get(n, {})
    pairs = _opp_move_type_weights(ctx, gen=9, move_db=move_db)
    d = {t: w for t, w in pairs}
    # belief-weighted: a's type carries 0.8, b's 0.2 (NOT the uniform 0.5/0.5)
    assert abs(d.get("Fire", 0.0) - 0.8) < 1e-6
    assert abs(d.get("Water", 0.0) - 0.2) < 1e-6
