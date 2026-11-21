"""M2a — ``PokeWorldModel`` (the real net as an MCTS WorldModel) tests.

CPU-only, random-init net (no GPU — the GPU is the LLM's). Verifies the
NON-BRANCHING adapters (``actions``/``value``/``policy``) against the REAL
``PokeNet`` on a realistic belief context (reusing ``test_s41.make_ctx``/
``make_vocabs`` — the same ``get_context()``-shaped dict the encoder is tested
on), and pins the M2b boundary (``step`` and the p2-POV orientation raise).

The value/policy tests compare the WorldModel's output to the net's OWN heads
on the identical obs — so they verify the bridge, not the net. The perspective
contract (value mapped to the root player's view) is pinned by the exact
``2V−1`` mapping.
"""
import math
import os

import pytest

from net import make_random_net, PokeAgent, encode_context
from net.world_model import (PokeWorldModel, State, _map_value, flip_decider,
                            _flip_pov, _flip_field)
from net.tests.test_s41 import make_ctx, make_vocabs


def _make(vocabs, seed=1234):
    """A random-init net + agent + WorldModel (deterministic under ``seed``)."""
    net = make_random_net(vocabs, device="cpu", seed=seed)
    agent = PokeAgent(net, seed=seed, device="cpu")
    return net, PokeWorldModel(agent, vocabs)


# ── actions: the D60 legal list, in the net's training order ─────────────────
def test_actions_p1_moves_plus_bench():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    _, wm = _make(v)
    st = State(ctx, "p1")
    acts = wm.actions(st)
    moves = {a["move"] for a in acts if a["kind"] == "move"}
    sws = {a["species"] for a in acts if a["kind"] == "switch"}
    # the active mon (p1a) has 4 known moves; the alive bench is p1b + p1c
    assert moves == {"earthquake", "stealthrock", "icebeam", "roar"}
    assert sws == {"volcarona", "dragonite"}
    # order the net was trained on: moves first, then bench (p1b, p1c)
    assert [a["kind"] for a in acts] == ["move"] * 4 + ["switch"] * 2
    # bench order is stable (p1b before p1c)
    assert [a["species"] for a in acts if a["kind"] == "switch"] == ["volcarona", "dragonite"]


def test_actions_exclude_fainted_bench():
    ctx = make_ctx()
    ctx["own"]["p1b"]["fainted"] = True   # a gone bench mon is not switchable
    v = make_vocabs(ctx)
    _, wm = _make(v)
    sws = {a["species"] for a in wm.actions(State(ctx, "p1")) if a["kind"] == "switch"}
    assert "volcarona" not in sws
    assert sws == {"dragonite"}


# ── value: the net's value head, mapped to the ROOT player's perspective ─────
def test_value_is_net_value_head_mapped_to_root():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    net, wm = _make(v, seed=7)
    st = State(ctx, "p1")
    acts = wm.actions(st)
    out = wm.agent.forward_obs(encode_context(ctx, v, acts))
    V = float(out["value"][0, 0, 0].item())           # (0,1), p1-POV
    # decider (p1) IS the root -> the + mapping, in reward space (−1, +1)
    expected = 2.0 * V - 1.0
    assert wm.value(st) == pytest.approx(expected, abs=1e-6)
    assert -1.0 <= wm.value(st) <= 1.0


# ── the ROOT-VIEW value mapping, in isolation (M2b's perspective contract) ──
def test_map_value_root_and_opponent_decider():
    """``_map_value`` is the PURE net-free perspective map. Testing it directly
    proves BOTH branches of the M2b value contract without needing the p2-POV
    observation orientation (M2b) or effect bookkeeping (OC-6):

      * decider IS the root   -> 2V − 1   (root's own win-prob, reward space)
      * decider is the opp    -> −(2V − 1) (zero-sum: opp's win is root's loss)
    """
    V = 0.8
    # root decider: + mapping
    assert _map_value(V, "p1", "p1") == 2.0 * V - 1.0
    # opponent decider: sign flip (zero-sum)
    assert _map_value(V, "p2", "p1") == -(2.0 * V - 1.0)
    # symmetric: whichever side is the root, the root's own view is +, the
    # opponent's view is − (the root is not hard-coded to "p1")
    assert _map_value(V, "p2", "p2") == 2.0 * V - 1.0
    assert _map_value(V, "p1", "p2") == -(2.0 * V - 1.0)
    # range: a (0,1) win-prob always maps into (−1, +1)
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        for d in ("p1", "p2"):
            assert -1.0 <= _map_value(v, d, "p1") <= 1.0


# ── value + policy are deterministic (random-init, CPU) ────────────────────
def test_value_and_policy_deterministic_under_seed():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    _, wm = _make(v, seed=23)
    st = State(ctx, "p1")
    acts = wm.actions(st)
    assert wm.value(st) == wm.value(st)
    assert wm.policy(st, acts) == wm.policy(st, acts)


# ── policy: the net's masked distribution over the legal set ─────────────────
def test_policy_matches_net_masked_prob_and_sums_to_one():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    net, wm = _make(v, seed=11)
    st = State(ctx, "p1")
    acts = wm.actions(st)
    priors = wm.policy(st, acts)
    assert len(priors) == len(acts)
    assert all(p >= 0.0 for p in priors)
    assert sum(priors) == pytest.approx(1.0, abs=1e-3)   # masked softmax over the legal set
    # each prior == the net's masked prob at that (legal) action position
    ref = wm.agent.forward_obs(encode_context(ctx, v, acts))["prob"][0, 0]
    ref = ref.detach().cpu().numpy()
    for i, p in enumerate(priors):
        assert p == pytest.approx(float(ref[i]), abs=1e-6)


def test_policy_is_position_identity_not_index():
    """The pointer head scores actions by IDENTITY. Shuffling the action list
    must give the SAME per-action prior for each action (matched by identity),
    so the WorldModel's priors track the action, not its slot (matters for
    MCTS, where the legal list order can vary)."""
    ctx = make_ctx()
    v = make_vocabs(ctx)
    _, wm = _make(v, seed=42)
    st = State(ctx, "p1")
    acts = wm.actions(st)

    def key(a):  # a hashable identity key for an action (dicts are not hashable)
        return (a["kind"], a.get("move"), a.get("species"))

    p1 = {key(a): p for a, p in zip(acts, wm.policy(st, acts))}
    shuffled = list(reversed(acts))
    p2 = {key(a): p for a, p in zip(shuffled, wm.policy(st, shuffled))}
    assert set(p1) == set(p2)  # same action identities, no collisions
    for k in p1:
        assert p1[k] == pytest.approx(p2[k], abs=1e-6)


# ── integration: a REAL trained checkpoint (not the random proxy) ────────────
def test_real_checkpoint_bridge_is_sane():
    """Run the bridge on a REAL trained net. A random-init net is only a proxy:
    this catches real-artifact bridge bugs (NaN/shape, an OOV action id that
    silently resolves to <pad>, a degenerate head) that a random net cannot
    reveal. Verifies value is finite + in reward space and policy is a valid
    distribution over the legal set. Skips if the scratch checkpoint is absent
    (it is a scratch artifact, not a guaranteed fixture)."""
    from net import PokeNet, build_vocabs
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ckpt = os.path.join(root, "scratch", "s43_bc_n2000.pt")
    if not os.path.exists(ckpt):
        pytest.skip("real checkpoint not present (scratch artifact)")
    vocabs = build_vocabs()
    net = PokeNet.load_file(ckpt, vocabs, device="cpu")
    wm = PokeWorldModel(PokeAgent(net, seed=1, device="cpu"), vocabs)
    st = State(make_ctx(), "p1")
    val = wm.value(st)
    acts = wm.actions(st)
    priors = wm.policy(st, acts)
    assert math.isfinite(val) and -1.0 <= val <= 1.0
    assert len(priors) == len(acts) and acts
    assert all(math.isfinite(p) and p >= 0.0 for p in priors)
    assert sum(priors) == pytest.approx(1.0, abs=1e-3)


# ── the M2b boundary (not built yet) ─────────────────────────────────────────
def test_step_is_m2b():
    ctx = make_ctx()
    _, wm = _make(make_vocabs(ctx))
    st = State(ctx, "p1")
    with pytest.raises(NotImplementedError):
        wm.step(st, {"kind": "move", "move": "earthquake"})


def test_p2_decider_is_m2b_boundary():
    """The opponent-POV (decider='p2') orientation is M2b (paired with step);
    M2a must raise on it rather than silently produce a wrong (untested) swap."""
    ctx = make_ctx()
    _, wm = _make(make_vocabs(ctx))
    with pytest.raises(NotImplementedError):
        wm.actions(State(ctx, "p2"))
    with pytest.raises(NotImplementedError):
        wm.value(State(ctx, "p2"))


# ── State isolation: M2b's step must build children from a DEEP copy ─────────
def test_state_copy_is_independent_of_parent():
    """M2b's step builds a child State and mutates its ctx. That mutation must
    NEVER leak into the parent node's ctx (a frozen dataclass does not make the
    dict immutable) — ``State.copy`` is the safe primitive that guarantees this.
    A shared-dict leak would corrupt the ENTIRE MCTS tree (parent + descendants
    would share one live dict)."""
    ctx = {
        "turn": 3,
        "own": {"p1a": {"hp_frac": 1.0, "moves": ["earthquake"]}},
        "opponent": {"p2a": {"species": "blastoise"}},
        "active": {"p1": "p1a", "p2": "p2a"},
    }
    parent = State(ctx, "p1")
    child = parent.copy("p2")                    # deep copy + flipped decider
    # mutate the CHILD's ctx in place (exactly what M2b's step will do)
    child.ctx["own"]["p1a"]["hp_frac"] = 0.0
    child.ctx["own"]["p1a"]["moves"].append("roar")
    child.ctx["opponent"]["p2a"]["species"] = "garchomp"
    child.ctx["turn"] = 4
    # the PARENT's ctx must be completely unaffected
    assert parent.ctx["own"]["p1a"]["hp_frac"] == 1.0
    assert parent.ctx["own"]["p1a"]["moves"] == ["earthquake"]
    assert parent.ctx["opponent"]["p2a"]["species"] == "blastoise"
    assert parent.ctx["turn"] == 3
    assert child.decider == "p2" and parent.decider == "p1"


def test_state_copy_keeps_decider_by_default():
    ctx = {"own": {"p1a": {"hp_frac": 1.0}}}
    parent = State(ctx, "p1")
    child = parent.copy()
    assert child.decider == parent.decider == "p1"
    child.ctx["own"]["p1a"]["hp_frac"] = 0.0
    assert parent.ctx["own"]["p1a"]["hp_frac"] == 1.0   # isolated


def test_flip_decider():
    assert flip_decider("p1") == "p2"
    assert flip_decider("p2") == "p1"


# ── _flip_pov: the invertible structural core of the p2-POV swap (M2b) ──────
def test_flip_pov_round_trip_is_identity():
    """``_flip_pov`` is a PERMUTATION of the context, not a lossy transform:
    applying it twice must be the IDENTITY. This is the correctness proof for
    the fiddliest part of M2b's ``_orient`` (slot remap + own<->opponent +
    field side-flip + fainted/active swap) — and it is INDEPENDENT of OC-9 (the
    posterior semantics), which this deliberately does NOT decide."""
    ctx = {
        "turn": 7,
        "fainted": {"p1": 0, "p2": 1},
        "active": {"p1": "p1a", "p2": "p2b"},
        "field": {
            "weather": "raindance", "terrain": "",
            "trickroom": True, "tailwind_side": "p1",
            "reflect": {"p1": 1, "p2": 0},
            "spikes": {"p1": 2, "p2": 0},
        },
        "own": {"p1a": {"name": "garchomp", "ability": "roughskin",
                        "item": "lifeorb", "moves": ["earthquake"], "hp_frac": 0.8}},
        "opponent": {"p2b": {"species": "blastoise",
                             "posterior": [{"ability": "raindish", "moves": ["surf"], "weight": 0.9}],
                             "top_moves": ["surf"], "moves": ["surf"],
                             "confidence": 0.9, "hp_frac": 0.5}},
    }
    assert _flip_pov(_flip_pov(ctx)) == ctx


def test_flip_pov_maps_sides_fields_and_slots():
    """The single flip maps each part to the RIGHT side: own<->opponent (slot
    remap + name<->species schema), field sides, fainted, active. The
    posterior/top_moves ride along with their block (carried opaquely — OC-9
    decides their SEMANTICS, not this structural move)."""
    ctx = {
        "turn": 7,
        "fainted": {"p1": 0, "p2": 1},
        "active": {"p1": "p1a", "p2": "p2b"},
        "field": {"tailwind_side": "p1", "reflect": {"p1": 1, "p2": 0},
                  "spikes": {"p1": 2, "p2": 0}, "trickroom": True},
        "own": {"p1a": {"name": "garchomp", "moves": ["earthquake"]}},
        "opponent": {"p2b": {"species": "blastoise", "top_moves": ["surf"]}},
    }
    p2 = _flip_pov(ctx)
    # own<->opponent; the slot LETTER is preserved across the p1/p2 axis
    assert set(p2["own"]) == {"p1b"} and set(p2["opponent"]) == {"p2a"}
    # schema: opponent's species -> own name (redundant key dropped)
    assert p2["own"]["p1b"]["name"] == "blastoise"
    assert "species" not in p2["own"]["p1b"]
    assert p2["opponent"]["p2a"]["species"] == "garchomp"
    assert "name" not in p2["opponent"]["p2a"]
    # posterior/top_moves carried opaquely with their block (OC-9 stays open)
    assert p2["own"]["p1b"]["top_moves"] == ["surf"]
    # field sides flipped; side-neutral field passes through
    assert p2["field"]["tailwind_side"] == "p2"
    assert p2["field"]["reflect"] == {"p1": 0, "p2": 1}
    assert p2["field"]["spikes"] == {"p1": 0, "p2": 2}
    assert p2["field"]["trickroom"] is True
    # fainted + active swapped (active slots remapped across the axis)
    assert p2["fainted"] == {"p1": 1, "p2": 0}
    assert p2["active"] == {"p1": "p1b", "p2": "p2a"}


def test_flip_field_covers_every_side_keyed_field_the_encoder_reads():
    """The p2-POV flip must swap EVERY per-side field the observation reads, or
    the flipped POV would silently read one field from the wrong side. Both the
    flip and the encoder now derive from ``encoder.ALL_SIDE_KEYED_FIELDS``
    (single source of truth), so this pins the coupling: add a per-side field to
    the encoder and the flip automatically covers it — and this test verifies it.
    (Catches the exact drift a hand-maintained parallel list would invite.)"""
    from net.encoder import ALL_SIDE_KEYED_FIELDS
    # non-empty and the known screens/guards + hazards are all present
    assert {"reflect", "spikes", "toxicspikes"} <= set(ALL_SIDE_KEYED_FIELDS)
    for name in ALL_SIDE_KEYED_FIELDS:
        assert _flip_field({name: {"p1": 1, "p2": 0}})[name] == {"p1": 0, "p2": 1}, name
