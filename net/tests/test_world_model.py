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
from net.world_model import PokeWorldModel, State
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
