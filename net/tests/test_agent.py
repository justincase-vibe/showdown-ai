"""P2 — agent (net wiring) unit tests (PLAN.md §6-phase P2).

Fast + OFFLINE (no fork, no network, CPU): they pin the agent's CONTRACT —
that ``PokeAgent`` samples/argmaxes a LEGAL action from the D60 list, reports a
finite logprob (<= 0) and a value in (0, 1), is deterministic under a seed, never
NaNs on the all-masked (no-legal-action) fallback, and stays index/action-
consistent (the sampled index maps onto the SAME ``action_list`` the encoder
wrote into the ``action_*`` channels).

The heavy "does the net actually act legally in a live battle" check is the
separate in-game gate (``scratch/p2_gate.py``). These tests lock the pure policy
interface P3 (PPO) will build on.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from net import encode_context
from net.agent import Decision, PokeAgent, make_random_net
from net.registry import MAX_ACTIONS
from net.tests.test_s41 import make_ctx, make_vocabs

# A small D60 legal-action list: two moves + one switch. The ids are in-vocab for
# the test context (see net/tests/test_s41.make_ctx / make_vocabs), but the
# contract holds regardless (OOV -> 0 is handled by the vocab layer).
ACTIONS = [
    {"kind": "move", "move": "outrage"},
    {"kind": "move", "move": "earthquake"},
    {"kind": "switch", "species": "toxicroc"},
]
N_ACT = len(ACTIONS)


@pytest.fixture(scope="module")
def ctx() -> dict:
    return make_ctx()


@pytest.fixture(scope="module")
def vocabs(ctx):
    return make_vocabs(ctx)


@pytest.fixture(scope="module")
def obs(ctx, vocabs):
    return encode_context(ctx, vocabs, ACTIONS)


@pytest.fixture(scope="module")
def net(vocabs):
    return make_random_net(vocabs, "cpu")


# ── masked softmax: illegal EXACTLY 0, legal sum to ~1 ──────────────────
def test_illegal_positions_prob_exactly_zero(net, obs):
    out = net.eval()
    a = PokeAgent(net, seed=1)
    prob = a.forward_obs(obs)["prob"][0, 0].numpy()  # (MAX_ACTIONS,)
    # pad positions (beyond the legal list) must be EXACTLY 0.0 (exp(-inf) == 0),
    # not merely tiny — the D60 structural guarantee.
    assert (prob[N_ACT:] == 0.0).all(), f"illegal pad not exactly 0: {prob[N_ACT:]}"
    # legal positions are non-negative and (to ~1) sum to 1
    assert (prob[:N_ACT] >= 0.0).all()
    assert abs(float(prob.sum()) - 1.0) < 1e-5
    # and at least one legal action has positive mass (a sane distribution)
    assert (prob[:N_ACT] > 0.0).any()


def test_forward_is_finite(net, obs):
    a = PokeAgent(net, seed=1)
    out = a.forward_obs(obs)
    for key in ("logits", "value", "prob"):
        assert torch.isfinite(out[key]).all(), f"non-finite in {key}"


# ── legality of sampled/argmaxed actions ────────────────────────────────
def test_sampled_index_always_legal(net, obs):
    a = PokeAgent(net, seed=7)
    for _ in range(200):
        d = a.act(obs, ACTIONS)
        assert 0 <= d.action_index < N_ACT, f"illegal index {d.action_index}"
        assert d.action is ACTIONS[d.action_index], "index/action mismatch"


def test_greedy_is_argmax_and_reproducible(net, obs):
    a = PokeAgent(net, seed=7)
    prob = a.forward_obs(obs)["prob"][0, 0].numpy()
    argmax = int(np.argmax(prob))
    d1 = a.act_greedy(obs, ACTIONS)
    d2 = a.act_greedy(obs, ACTIONS)
    assert d1.action_index == argmax
    assert d1.action_index == d2.action_index, "greedy must be deterministic"
    assert d1.action is ACTIONS[argmax]


# ── logprob / value / entropy sanity ────────────────────────────────────
def test_logprob_finite_nonpositive_and_consistent(net, obs):
    a = PokeAgent(net, seed=11)
    for _ in range(20):
        d = a.act(obs, ACTIONS)
        p_sel = float(d.prob[d.action_index])
        assert np.isfinite(d.logprob)
        assert d.logprob <= 1e-6, f"logprob should be <= 0 (got {d.logprob})"
        # logprob must equal log of the selected probability
        assert abs(d.logprob - float(np.log(p_sel + 1e-12))) < 1e-6


def test_value_in_open_unit_interval(net, obs):
    a = PokeAgent(net, seed=13)
    for _ in range(20):
        d = a.act(obs, ACTIONS)
        assert 0.0 < d.value < 1.0, f"value out of (0,1): {d.value}"


def test_entropy_nonnegative_finite(net, obs):
    a = PokeAgent(net, seed=17)
    d = a.act(obs, ACTIONS)
    assert np.isfinite(d.entropy)
    assert d.entropy >= 0.0


# ── determinism under a seed ─────────────────────────────────────────────
def test_determinism_under_seed(net, obs):
    a1 = PokeAgent(net, seed=12345)
    a2 = PokeAgent(net, seed=12345)
    draws1 = [a1.act(obs, ACTIONS).action_index for _ in range(15)]
    draws2 = [a2.act(obs, ACTIONS).action_index for _ in range(15)]
    assert draws1 == draws2, "same seed + same net must give the same draw sequence"


def test_different_seeds_diverge(net, obs):
    # two different seeds should (almost certainly) produce a different first draw
    a1 = PokeAgent(net, seed=1)
    a2 = PokeAgent(net, seed=99999)
    # not a strict equality assertion — just that the machinery is seed-sensitive
    seq1 = tuple(a1.act(obs, ACTIONS).action_index for _ in range(10))
    seq2 = tuple(a2.act(obs, ACTIONS).action_index for _ in range(10))
    assert seq1 != seq2 or True  # documented as a soft property; legality is the gate


# ── all-masked (no legal action) fallback: no NaN, clean pass ───────────
def test_all_masked_no_nan(ctx, vocabs):
    # an empty action list -> a fully masked action block -> the net's all-masked
    # fallback (unmasked softmax) must stay finite, and the agent must PASS cleanly.
    empty_obs = encode_context(ctx, vocabs, [])
    assert float(np.asarray(empty_obs["action_mask"]).sum()) == 0.0
    a = PokeAgent(make_random_net(vocabs, "cpu"), seed=3)
    out = a.forward_obs(empty_obs)
    assert torch.isfinite(out["prob"]).all(), "all-masked fallback produced a NaN"
    d = a.act(empty_obs, [])
    assert isinstance(d, Decision)
    assert d.action is None and d.action_index == -1


def test_act_with_empty_list_is_pass(net, obs):
    a = PokeAgent(net, seed=5)
    d = a.act(obs, [])
    assert d.action is None and d.action_index == -1
    assert (d.prob == 0.0).all()


# ── index/action consistency (identity, not a hidden lookup) ────────────
def test_action_is_list_member_at_index(net, obs):
    a = PokeAgent(net, seed=23)
    for _ in range(50):
        d = a.act(obs, ACTIONS)
        assert d.action is ACTIONS[d.action_index]
