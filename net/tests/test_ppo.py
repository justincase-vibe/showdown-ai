"""P3 (PPO core) gate tests — GAE + buffer + the PPO update (PLAN.md §6-phase P3).

Fast + OFFLINE (no fork, CPU): they pin the PPO TRAINING CORE that P4 will scale
up. The most important is the **policy-direction "disaster guard"** — the analog
of the S4.2 overfit-32 "never learns" check: a PPO update with a positive
advantage must make the net MORE likely to choose the stored actions, and a
negative advantage LESS likely (sign-correct gradients flowing through the pointer
head to the embeddings), with finite loss and no NaN.
"""
from __future__ import annotations

import numpy as np
import pytest

from net import encode_context
from net.agent import make_random_net
from net.ppo import (PPOTrainer, RolloutBuffer, RolloutStep, chosen_logprobs,
                     gae, steps_to_batch)
from net.tests.test_s41 import make_ctx, make_vocabs
from net.registry import LAYOUT

# in-vocab action pool (from the test context's top_moves / opponent species)
MOVE_POOL = ["outrage", "earthquake", "dragonclaw", "roost"]
SWITCH_POOL = ["toxicroc", "garchomp"]


def _random_action_lists(rng, n):
    """n varied legal-action lists (2..4 entries) for realistic distinct obs."""
    lists = []
    for _ in range(n):
        acts = []
        for m in rng.choice(MOVE_POOL, size=rng.randint(1, 3), replace=False):
            acts.append({"kind": "move", "move": str(m)})
        for s in rng.choice(SWITCH_POOL, size=rng.randint(1, 2), replace=False):
            acts.append({"kind": "switch", "species": str(s)})
        lists.append(acts)
    return lists


def make_steps(vocabs, ctx, n=16, seed=0):
    rng = np.random.RandomState(seed)
    acts_lists = _random_action_lists(rng, n)
    steps = []
    for i, acts in enumerate(acts_lists):
        obs = encode_context(ctx, vocabs, acts)
        idx = int(rng.randint(0, len(acts)))
        steps.append(RolloutStep(obs=obs, action_index=idx, old_logprob=0.0,
                                 old_value=0.5, reward=0.0, done=(i == n - 1)))
    return steps


# ── P3.1 GAE ──────────────────────────────────────────────────────────
def test_gae_hand_computed():
    # 3 steps, win at terminal, hand-computed (gamma=0.99, lam=0.95)
    r = [0.0, 0.0, 1.0]
    v = [0.3, 0.5, 0.4]
    adv, ret = gae(r, v, gamma=0.99, lam=0.95)
    assert adv.shape == (3,) and ret.shape == (3,)
    # terminal return == the terminal reward (no continuation) exactly
    assert ret[2] == pytest.approx(1.0, abs=1e-6)
    # hand-computed (gamma*lam = 0.9405):
    #   adv[2]=0.6 ; adv[1]=(-0.104)+0.9405*0.6=0.4603 ; adv[0]=0.195+0.9405*0.4603=0.627912
    assert adv[1] == pytest.approx(0.4603, abs=1e-4)
    assert adv[0] == pytest.approx(0.627912, abs=1e-4)
    assert ret[1] == pytest.approx(0.9603, abs=1e-4)
    assert ret[0] == pytest.approx(0.927912, abs=1e-4)


def test_gae_perfect_value_zero_advantage():
    # if V(s_t) already equals the true discounted terminal return, advantages ~ 0
    gamma = 0.99
    T = 5
    terminal = 1.0
    values = [terminal * gamma ** (T - 1 - t) for t in range(T)]  # true value at each t
    rewards = [0.0] * (T - 1) + [terminal]
    adv, ret = gae(rewards, values, gamma=gamma, lam=0.9)
    assert np.allclose(adv, 0.0, atol=1e-6), f"advantages not ~0: {adv}"
    assert np.allclose(ret, values, atol=1e-5)


def test_gae_loss_terminal_return_is_minus_one():
    adv, ret = gae([0.0, 0.0, -1.0], [0.6, 0.4, 0.5], gamma=0.99, lam=0.95)
    assert ret[2] == pytest.approx(-1.0, abs=1e-6)
    assert np.isfinite(adv).all() and np.isfinite(ret).all()


def test_gae_rejects_bad_shapes():
    with pytest.raises(ValueError):
        gae([0.0, 1.0], [0.5])
    with pytest.raises(ValueError):
        gae([], [])


# ── P3.2 buffer + batching ─────────────────────────────────────────────
def test_buffer_append_len_clear_slice():
    buf = RolloutBuffer()
    steps = make_steps(make_vocabs(make_ctx()), make_ctx(), n=5)
    buf.extend(steps)
    assert len(buf) == 5
    assert len(buf.slice(1, 3)) == 2
    buf.append(steps[0])
    assert len(buf) == 6
    buf.clear()
    assert len(buf) == 0


def test_steps_to_batch_shapes():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    steps = make_steps(v, ctx, n=7)
    batch = steps_to_batch(steps, "cpu")
    import torch
    B = 7
    for fld in LAYOUT:
        assert batch[fld.name].shape == (B, 1) + tuple(fld.shape), fld.name
    assert batch["battle_mask"].shape == (B, 1)
    assert batch["action_index"].shape == (B,)
    assert batch["old_logprob"].shape == (B,)
    assert batch["old_value"].shape == (B,)
    # every action_index is legal for its row's mask
    for i in range(B):
        assert batch["action_mask"][i, 0, batch["action_index"][i]] == 1.0


def test_steps_to_batch_rejects_empty():
    with pytest.raises(ValueError):
        steps_to_batch([], "cpu")


# ── P3.3 PPO update: the disaster guard (policy moves, sign-correct) ──
@pytest.fixture(scope="module")
def ctx_vocabs():
    ctx = make_ctx()
    return ctx, make_vocabs(ctx)


def _build(vocabs, ctx, n=24, seed=3):
    steps = make_steps(vocabs, ctx, n=n, seed=seed)
    return steps_to_batch(steps, "cpu")


def test_ppo_positive_advantage_raises_chosen_logprob(ctx_vocabs):
    import torch
    torch.manual_seed(2026)  # fixed random-init so the margin is deterministic
    ctx, vocabs = ctx_vocabs
    net = make_random_net(vocabs, "cpu")
    batch = _build(vocabs, ctx, seed=11)
    with torch.no_grad():
        batch["old_logprob"] = chosen_logprobs(net, batch).clone()
    tr = PPOTrainer(net, lr=1e-3, ent_coef=0.0, vf_coef=0.0)
    before = float(chosen_logprobs(net, batch).mean().item())
    tr.update(batch, advantages=np.ones(len(batch["action_index"])),
              returns=np.zeros(len(batch["action_index"])), epochs=30)
    after = float(chosen_logprobs(net, batch).mean().item())
    assert after > before + 0.02, f"positive advantage did not raise chosen logprob ({before}->{after})"


def test_ppo_negative_advantage_lowers_chosen_logprob(ctx_vocabs):
    import torch
    torch.manual_seed(2026)  # fixed random-init so the margin is deterministic
    ctx, vocabs = ctx_vocabs
    net = make_random_net(vocabs, "cpu")
    batch = _build(vocabs, ctx, seed=12)
    with torch.no_grad():
        batch["old_logprob"] = chosen_logprobs(net, batch).clone()
    tr = PPOTrainer(net, lr=1e-3, ent_coef=0.0, vf_coef=0.0)
    before = float(chosen_logprobs(net, batch).mean().item())
    tr.update(batch, advantages=-np.ones(len(batch["action_index"])),
              returns=np.zeros(len(batch["action_index"])), epochs=30)
    after = float(chosen_logprobs(net, batch).mean().item())
    assert after < before - 0.02, f"negative advantage did not lower chosen logprob ({before}->{after})"


def test_ppo_update_finite_and_changes_weights(ctx_vocabs):
    ctx, vocabs = ctx_vocabs
    net = make_random_net(vocabs, "cpu")
    batch = _build(vocabs, ctx, n=20, seed=13)
    import torch
    n = len(batch["action_index"])
    with torch.no_grad():
        batch["old_logprob"] = chosen_logprobs(net, batch).clone()
        batch["old_value"] = net(batch)["value"][:, 0, 0].clone()
    # a GAE'd (advantage, return) pair from a fake terminal-reward trajectory.
    # value head = P(win) in (0,1) -> map to reward space (2V-1) before GAE.
    values = 2.0 * batch["old_value"].numpy() - 1.0
    rewards = np.zeros(n); rewards[-1] = 1.0
    adv, ret = gae(rewards, values, gamma=0.99, lam=0.95)

    before_params = [p.detach().clone() for p in net.parameters()]
    tr = PPOTrainer(net, lr=3e-4)
    metrics = tr.update(batch, advantages=adv, returns=ret, epochs=5)
    after_params = [p.detach().clone() for p in net.parameters()]

    # finite metrics, no NaN
    for k, v in metrics.items():
        assert np.isfinite(v), f"non-finite metric {k}={v}"
    # weights actually changed (gradients flowed)
    changed = any(not torch.allclose(b, a, atol=0.0) for b, a in zip(before_params, after_params))
    assert changed, "PPO update did not change any weights (no learning signal)"
    # the net is back in eval mode after update
    assert not net.training
