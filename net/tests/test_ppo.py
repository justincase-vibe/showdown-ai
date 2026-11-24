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
                     gae, normalize_advantages, steps_to_batch)
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
    """Disaster guard (fix-pack edition): with MIXED-sign advantages (a real
    buffer has wins AND losses), rows with positive advantage must go UP and
    rows with negative advantage DOWN. (A constant-advantage buffer normalizes
    to zero by design — no relative information — so the guard uses a
    realistic mixed vector and asserts PER GROUP.)"""
    import torch
    torch.manual_seed(2026)  # fixed random-init so the margin is deterministic
    ctx, vocabs = ctx_vocabs
    net = make_random_net(vocabs, "cpu")
    batch = _build(vocabs, ctx, seed=11)
    n = len(batch["action_index"])
    with torch.no_grad():
        batch["old_logprob"] = chosen_logprobs(net, batch).clone()
    adv = np.concatenate([np.ones(n // 2), -np.ones(n - n // 2)])
    tr = PPOTrainer(net, lr=1e-3, ent_coef=0.0, vf_coef=0.0, target_kl=None)
    lp0 = chosen_logprobs(net, batch)
    before_pos = float(lp0[:n // 2].mean())
    before_neg = float(lp0[n // 2:].mean())
    tr.update(batch, advantages=adv, returns=np.zeros(n), epochs=30)
    lp1 = chosen_logprobs(net, batch)
    after_pos = float(lp1[:n // 2].mean())
    after_neg = float(lp1[n // 2:].mean())
    assert after_pos > before_pos + 0.02, \
        f"positive advantage did not raise chosen logprob ({before_pos}->{after_pos})"
    assert after_neg < before_neg - 0.02, \
        f"negative advantage did not lower chosen logprob ({before_neg}->{after_neg})"


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


# ── PPO fix pack (2026-11-24): adv-normalization + minibatch + KL early stop ──
def test_normalize_advantages():
    """normalize_advantages: mean 0 / std 1; constant → zeros; empty OK."""
    rng = np.random.RandomState(0)
    a = rng.randn(200) * 5.0 + 3.0          # std ≈ 5, mean ≈ 3
    out = normalize_advantages(a)
    assert out.dtype == np.float32 and out.shape == a.shape
    assert float(out.mean()) == pytest.approx(0.0, abs=1e-5)
    assert float(out.std()) == pytest.approx(1.0, rel=1e-5)
    # a constant vector carries no RELATIVE information → zeros (textbook)
    assert np.allclose(normalize_advantages(np.full(10, 7.5)), 0.0)
    assert np.allclose(normalize_advantages(np.full(10, -2.0)), 0.0)
    # empty input is fine (nothing to normalize)
    e = normalize_advantages(np.array([], dtype=np.float64))
    assert e.size == 0 and e.dtype == np.float32
    # accepts a plain list
    o2 = normalize_advantages([1.0, 2.0, 3.0])
    assert float(o2.mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(o2.std()) == pytest.approx(1.0, rel=1e-6)


def test_update_minibatch_grad_steps(ctx_vocabs):
    """n=600, mb=256, epochs=2, target_kl=None → exactly 6 Adam steps, no stop.

    600/256 → 3 minibatches per epoch (256+256+88); 2 epochs → 6 grad steps.
    Pins the minibatch-loop arithmetic (the pre-fix code took ONE step/epoch —
    the "80 rounds = 160 gradient steps" finding).
    """
    import torch
    torch.manual_seed(7)
    ctx, vocabs = ctx_vocabs
    net = make_random_net(vocabs, "cpu")
    batch = _build(vocabs, ctx, n=600, seed=5)
    n = len(batch["action_index"])
    with torch.no_grad():
        batch["old_logprob"] = chosen_logprobs(net, batch).clone()
        batch["old_value"] = net(batch)["value"][:, 0, 0].clone()
    adv = np.random.RandomState(1).randn(n)
    tr = PPOTrainer(net, lr=3e-4, minibatch_size=256, target_kl=None)
    m = tr.update(batch, advantages=adv, returns=np.zeros(n), epochs=2)
    assert m["grad_steps"] == pytest.approx(6.0), m
    assert m["early_stopped"] == 0.0, m
    for k, v in m.items():
        assert np.isfinite(v), f"non-finite metric {k}={v}"


def test_update_kl_early_stop(ctx_vocabs):
    """target_kl=0.005 at a high lr must stop the update EARLY:
    early_stopped=1 and strictly fewer than the full 10 epochs × 3
    minibatches = 30 steps."""
    import torch
    torch.manual_seed(21)
    ctx, vocabs = ctx_vocabs
    net = make_random_net(vocabs, "cpu")
    batch = _build(vocabs, ctx, n=96, seed=17)
    n = 96
    with torch.no_grad():
        batch["old_logprob"] = chosen_logprobs(net, batch).clone()
        batch["old_value"] = net(batch)["value"][:, 0, 0].clone()
    adv = np.random.RandomState(9).randn(n)
    tr = PPOTrainer(net, lr=3e-3, minibatch_size=32, target_kl=0.005)
    m = tr.update(batch, advantages=adv, returns=np.zeros(n), epochs=10)
    assert m["early_stopped"] == 1.0, f"KL early-stop did not fire: {m}"
    assert 1.0 <= m["grad_steps"] < 30.0, m


def test_update_normalizes_advantages(ctx_vocabs):
    """update() must normalize advantages ONCE before the minibatch loop:
    a std≈5 input buffer reaches the loss as std≈1 / mean≈0, and a constant
    input buffer reaches it as all-zeros (no relative information)."""
    import torch
    torch.manual_seed(31)
    ctx, vocabs = ctx_vocabs
    net = make_random_net(vocabs, "cpu")
    batch = _build(vocabs, ctx, n=64, seed=23)
    n = 64
    with torch.no_grad():
        batch["old_logprob"] = chosen_logprobs(net, batch).clone()
    tr = PPOTrainer(net, lr=1e-5, minibatch_size=64, target_kl=None,
                    vf_coef=0.0, ent_coef=0.0)
    captured: list = []
    orig_loss = tr._loss
    def spy(b, adv, ret):
        captured.append(adv.detach().clone())
        return orig_loss(b, adv, ret)
    tr._loss = spy  # instance-attr shadow — update() calls self._loss(...)

    rng = np.random.RandomState(4)
    adv = rng.randn(n) * 5.0 + 2.0        # std≈5, mean≈2 going IN
    tr.update(batch, advantages=adv, returns=np.zeros(n), epochs=1)
    seen = torch.cat([c for c in captured if c.numel()])
    assert seen.numel() == n
    mu = float(seen.mean())
    # POPULATION std, computed explicitly (torch's whole-tensor .std() applies
    # Bessel's correction by default in recent builds — √(n/(n-1)) off).
    pop = float(((seen - mu) ** 2).mean().sqrt())
    assert mu == pytest.approx(0.0, abs=1e-4)
    assert pop == pytest.approx(1.0, rel=1e-4)

    captured.clear()
    tr.update(batch, advantages=np.full(n, -2.5), returns=np.zeros(n), epochs=1)
    seen2 = torch.cat([c for c in captured if c.numel()])
    assert seen2.numel() == n
    assert torch.allclose(seen2, torch.zeros_like(seen2), atol=1e-6)


# ── D4: GAE with draw reward -1.0 stays sane (2026-11-23 ruling) ──────────
def test_gae_draw_terminal_minus_one():
    """A draw terminal reward of -1.0 (2026-11-23 ruling) must produce
    finite, sane advantages and returns. The value head is P(win) in (0,1);
    the 2V-1 map gives a reward-space value in (-1, +1). A terminal reward
    of -1.0 is within range and must not cause NaN/inf."""
    r = [0.0, 0.0, -1.0]  # draw at terminal
    v = [0.5, 0.4, 0.3]   # P(win) values (all < 0.5 = losing side)
    adv, ret = gae(r, v, gamma=0.99, lam=0.95)
    assert adv.shape == (3,) and ret.shape == (3,)
    # terminal return == terminal reward (no continuation) exactly
    assert ret[2] == pytest.approx(-1.0, abs=1e-6)
    # all values finite
    for a in adv:
        assert np.isfinite(a)
    for rv in ret:
        assert np.isfinite(rv)
    # terminal advantage = reward - value (no bootstrap, nonterm=0)
    # delta = r[2] - v[2] = -1.0 - 0.3 = -1.3
    assert adv[2] == pytest.approx(-1.3, abs=1e-6)
    # terminal return = adv + v = -1.3 + 0.3 = -1.0 (== terminal reward)
    assert ret[2] == pytest.approx(-1.0, abs=1e-6)


def test_gae_draw_vs_win_symmetry():
    """A draw (-1.0) and a loss (-1.0) produce IDENTICAL GAE outputs given the
    same value estimates. This is by design: the agent cannot distinguish
    'stalled' from 'lost' in the reward signal — it must learn from the
    trajectory, not the terminal label."""
    r_draw = [0.0, -1.0]   # draw
    r_loss = [0.0, -1.0]   # loss (same reward magnitude)
    v = [0.5, 0.4]
    adv_d, ret_d = gae(r_draw, v, gamma=0.99, lam=0.95)
    adv_l, ret_l = gae(r_loss, v, gamma=0.99, lam=0.95)
    np.testing.assert_allclose(adv_d, adv_l, atol=1e-12)
    np.testing.assert_allclose(ret_d, ret_l, atol=1e-12)
