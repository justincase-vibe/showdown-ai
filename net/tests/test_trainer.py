"""P4.1 (self-play trainer core) gate tests — OFFLINE (no fork, CPU).

These pin the reusable training core the P4 GUI drives, without needing a live
battle: the TrainConfig (GUI-serializable), the 2V-1 value mapping (the P3
bug-hunt fix, must survive in the core), the checkpoint save/resume roundtrip,
and the trainer wiring. The LIVE end-to-end (collect -> GAE -> update ->
checkpoint over real battles) is the separate CPU gate ``scratch/p41_gate.py``.
"""
from __future__ import annotations

import pytest
import torch

from net.tests.test_s41 import make_ctx, make_vocabs
from net.agent import make_random_net
from net.ppo import RolloutStep
from net.trainer import TrainConfig, SelfPlayTrainer


# ── TrainConfig ─────────────────────────────────────────────────────────
def test_trainconfig_defaults_and_roundtrip():
    c = TrainConfig()
    # P4 defaults: 50 battles (the user's starting scale), on-policy 1-2 epochs, CPU.
    assert c.battles_per_round == 50
    assert c.epochs == 2
    assert c.device == "cpu"
    d = c.to_dict()
    c2 = TrainConfig.from_dict(d)
    assert c2.to_dict() == d
    # unknown (future) keys are ignored -> an old GUI JSON can't break a newer
    # dataclass and vice-versa.
    c3 = TrainConfig.from_dict({**d, "bogus_future_key": 123})
    assert c3.to_dict() == d


# ── the 2V-1 value mapping (P3 bug-hunt fix, must hold in the core) ─────
def test_apply_gae_maps_value_to_reward_space():
    # 3 steps, all old_value=0.5 (P(win)=0.5 -> reward-space value 0.0), win at terminal.
    traj = [RolloutStep(obs={}, action_index=0, old_logprob=0.0, old_value=0.5,
                        reward=r, done=(i == 2)) for i, r in enumerate([0.0, 0.0, 1.0])]
    net = make_random_net(make_vocabs(make_ctx()), "cpu")
    tr = SelfPlayTrainer(net, TrainConfig(gamma=0.99, lam=0.95))
    tr.apply_gae(traj)
    # With the 2V-1 mapping the values are [0,0,0]; gae -> adv[2]=1.0 (terminal
    # return == the terminal reward, value 0). WITHOUT the mapping (raw 0.5)
    # adv[2] would be 0.5 — so adv[2]==1.0 proves the mapping is applied.
    assert abs(traj[2].advantage - 1.0) < 1e-6, f"adv[2]={traj[2].advantage} (want 1.0 => 2V-1 mapping)"
    assert abs(traj[2].ret - 1.0) < 1e-6
    # hand-check the mapped gae (values=[0,0,0], r=[0,0,1], gamma.99 lam.95):
    #   adv[2]=1.0 ; adv[1]=0.99*0.95=0.9405 ; adv[0]=0.9405^2=0.88454025
    assert abs(traj[1].advantage - 0.9405) < 1e-6
    assert abs(traj[0].advantage - 0.9405 ** 2) < 1e-6


# ── trainer wiring ────────────────────────────────────────────────────────
def test_trainer_construct():
    ctx = make_ctx(); v = make_vocabs(ctx)
    net = make_random_net(v, "cpu")
    cfg = TrainConfig(seed=42)
    tr = SelfPlayTrainer(net, cfg)
    assert tr.cfg is cfg and tr.net is net and tr.round == 0
    assert tr.agent is not None and tr.ppo is not None and len(tr.buffer) == 0


def test_update_requires_min_transitions():
    ctx = make_ctx(); v = make_vocabs(ctx)
    net = make_random_net(v, "cpu")
    tr = SelfPlayTrainer(net, TrainConfig())
    with pytest.raises(RuntimeError):
        tr.update()


# ── checkpoint save / resume roundtrip ────────────────────────────────────
def test_checkpoint_save_load_roundtrip(tmp_path):
    ctx = make_ctx(); v = make_vocabs(ctx)
    torch.manual_seed(1)
    netA = make_random_net(v, "cpu")
    cfg = TrainConfig(battles_per_round=1, checkpoint_dir=str(tmp_path), save_every=1)
    trA = SelfPlayTrainer(netA, cfg)
    trA.round = 7
    trA._last = {"round": 7, "loss": 0.5}
    path = trA.save_checkpoint()
    assert path.endswith("selfplay_round7.pt")

    # a FRESH net with different weights; load A's checkpoint into it.
    torch.manual_seed(999)
    netB = make_random_net(v, "cpu")
    # they must DIFFER before the load (else the roundtrip is vacuous)...
    assert not all(torch.allclose(pa, pb) for pa, pb in zip(netA.parameters(), netB.parameters()))
    trB = SelfPlayTrainer(netB, TrainConfig())
    trB.load_checkpoint(path)
    # ...and MATCH after it.
    for pa, pb in zip(netA.parameters(), netB.parameters()):
        assert torch.allclose(pa, pb)
    assert trB.round == 7
    assert trB._last.get("round") == 7


# ── D80: the graceful-stop signal (stop_check) ──────────────────────────────
def test_rollout_stop_check_halts_between_battles(monkeypatch):
    """D80: the graceful-stop signal must be honored BETWEEN battles — the
    in-flight battle finishes, the rest of the round's battles are skipped,
    and ``stopped_early`` is flagged. Without this, a stop with
    battles_per_round=50 would run all 50 battles before noticing (or force-
    kill mid-battle after the grace period)."""
    import net.trainer as nt

    ctx = make_ctx(); v = make_vocabs(ctx)
    net = make_random_net(v, "cpu")
    tr = SelfPlayTrainer(net, TrainConfig(battles_per_round=5))

    calls = []

    def fake_collect(agent, fmt, seed, max_steps=500, **kw):
        calls.append(seed)
        return {"crashed": True, "error": "fake", "traj": [], "n_illegal": 0}
    monkeypatch.setattr(nt, "play_and_collect", fake_collect)

    # stop fires once 2 battles have been attempted
    summary = tr.rollout(stop_check=lambda: len(calls) >= 2)
    assert summary["stopped_early"] is True
    assert len(calls) == 2, f"expected halt after 2 battles, got {len(calls)}"
    assert summary["crashed"] == 2 and summary["battles"] == 0

    # no stop_check -> the full 5
    summary2 = tr.rollout()
    assert summary2["stopped_early"] is False
    assert len(calls) == 7

    # stop BEFORE the round: zero battles, and train_round stays atomic
    # (nothing buffered -> the update is skipped, no crash, round recorded)
    out = tr.train_round(stop_check=lambda: True)
    assert out["stopped_early"] is True and out["battles"] == 0
    assert out.get("skipped") is True  # 0 transitions -> no partial update
    assert len(calls) == 7  # the stopped round attempted nothing
