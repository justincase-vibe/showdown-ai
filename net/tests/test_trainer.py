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


# ── per-battle watchdog (2026-11-22: one hung battle must never stall a run) ──
def test_watchdog_kills_hung_battle(monkeypatch):
    """A battle stuck inside play_and_collect (a network wait that never
    resolves) must be KILLED after cfg.battle_timeout and reported as a
    crashed+timed_out battle — NOT hang the run. This is the exact bug that
    stalled the 2026-11-22 160-battle run for 14+ min."""
    import time
    import net.trainer as nt

    ctx = make_ctx(); v = make_vocabs(ctx)
    net = make_random_net(v, "cpu")
    tr = SelfPlayTrainer(net, TrainConfig(battle_timeout=1))  # 1s watchdog

    def fake_hang(agent, fmt, seed, max_steps=500, **kw):
        time.sleep(4)  # longer than the 1s watchdog -> must be killed
        return {"crashed": False, "traj": [], "steps": 1, "n_illegal": 0}
    monkeypatch.setattr(nt, "play_and_collect", fake_hang)

    t0 = time.time()
    r = tr._play_with_watchdog(seed=123)
    elapsed = time.time() - t0
    assert r.get("timed_out") is True and r.get("crashed") is True
    assert elapsed < 3.5, f"watchdog did not kill the hang (elapsed {elapsed:.1f}s > ~1s timeout)"


def test_watchdog_allows_fast_battle(monkeypatch):
    """A battle that finishes INSIDE the timeout must pass through UNCHANGED
    (the watchdog only fires on a real hang) — no spurious kills."""
    import net.trainer as nt

    ctx = make_ctx(); v = make_vocabs(ctx)
    net = make_random_net(v, "cpu")
    tr = SelfPlayTrainer(net, TrainConfig(battle_timeout=5))

    def fake_fast(agent, fmt, seed, max_steps=500, **kw):
        return {"crashed": False, "traj": [], "steps": 3,
                "a1_won": True, "a2_won": False, "n_illegal": 0, "draw": False}
    monkeypatch.setattr(nt, "play_and_collect", fake_fast)

    r = tr._play_with_watchdog(seed=7)
    assert not r.get("timed_out") and r["crashed"] is False
    assert r["steps"] == 3 and r["a1_won"] is True


def test_watchdog_disabled_when_timeout_zero(monkeypatch):
    """battle_timeout=0 disables the watchdog (the old direct-call path) —
    the battle runs inline, no thread, and a hang would (as before) block."""
    import net.trainer as nt

    ctx = make_ctx(); v = make_vocabs(ctx)
    net = make_random_net(v, "cpu")
    tr = SelfPlayTrainer(net, TrainConfig(battle_timeout=0))

    saw = {}
    def fake_inline(agent, fmt, seed, max_steps=500, **kw):
        saw["called"] = True
        return {"crashed": False, "traj": [], "steps": 2, "n_illegal": 0}
    monkeypatch.setattr(nt, "play_and_collect", fake_inline)

    r = tr._play_with_watchdog(seed=1)
    assert saw.get("called") is True and r["steps"] == 2 and not r.get("timed_out")


def test_rollout_counts_watchdog_timeouts(monkeypatch):
    """A rollout where a battle hangs must still COMPLETE (the round doesn't
    stall) and record the timeout in n_timeouts — the run-level guarantee."""
    import time
    import net.trainer as nt

    ctx = make_ctx(); v = make_vocabs(ctx)
    net = make_random_net(v, "cpu")
    tr = SelfPlayTrainer(net, TrainConfig(battles_per_round=3, battle_timeout=1))

    state = {"i": 0}
    def fake(agent, fmt, seed, max_steps=500, **kw):
        state["i"] += 1
        if state["i"] == 2:
            time.sleep(4)  # battle 2 hangs -> watchdog kills it
            return {"crashed": False, "traj": [], "steps": 1,
                    "a1_won": True, "a2_won": False, "draw": False, "n_illegal": 0}
        return {"crashed": False, "traj": [], "steps": 1, "a1_won": True,
                "a2_won": False, "draw": False, "n_illegal": 0}
    monkeypatch.setattr(nt, "play_and_collect", fake)

    summary = tr.rollout()
    assert summary["n_timeouts"] == 1, f"n_timeouts={summary.get('n_timeouts')} (want 1)"
    assert summary["crashed"] == 1
    assert summary["stopped_early"] is False  # the round ran to the end, not stuck
