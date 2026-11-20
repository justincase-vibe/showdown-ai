"""League engine tests — FAST (no net, no torch, no node servers).

The pool is a fake (same surface SelfPlayRun uses), and the driver factory is
INJECTED (``driver_factory=``) with scripted results — so the engine's real
logic is exercised hermetically: planning, roster->pairing mapping, ELO
wiring (positional winner mapping, draws, errored pairings), and the
SelfPlayRun delegation (summary shape, stop, double-start).

The real checkpoint-vs-checkpoint path is ``league.driver`` (tested in
test_driver.py) + the live league run (manual / CI with a server pool).

Run:  python -m pytest league/tests/test_engine.py -q
"""
from __future__ import annotations

import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.engine import LeagueConfig, LeagueRun
from league.registry import BotRegistry, ELO_INITIAL


# ── fakes (same surfaces as control.tests.test_selfplay) ────────────────────
class FakePool:
    def __init__(self, slots: int = 1, healthy: bool = True):
        self._sem = threading.BoundedSemaphore(slots)
        self.healthy = healthy

    def start(self, wait_healthy: bool = True):
        return {8000: self.healthy}

    def shutdown(self):
        pass

    @contextmanager
    def battle_slot(self):
        self._sem.acquire()
        try:
            yield "server-cfg"
        finally:
            self._sem.release()


def make_fake_factory(results_by_idx: dict, fail_idx: int = -1):
    """An ``idx -> battle_driver`` factory with scripted results.

    ``results_by_idx[idx] = (a1_won, a2_won)``; ``fail_idx`` raises instead
    (an infra error — must get NO ELO update)."""
    def factory(idx: int):
        def driver(server_cfg, fmt="gen9ou", seed=0, vocabs=None,
                   max_steps=200, on_finish=None):
            if idx == fail_idx:
                raise RuntimeError("pool exploded")
            a1, a2 = results_by_idx[idx]
            return a1, a2, 3
        return driver
    return factory


def _cfg(**kw) -> LeagueConfig:
    base = dict(save_replays=False, n_servers=1, per_server=1, n_workers=1)
    base.update(kw)
    return LeagueConfig(**base)


def _roster(tmp_path, *names):
    reg = BotRegistry(path=tmp_path / "bots.json")
    for i, n in enumerate(names):
        reg.add(n, f"ckpt{i}.pt")
    return reg


# ── planning ────────────────────────────────────────────────────────────────
def test_plan_round_robin_covers_every_pair_once(tmp_path):
    reg = _roster(tmp_path, "a", "b", "c", "d")
    run = LeagueRun(_cfg(), registry=reg)
    pairs = run.plan()
    assert len(pairs) == 6
    assert sorted(tuple(sorted(p)) for p in pairs) == sorted(
        {("a", "b"), ("a", "c"), ("a", "d"), ("b", "c"), ("b", "d"), ("c", "d")})


def test_plan_best_of_repeats_each_pairing(tmp_path):
    reg = _roster(tmp_path, "a", "b")
    run = LeagueRun(_cfg(best_of=3), registry=reg)
    assert run.plan() == [("a", "b")] * 3


def test_plan_elos_weighted_pairs_top_with_bottom(tmp_path):
    reg = _roster(tmp_path, "weak", "mid1", "mid2", "strong")
    reg.get("strong").elo = 1700
    reg.get("mid2").elo = 1550
    reg.get("mid1").elo = 1450
    reg.get("weak").elo = 1300
    run = LeagueRun(_cfg(pairing="elo_weighted"), registry=reg)
    assert run.plan() == [("strong", "weak"), ("mid2", "mid1")]


def test_plan_rejects_single_bot(tmp_path):
    reg = _roster(tmp_path, "only")
    with pytest.raises(ValueError, match=">= 2 bots"):
        LeagueRun(_cfg(), registry=reg).plan()


def test_plan_rejects_unknown_strategy(tmp_path):
    reg = _roster(tmp_path, "a", "b")
    with pytest.raises(ValueError, match="unknown pairing"):
        LeagueRun(_cfg(pairing="swiss"), registry=reg).plan()


# ── the full run (delegated to SelfPlayRun with a fake factory) ────────────
def test_full_run_wires_elo(tmp_path):
    reg = _roster(tmp_path, "b1", "b2", "b3")
    # round_robin (names sorted): (b1,b2) (b1,b3) (b2,b3)
    factory = make_fake_factory({0: (True, False),    # b1 beats b2
                                 1: (False, True),     # b3 beats b1
                                 2: (False, False)})   # b2/b3 draw
    run = LeagueRun(_cfg(), registry=reg, pool=FakePool(), driver_factory=lambda: factory)
    run.start()
    s = run.wait()
    run.shutdown()

    assert s["state"] == "completed"
    assert s["pairings_planned"] == 3
    assert s["pairings_played"] == 3
    assert s["errors"] == 0
    # W/L/D bookkeeping (positional winner mapping)
    assert (reg.get("b1").wins, reg.get("b1").losses, reg.get("b1").draws) == (1, 1, 0)
    assert (reg.get("b2").wins, reg.get("b2").losses, reg.get("b2").draws) == (0, 1, 1)
    assert (reg.get("b3").wins, reg.get("b3").losses, reg.get("b3").draws) == (1, 0, 1)
    # ELO is zero-sum
    assert sum(reg.get(n).elo for n in ("b1", "b2", "b3")) == 3 * ELO_INITIAL
    # standings order: b3 (1W 1D) > b1 (1W 1L) > b2 (1L 1D)
    assert [row["name"] for row in s["standings"]] == ["b3", "b1", "b2"]
    # the per-pairing result rows (for the GUI / logs)
    rows = run.results()
    assert [r["winner"] for r in rows] == ["b1", "b3", None]


def test_errored_pairing_gets_no_elo(tmp_path):
    reg = _roster(tmp_path, "a", "b")
    factory = make_fake_factory({}, fail_idx=0)
    run = LeagueRun(_cfg(), registry=reg, pool=FakePool(), driver_factory=lambda: factory)
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["state"] == "completed"
    assert s["errors"] == 1
    assert s["pairings_played"] == 0
    # no silent draw: the roster is UNTOUCHED
    assert reg.get("a").elo == ELO_INITIAL
    assert reg.get("b").elo == ELO_INITIAL
    assert reg.get("a").draws == 0 and reg.get("b").draws == 0
    assert reg.get("a").wins == 0 and reg.get("a").losses == 0


def test_both_flagged_is_a_draw(tmp_path):
    reg = _roster(tmp_path, "a", "b")
    factory = make_fake_factory({0: (True, True)})   # weird end state -> draw
    run = LeagueRun(_cfg(), registry=reg, pool=FakePool(), driver_factory=lambda: factory)
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["state"] == "completed"
    assert reg.get("a").draws == 1 and reg.get("b").draws == 1
    assert reg.get("a").elo == ELO_INITIAL and reg.get("b").elo == ELO_INITIAL


# ── lifecycle + config ──────────────────────────────────────────────────────
def test_double_start_rejected(tmp_path):
    reg = _roster(tmp_path, "a", "b")
    run = LeagueRun(_cfg(), registry=reg, pool=FakePool(),
                    driver_factory=lambda: make_fake_factory({0: (True, False)}))
    run.start()
    try:
        with pytest.raises(RuntimeError, match="already started"):
            run.start()
    finally:
        run.wait()
        run.shutdown()


def test_wait_before_start_raises(tmp_path):
    with pytest.raises(RuntimeError, match="not started"):
        LeagueRun(_cfg(), registry=_roster(tmp_path, "a", "b")).wait()


# ── learning mode (D103 step 2: experience collection into bot buffers) ──
# The engine's OWN driver factory is exercised (the learning path); the
# battle layer (``league.driver.play_agent_battle_on``) is monkeypatched to a
# scripted fake that drives the REAL capture seam with REAL obs + decisions
# (a real net over the D92 fake battle) — no server, no pool, no GPU.

def _learning_roster(tmp_path, *names, seed: int = 11):
    """A roster of two+ bots on real (random-init) checkpoints, BC format.

    Production vocabs (``build_vocabs``) — the SAME the engine's
    ``_load_bots`` uses: a checkpoint must load through the ONE load path
    with the production vocab, or the size check rightly fails loud."""
    import torch
    from net.agent import make_random_net
    from net.vocab_db import build_vocabs
    v = build_vocabs()
    reg = BotRegistry(path=tmp_path / "bots.json")
    for i, name in enumerate(names):
        p = tmp_path / f"{name}.pt"
        net = make_random_net(v, device="cpu", seed=seed + i)
        torch.save(net.state_for_save(), str(p))
        reg.add(name, str(p))
    return reg


def _fake_play_factory(result, partial_before_error: bool = False):
    """A stand-in for ``play_agent_battle_on`` that fires the REAL capture
    seam: one real decision per side (real net, real build_obs, D92 battle),
    then the scripted result (or a scripted infra error).

    Production vocabs too — the obs must match the bot's net (built with the
    production vocab), exactly as the live path does."""
    from net.env import build_action_list, build_obs, my_side
    from net.tests.test_env_obs import _FakeBattle, _FakeMon
    from net.vocab_db import build_vocabs

    def fake_play(server_cfg, **kw):
        v = build_vocabs()
        b = _FakeBattle(
            active=_FakeMon("garchomp", moves=["outrage", "earthquake"]),
            avail_moves=["outrage", "earthquake"], tera=True)
        lines = [ln for ln in b._build_replay_log().split("\n") if ln.strip()]
        side = my_side(b, lines)
        obs = build_obs(b, "gen9ou", v, side)
        acts = build_action_list(b)
        for cap, ag in ((kw["capture_a"], kw["agent_a"]),
                        (kw["capture_b"], kw["agent_b"])):
            if cap is not None:
                cap(obs, ag.act(obs, acts, deterministic=True))
        if partial_before_error:
            raise RuntimeError("pool exploded")
        return result
    return fake_play


def _fake_play_factory_n(result, n_decisions: int = 3,
                         partial_before_error: bool = False):
    """Like ``_fake_play_factory`` but fires ``n_decisions`` real decisions per
    side — enough (>=2) to drive the D103 step-3 PPO update path (the default
    1-decision fake only exercises the <2 skip)."""
    from net.env import build_action_list, build_obs, my_side
    from net.tests.test_env_obs import _FakeBattle, _FakeMon
    from net.vocab_db import build_vocabs

    def fake_play(server_cfg, **kw):
        v = build_vocabs()
        b = _FakeBattle(
            active=_FakeMon("garchomp", moves=["outrage", "earthquake"]),
            avail_moves=["outrage", "earthquake"], tera=True)
        lines = [ln for ln in b._build_replay_log().split("\n") if ln.strip()]
        side = my_side(b, lines)
        obs = build_obs(b, "gen9ou", v, side)
        acts = build_action_list(b)
        for cap, ag in ((kw["capture_a"], kw["agent_a"]),
                        (kw["capture_b"], kw["agent_b"])):
            if cap is not None:
                for _ in range(n_decisions):
                    cap(obs, ag.act(obs, acts, deterministic=True))
        if partial_before_error:
            raise RuntimeError("pool exploded")
        return result
    return fake_play


def test_learning_run_buffers_experience_with_terminal_rewards(tmp_path,
                                                                monkeypatch):
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory((True, False, 2)))   # alpha wins
    reg = _learning_roster(tmp_path, "alpha", "beta")
    run = LeagueRun(_cfg(learning=True), registry=reg, pool=FakePool())
    run.start()          # the engine's OWN factory (learning path) runs now
    s = run.wait()
    run.shutdown()

    assert s["state"] == "completed"
    assert s["errors"] == 0
    # ELO still wired (the learning mode does not change the standings math)
    assert reg.get("alpha").wins == 1 and reg.get("beta").losses == 1

    # BOTH bots buffered real decisions
    bots = run._bots
    a_steps = list(bots["alpha"].buffer.steps)
    b_steps = list(bots["beta"].buffer.steps)
    assert a_steps and b_steps
    # sparse-terminal reward, each from its OWN POV
    assert a_steps[-1].reward == 1.0 and a_steps[-1].done is True
    assert all(st.reward == 0.0 and st.done is False for st in a_steps[:-1])
    assert b_steps[-1].reward == -1.0 and b_steps[-1].done is True
    assert all(st.reward == 0.0 for st in b_steps[:-1])
    # the decision numbers are the REAL policy's (finite, sane ranges)
    for st in a_steps:
        assert st.old_logprob <= 0.0
        assert 0.0 <= st.old_value <= 1.0
        assert st.action_index >= 0
        assert len(st.obs) > 0
    # the summary reports what was collected (the GUI's checkpoint signal)
    assert s["steps_collected_by_bot"] == {
        "alpha": len(a_steps), "beta": len(b_steps)}


def test_learning_run_draw_buffers_zero_terminal(tmp_path, monkeypatch):
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory((False, False, 2)))   # draw
    reg = _learning_roster(tmp_path, "a", "b")
    run = LeagueRun(_cfg(learning=True), registry=reg, pool=FakePool())
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["errors"] == 0
    assert reg.get("a").draws == 1
    for name in ("a", "b"):
        steps = list(run._bots[name].buffer.steps)
        assert steps and steps[-1].reward == 0.0 and steps[-1].done is True


def test_learning_run_discards_partial_trajectory_on_error(tmp_path,
                                                            monkeypatch):
    """An errored battle has no decided outcome -> its PARTIAL trajectory is
    poison for PPO: it must be DISCARDED (no buffer, no ELO)."""
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory((True, False, 2),
                                           partial_before_error=True))
    reg = _learning_roster(tmp_path, "alpha", "beta")
    run = LeagueRun(_cfg(learning=True), registry=reg, pool=FakePool())
    run.start()
    s = run.wait()
    run.shutdown()

    assert s["state"] == "completed" and s["errors"] == 1
    # NO ELO update (the errored pairing rule)
    assert reg.get("alpha").elo == ELO_INITIAL and reg.get("beta").elo == ELO_INITIAL
    # NO buffered steps (the partial trajectory was discarded, not finalized)
    assert len(run._bots["alpha"].buffer) == 0
    assert len(run._bots["beta"].buffer) == 0


# ── D103 step 3: the round-end PPO update (>=2 -> update+save+drain) ──────
def test_learning_run_updates_saves_and_drains_when_enough_steps(tmp_path,
                                                                 monkeypatch):
    """A bot with >=2 buffered decisions runs ONE PPO update, saves its UPDATED
    weights to its own checkpoint, and drains the buffer (the round-end
    contract). The checkpoint bytes MUST change (the weights were learned)."""
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory_n((True, False, 2), n_decisions=3))
    reg = _learning_roster(tmp_path, "alpha", "beta")
    a_ckpt, b_ckpt = reg.get("alpha").ckpt_path, reg.get("beta").ckpt_path
    before_a, before_b = Path(a_ckpt).read_bytes(), Path(b_ckpt).read_bytes()
    run = LeagueRun(_cfg(learning=True), registry=reg, pool=FakePool())
    run.start()
    s = run.wait()
    run.shutdown()

    assert s["state"] == "completed" and s["errors"] == 0
    assert run._bots["alpha"].n_updates == 1 and run._bots["beta"].n_updates == 1
    assert len(run._bots["alpha"].buffer) == 0      # drained after the update
    assert len(run._bots["beta"].buffer) == 0
    assert Path(a_ckpt).read_bytes() != before_a      # weights changed + re-saved
    assert Path(b_ckpt).read_bytes() != before_b
    assert s["updates_by_bot"] == {"alpha": 1, "beta": 1}
    assert s["n_saved_by_bot"] == {"alpha": 1, "beta": 1}
    assert s["steps_collected_by_bot"] == {"alpha": 3, "beta": 3}
    assert "loss" in s["last_update_metrics_by_bot"]["alpha"]


def test_learning_run_skips_update_when_too_few_steps(tmp_path, monkeypatch):
    """A bot with <2 buffered decisions (the default 1-decision fake) is SKIPPED
    — no update, no save (checkpoint UNCHANGED) — but KEEPS its buffer to
    accumulate more next round (the <2-skip contract, league must not raise)."""
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory((True, False, 2)))   # 1 decision/side
    reg = _learning_roster(tmp_path, "alpha", "beta")
    a_ckpt = reg.get("alpha").ckpt_path
    before_a = Path(a_ckpt).read_bytes()
    run = LeagueRun(_cfg(learning=True), registry=reg, pool=FakePool())
    run.start()
    s = run.wait()
    run.shutdown()

    assert s["state"] == "completed" and s["errors"] == 0
    assert run._bots["alpha"].n_updates == 0 and run._bots["beta"].n_updates == 0
    assert len(run._bots["alpha"].buffer) == 1      # retained (kept for next round)
    assert Path(a_ckpt).read_bytes() == before_a     # NOT re-saved (no update)
    assert s["updates_by_bot"] == {"alpha": 0, "beta": 0}
    assert s["steps_collected_by_bot"] == {"alpha": 1, "beta": 1}


def test_learning_run_multi_round_updates_each_round(tmp_path, monkeypatch):
    """n_rounds=2 -> each bot plays + updates TWICE (fresh rollout per round over
    the PREVIOUS round's updated net — the self-improving loop). The shared
    pool is booted once (the FakePool is reused across both rounds)."""
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory_n((True, False, 2), n_decisions=3))
    reg = _learning_roster(tmp_path, "alpha", "beta")
    run = LeagueRun(_cfg(learning=True, n_rounds=2), registry=reg, pool=FakePool())
    run.start()
    s = run.wait()
    run.shutdown()

    assert s["state"] == "completed" and s["errors"] == 0
    assert s["rounds"] == 2 and s["rounds_planned"] == 2
    assert run._bots["alpha"].n_updates == 2 and run._bots["beta"].n_updates == 2
    assert s["updates_by_bot"] == {"alpha": 2, "beta": 2}
    assert s["n_saved_by_bot"] == {"alpha": 2, "beta": 2}
    assert s["steps_collected_by_bot"] == {"alpha": 6, "beta": 6}  # 3/round x 2


def test_eval_mode_run_has_no_learning_side_effects(tmp_path):
    """learning=False (the legacy eval-only run): an injected factory, no
    capture, no bot materialization, no summary key — the path that existed
    before step 2, unchanged."""
    reg = _roster(tmp_path, "a", "b")
    factory = make_fake_factory({0: (True, False)})
    run = LeagueRun(_cfg(learning=False), registry=reg, pool=FakePool(),
                    driver_factory=lambda: factory)
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["state"] == "completed"
    assert reg.get("a").wins == 1
    assert run._bots == {}
    assert "steps_collected_by_bot" not in s


def test_config_roundtrip():
    cfg = LeagueConfig(pairing="elo_weighted", best_of=2, greedy=False,
                       device="cuda", n_workers=4, max_consecutive_errors=7,
                       learning=False)
    same = LeagueConfig.from_dict(cfg.to_dict())
    assert same == cfg
    assert LeagueConfig().learning is True      # the league IS the learning loop


def test_config_from_dict_ignores_unknown_keys():
    cfg = LeagueConfig.from_dict({"pairing": "round_robin", "bogus": 1})
    assert cfg.pairing == "round_robin"
    assert not hasattr(cfg, "bogus")


# ── D103 step 4: cull + refresh (the population loop) ───────────────────────
# The fake play always lets pairing[0] win, so in a 4-bot round-robin the
# ELO order after one round is deterministic (b1 > b2 > b3 > b4) — the cull
# target is KNOWN, not luck.

def _cull_cfg(tmp_path, **kw) -> LeagueConfig:
    base = dict(learning=True, cull_every=1, cull_frac=0.25, min_roster=2,
                refresh_dir=str(tmp_path / "fresh"), save_replays=False,
                n_servers=1, per_server=1, n_workers=1)
    base.update(kw)
    return LeagueConfig(**base)


def test_cull_retires_weakest_and_mints_fresh(tmp_path, monkeypatch):
    """n_rounds=2, cull_every=1: after round 1 the weakest bot (b4) is
    retired and ONE fresh random bot takes its slot (roster size constant).
    The culled bot's .pt stays (artifact); the fresh bot's .pt is a NEW OWN
    file under refresh_dir; round 2 replans (b4 never fights again)."""
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory_n((True, False, 2), n_decisions=3))
    reg = _learning_roster(tmp_path, "b1", "b2", "b3", "b4")
    run = LeagueRun(_cull_cfg(tmp_path, n_rounds=2), registry=reg, pool=FakePool())
    run.start()
    s = run.wait()
    run.shutdown()

    assert s["state"] == "completed" and s["errors"] == 0
    # exactly ONE cull (after round 1; NOT after round 2 — the last round)
    events = s["cull_events"]
    assert len(events) == 1 and events[0]["round"] == 1
    assert events[0]["culled"] == ["b4"]
    added = events[0]["added"]
    assert len(added) == 1 and added[0].startswith("league_")
    # roster: b4 retired, fresh bot in, size constant
    names = reg.names()
    assert "b4" not in names and added[0] in names and len(names) == 4
    # the culled .pt is KEPT (artifact — deleting weights is a user action)
    assert (tmp_path / "b4.pt").is_file()
    # the fresh bot has its OWN file + meta sidecar under refresh_dir
    fresh = Path(tmp_path / "fresh" / (added[0] + ".pt"))
    assert fresh.is_file()
    assert Path(str(fresh) + ".meta.json").is_file()
    # live bots swapped too (the driver factory reads self._bots)
    assert "b4" not in run._bots and added[0] in run._bots
    # both original bots still updated in BOTH rounds (b4 only in round 1)
    assert s["updates_by_bot"]["b1"] == 2 and s["updates_by_bot"]["b2"] == 2
    assert s["updates_by_bot"][added[0]] == 1   # played round 2 only
    # round 2 replanned over the NEW roster (6 pairings, 12 total played)
    assert s["pairings_played"] == 12


def test_cull_skipped_on_the_last_round(tmp_path, monkeypatch):
    """n_rounds=1: the single round IS the last one -> no cull (minting bots
    that never fight would be pointless)."""
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory_n((True, False, 2), n_decisions=3))
    reg = _learning_roster(tmp_path, "b1", "b2", "b3", "b4")
    run = LeagueRun(_cull_cfg(tmp_path, n_rounds=1), registry=reg, pool=FakePool())
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["cull_events"] == []
    assert len(reg.names()) == 4 and "b4" in reg.names()
    assert not (tmp_path / "fresh").exists() or not list((tmp_path / "fresh").iterdir())


def test_cull_disabled_by_knob(tmp_path, monkeypatch):
    """cull_every=0: the loop is OFF — no cull even on a round boundary."""
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory_n((True, False, 2), n_decisions=3))
    reg = _learning_roster(tmp_path, "b1", "b2", "b3", "b4")
    run = LeagueRun(_cull_cfg(tmp_path, n_rounds=2, cull_every=0),
                    registry=reg, pool=FakePool())
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["cull_events"] == []
    assert sorted(reg.names()) == ["b1", "b2", "b3", "b4"]


def test_cull_floor_keeps_the_roster(tmp_path, monkeypatch):
    """min_roster=4 with a 4-bot roster: at the floor -> nothing may be
    retired (the floor is max(2, min_roster), the league's hard minimum)."""
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory_n((True, False, 2), n_decisions=3))
    reg = _learning_roster(tmp_path, "b1", "b2", "b3", "b4")
    run = LeagueRun(_cull_cfg(tmp_path, n_rounds=2, min_roster=4),
                    registry=reg, pool=FakePool())
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["cull_events"] == []
    assert sorted(reg.names()) == ["b1", "b2", "b3", "b4"]


def test_cull_every_k_not_every_round(tmp_path, monkeypatch):
    """cull_every=2 over 3 rounds: cull ONLY after round 2 (not round 1,
    not round 3 — the last round). 6 bots -> the single weakest (b6) is
    retired once; roster size stays 6."""
    import league.driver as drv
    monkeypatch.setattr(drv, "play_agent_battle_on",
                        _fake_play_factory_n((True, False, 2), n_decisions=3))
    reg = _learning_roster(tmp_path, "b1", "b2", "b3", "b4", "b5", "b6")
    run = LeagueRun(_cull_cfg(tmp_path, n_rounds=3, cull_every=2),
                    registry=reg, pool=FakePool())
    run.start()
    s = run.wait()
    run.shutdown()

    events = s["cull_events"]
    assert len(events) == 1 and events[0]["round"] == 2
    assert events[0]["culled"] == ["b6"]
    added = events[0]["added"]
    assert len(added) == 1
    assert "b6" not in reg.names() and added[0] in reg.names()
    assert len(reg.names()) == 6


def test_eval_mode_never_culls(tmp_path):
    """learning=False (the legacy eval-only run) must NOT mutate the roster —
    culling there would destroy the ranking artifacts."""
    reg = _roster(tmp_path, "a", "b", "c", "d")
    factory = make_fake_factory({i: (True, False) for i in range(6)})
    run = LeagueRun(_cfg(learning=False, n_rounds=2, cull_every=1),
                    registry=reg, pool=FakePool(), driver_factory=lambda: factory)
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["state"] == "completed"
    assert sorted(reg.names()) == ["a", "b", "c", "d"]
    assert "cull_events" not in s


# ── D103 step 4 hardening: the _after_round guard (white-box) ───────────────
#
# These three test the guard/surgery logic of ``_cull_and_refresh`` in ISOLATION
# (no SelfPlayRun workers): materialize the live bots, set the round state, and
# drive ``_after_round`` directly with the round-end state. With EMPTY buffers
# ``_update_round`` is a pure no-op (``n == 0 -> continue`` for every bot), so
# what is under test is exactly the cull guard — not the battle loop.
#
# The existing cull tests above get a deterministic ELO order by PLAYING a round
# (the fake always lets pairing[0] win => b1>b2>b3>b4). A direct ``_after_round``
# call has not played anything, so every bot still sits at ELO_INITIAL (1500);
# ``select_cull`` breaks ties by NAME, which would cull the alphabetically-first
# bot (b1), not b4. So each test sets DISTINCT ELOs up front to make the cull
# target known without a battle.

def _set_elos(reg, mapping) -> None:
    """Give the roster DISTINCT ELOs (a deterministic cull target) and persist
    them, without playing a round."""
    for name, elo in mapping.items():
        reg.get(name).elo = elo
    reg.save()


def test_cull_skipped_when_round_stopped(tmp_path):
    """The step-4 guard: the cull runs ONLY on a COMPLETED round.

    After a user stop (or a failure) the campaign is over — minting explorers
    there would mutate the roster for no reason. A STOPPED round still runs the
    PPO update (it saves whatever was learned) but must NOT cull; a COMPLETED
    round at the same boundary does cull."""
    reg = _learning_roster(tmp_path, "b1", "b2", "b3", "b4")
    _set_elos(reg, {"b1": 1600, "b2": 1550, "b3": 1500, "b4": 1400})  # b4 weakest

    run = LeagueRun(_cull_cfg(tmp_path, n_rounds=2), registry=reg, pool=FakePool())
    run._load_bots()              # materialize the live bots (NO SelfPlayRun workers)
    run._n_rounds = 2
    run._round_idx = 1            # "round 1 just ended" (1 % cull_every == 0, 1 < 2)

    # a STOPPED round: the PPO update runs (no-op, empty buffers) but NO cull
    run._after_round({"state": "stopped"})
    assert run._cull_events == []
    assert sorted(reg.names()) == ["b1", "b2", "b3", "b4"]

    # a COMPLETED round at the same boundary: the cull fires (b4 = weakest)
    run._after_round({"state": "completed"})
    assert len(run._cull_events) == 1
    assert run._cull_events[0]["culled"] == ["b4"]
    assert "b4" not in reg.names()
    run.shutdown()


def test_cull_mint_failure_keeps_roster(tmp_path, monkeypatch):
    """Mint-first atomicity: minting a fresh bot is done BEFORE retiring the
    culled. If minting fails (disk error) the roster is left UNCHANGED — no
    partial retirement. (The old retire-first order could have shrunk the roster
    below the floor, or even below 2, and the next plan() would raise.)

    Patching ``LearningBot.random`` (the mint) to raise does NOT touch the load
    path — the engine's ``_load_bots`` uses ``from_snapshot``, so the roster
    materializes normally and only the mint blows up."""
    import league.learner as learner

    def boom(*a, **k):
        raise OSError("disk full (simulated mint failure)")
    monkeypatch.setattr(learner.LearningBot, "random", boom)

    reg = _learning_roster(tmp_path, "b1", "b2", "b3", "b4")
    _set_elos(reg, {"b1": 1600, "b2": 1550, "b3": 1500, "b4": 1400})  # b4 weakest

    run = LeagueRun(_cull_cfg(tmp_path, n_rounds=2), registry=reg, pool=FakePool())
    run._load_bots()
    run._n_rounds = 2
    run._round_idx = 1

    with pytest.raises(OSError, match="disk full"):
        run._after_round({"state": "completed"})

    # mint-first => NOTHING was retired before the mint blew up: the roster is
    # UNCHANGED (all 4 bots, no fresh bot, b4 still present).
    assert sorted(reg.names()) == ["b1", "b2", "b3", "b4"]
    assert sorted(run._bots) == ["b1", "b2", "b3", "b4"]
    assert "b4" in run._bots
    # no cull was recorded (the failure aborted the surgery)
    assert run._cull_events == []
    run.shutdown()


def test_cull_name_collision_redraws(tmp_path, monkeypatch):
    """Name-shadow guard: a fresh bot's name must not SHADOW an existing roster
    name (a hand-registered bot could own a ``league_*`` name, and
    ``registry.add`` would silently OVERWRITE its entry — ELO/W-L lost).

    The roster has a bot literally named ``league_abcd`` (high ELO, so it is NOT
    culled). ``fresh_bot_name`` is scripted to collide with it on the first
    draw and return a clean ``league_zzzz`` on the second: the engine must skip
    the collision, mint ``league_zzzz``, and leave ``league_abcd``'s original
    checkpoint untouched."""
    import league.learner as learner

    reg = _learning_roster(tmp_path, "league_abcd", "c1", "c2", "c3")
    # league_abcd is the STRONGEST (survives); c3 the weakest (culled).
    _set_elos(reg, {"league_abcd": 2000, "c1": 1500, "c2": 1450, "c3": 1400})
    original_abcd_ckpt = reg.get("league_abcd").ckpt_path

    # scripted 2-draw fresh_bot_name: draw 1 COLLIDES with the roster name, draw 2 is clean
    draws = iter([Path(tmp_path) / "fresh" / "league_abcd.pt",
                  Path(tmp_path) / "fresh" / "league_zzzz.pt"])
    monkeypatch.setattr(learner, "fresh_bot_name", lambda *a, **k: next(draws))

    run = LeagueRun(_cull_cfg(tmp_path, n_rounds=2), registry=reg, pool=FakePool())
    run._load_bots()
    run._n_rounds = 2
    run._round_idx = 1

    run._after_round({"state": "completed"})

    events = run._cull_events
    assert len(events) == 1 and events[0]["culled"] == ["c3"]
    added = events[0]["added"]
    # the collision was skipped; the fresh bot took the CLEAN second draw
    assert added == ["league_zzzz"]
    assert "league_zzzz" in reg.names() and "league_zzzz" in run._bots
    # the pre-existing league_abcd SURVIVED untouched (its original checkpoint,
    # not shadowed/overwritten by the fresh bot)
    assert "league_abcd" in reg.names()
    assert reg.get("league_abcd").ckpt_path == original_abcd_ckpt
    # the fresh bot has its OWN file (one-file-per-bot) under refresh_dir
    assert (Path(tmp_path) / "fresh" / "league_zzzz.pt").is_file()
    run.shutdown()
