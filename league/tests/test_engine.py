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


def test_config_roundtrip():
    cfg = LeagueConfig(pairing="elo_weighted", best_of=2, greedy=False,
                       device="cuda", n_workers=4, max_consecutive_errors=7)
    same = LeagueConfig.from_dict(cfg.to_dict())
    assert same == cfg


def test_config_from_dict_ignores_unknown_keys():
    cfg = LeagueConfig.from_dict({"pairing": "round_robin", "bogus": 1})
    assert cfg.pairing == "round_robin"
    assert not hasattr(cfg, "bogus")
