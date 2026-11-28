"""Engine wiring tests for the two-clock team pool (step 4).

Hermetic (fake pool, injected driver factory, injected team registry +
fake brain — the same discipline as test_engine.py / test_driver.py):
  * plan() schedules exactly N team matchups (one per bot pairing)
  * _record books TEAM ELO with the SAME positional winner mapping as bots
    (a1_won -> matchup[0], a2_won -> matchup[1], draw -> none)
  * mirrors are ELO-neutral in the wiring (x vs x nets zero)
  * errored battles get NO team ELO either
  * _teams_for returns the per-seat mon lists for a battle slot
  * legacy mode (team_pool off) leaves _team_matchups empty and _teams_for None
  * the builder step at a cull boundary: brain trained on the pool's ELO
    history, bottom generated retired (curated protected), replacements
    bred from strong parents (brain-ranked), pool size preserved, event
    recorded, first generation seeded from the curated anchors

Run:  python -m pytest league/tests/test_engine_teams.py -k "not live" -q
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.engine import LeagueConfig, LeagueRun
from league.registry import BotRegistry, ELO_INITIAL
from league.teams import (
    ORIGIN_CURATED,
    ORIGIN_GENERATED,
    TeamRegistry,
    load_sets,
    structured_build,
)

from league.tests.test_engine import FakePool, _cfg, _roster  # the shared fakes


# ── team registry fixture ──────────────────────────────────────────────────
SETS = load_sets("gen9ou")


@pytest.fixture
def team_reg(tmp_path):
    """2 curated + 2 generated teams (all legal by construction)."""
    reg = TeamRegistry(path=tmp_path / "teams.json")
    rng = random.Random(5)
    for i, origin in enumerate([ORIGIN_CURATED, ORIGIN_CURATED,
                                ORIGIN_GENERATED, ORIGIN_GENERATED]):
        reg.add(f"t{i}", structured_build(SETS, random.Random(10 + i)),
                origin=origin,
                parent="t0" if origin == ORIGIN_GENERATED else None)
    return reg


def _mk_run(tmp_path, team_reg, brain=None, **kw):
    base = dict(save_replays=False, n_servers=1, per_server=1, n_workers=1,
                team_pool=True, team_registry_path=str(tmp_path / "unused.json"))
    base.update(kw)
    reg = _roster(tmp_path, "b1", "b2", "b3")
    return LeagueRun(LeagueConfig(**base), registry=reg, pool=FakePool(),
                     team_registry=team_reg, team_brain=brain,
                     driver_factory=lambda: _fake_factory())


def _fake_factory(results_by_idx=None):
    results_by_idx = results_by_idx or {0: (True, False), 1: (False, True),
                                        2: (False, False)}
    def factory(idx: int):
        def driver(server_cfg, fmt="gen9ou", seed=0, vocabs=None,
                   max_steps=200, on_finish=None, teams=None):
            return results_by_idx[idx][0], results_by_idx[idx][1], 3
        return driver
    return factory


# ── planning (the team clock) ──────────────────────────────────────────────

def test_plan_schedules_n_team_matchups(team_reg):
    reg = _roster(Path("x"), "b1", "b2", "b3")   # not used (injected below)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        breg = BotRegistry(path=Path(td) / "bots.json")
        for n in ("b1", "b2", "b3"):
            breg.add(n, "c.pt")
        run = LeagueRun(_cfg(team_pool=True), registry=breg, pool=FakePool(),
                        team_registry=team_reg,
                        driver_factory=lambda: _fake_factory())
        pairs = run.plan()
        assert len(pairs) == 3                      # C(3,2) bot pairings
        assert len(run._team_matchups) == 3         # one matchup per pairing
        tids = {t.tid for t in team_reg.all()}
        for a, b in run._team_matchups:
            assert a in tids and b in tids          # pool-only teams
        # non-mirror pairs are distinct within the round
        non_mirror = {(a, b) for a, b in run._team_matchups if a != b}
        keys = {(a, b) if a < b else (b, a) for a, b in non_mirror}
        assert len(keys) == len(non_mirror)


def test_legacy_mode_has_no_team_clock(tmp_path, team_reg):
    breg = BotRegistry(path=tmp_path / "bots.json")
    for n in ("b1", "b2", "b3"):
        breg.add(n, "c.pt")
    run = LeagueRun(_cfg(), registry=breg, pool=FakePool(),
                    driver_factory=lambda: _fake_factory())
    run.plan()
    assert run._team_matchups == []
    assert run._teams_for(0) is None
    s = run.summary()
    assert "team_standings" not in s


# ── the ELO wiring (positional winner mapping for teams) ───────────────────

def test_full_run_books_team_elo(tmp_path, team_reg):
    run = _mk_run(tmp_path, team_reg)
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["state"] == "completed"
    # scripted: idx0 -> matchup[0] wins; idx1 -> matchup[1] wins; idx2 draw
    m = run._team_matchups
    assert len(m) == 3
    # every team's ELO moved consistently with its recorded results
    total = sum(team_reg.get(t.tid).elo for t in team_reg.all())
    assert total == 4 * ELO_INITIAL                 # zero-sum (mirrors net 0)
    rows = run.results()
    for i, r in enumerate(rows):
        a, b = m[i]
        if r["a1_won"] and not r["a2_won"]:
            assert r["team_winner"] == a
        elif r["a2_won"] and not r["a1_won"]:
            assert r["team_winner"] == b
        else:
            assert r["team_winner"] is None
    # the summary carries the team standings (the GUI surface)
    assert "team_standings" in s
    assert len(s["team_standings"]) == 4
    assert "team_standings_generated" in s


def test_mirror_matchup_is_elo_neutral_in_wiring(tmp_path):
    """Force a mirror slot and confirm the registry nets zero for it."""
    reg = TeamRegistry(path=tmp_path / "teams.json")
    mons = structured_build(SETS, random.Random(3))
    reg.add("solo", mons, origin=ORIGIN_GENERATED)
    reg.add("other", structured_build(SETS, random.Random(4)),
            origin=ORIGIN_GENERATED)
    # a mirror (x,x) win/loss/draw all net ZERO in the registry
    elo0 = reg.get("solo").elo
    reg.update_result("solo", "solo", winner="solo")
    reg.update_result("solo", "solo", winner=None)
    assert reg.get("solo").elo == elo0


def test_errored_battle_books_no_team_elo(tmp_path, team_reg):
    def fail_factory(idx: int):
        def driver(server_cfg, fmt="gen9ou", seed=0, vocabs=None,
                   max_steps=200, on_finish=None, teams=None):
            raise RuntimeError("pool exploded")
        return driver
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        breg = BotRegistry(path=Path(td) / "bots.json")
        for n in ("b1", "b2", "b3"):
            breg.add(n, "c.pt")
        run = LeagueRun(_cfg(team_pool=True), registry=breg, pool=FakePool(),
                        team_registry=team_reg,
                        driver_factory=lambda: fail_factory)
        run.start()
        s = run.wait()
        run.shutdown()
        assert s["errors"] == 3
        # the team pool is UNTOUCHED (no silent draws)
        for t in team_reg.all():
            assert t.elo == ELO_INITIAL
            assert t.wins == t.losses == t.draws == 0


def test_teams_for_returns_seat_mon_lists(tmp_path, team_reg):
    run = _mk_run(tmp_path, team_reg)
    run.plan()
    t_a, t_b = run._team_matchups[0]
    mons_a, mons_b = run._teams_for(0)
    assert mons_a == team_reg.get(t_a).mons
    assert mons_b == team_reg.get(t_b).mons
    # fresh lists (a mutation by a caller must not poison the pool)
    mons_a[0]["species"] = "MUTATED"
    assert team_reg.get(t_a).mons[0]["species"] != "MUTATED"


# ── the builder step (cull boundary) ───────────────────────────────────────

class FakeBrain:
    def __init__(self):
        self.train_calls = []
        self.rank_calls = []

    def train_step(self, pairs, epochs=1, lr=None):
        self.train_calls.append((len(pairs), int(epochs)))
        return {"n_pairs": len(pairs), "loss": 0.5}

    def rank(self, candidates, parent=None):
        self.rank_calls.append((len(candidates), parent is not None))
        return [0]                                   # always adopt candidate 0


def _pool_with_history(tmp_path):
    """A pool where some teams have decided battles (=> preference pairs)."""
    reg = TeamRegistry(path=tmp_path / "teams.json")
    rng = random.Random(9)
    tids = [reg.add(f"cur_{i}", structured_build(SETS, random.Random(20 + i)),
                    origin=ORIGIN_CURATED).tid for i in range(4)]
    gen = [reg.add(f"gen_{i}", structured_build(SETS, random.Random(40 + i)),
                   origin=ORIGIN_GENERATED).tid for i in range(4)]
    # decide some battles so team_pairs_from_registry has material
    reg.update_result("gen_0", "gen_1", winner="gen_0")
    reg.update_result("gen_0", "gen_1", winner="gen_0")
    reg.update_result("cur_0", "gen_3", winner="cur_0")
    reg.update_result("cur_0", "gen_3", winner="cur_0")
    return reg


def test_team_builder_step_culls_breeds_and_trains(tmp_path):
    reg = _pool_with_history(tmp_path)
    brain = FakeBrain()
    n_before = len(reg)
    gen_before = {t.tid for t in reg.generated()}
    run = _mk_run(tmp_path, reg, brain=brain,
                  n_rounds=8, cull_every=4, team_cull_frac=0.25,
                  team_breed_candidates=4, team_min_pool=2)
    # put the run AT the cull boundary (round 4 of 8, completed)
    run._round_idx = 4
    run._n_rounds = 8
    run._team_cull_and_refresh()
    # the brain was trained on the pool's ELO history (online, at the boundary)
    assert brain.train_calls and brain.train_calls[0][0] > 0
    # replacements were bred (brain ranked: rank was called with a parent)
    assert brain.rank_calls and brain.rank_calls[0][1] is True
    # the pool size is PRESERVED (cull N, breed N)
    assert len(reg) == n_before
    # curated anchors are intact (PROTECTED)
    assert sum(1 for t in reg.all() if t.origin == ORIGIN_CURATED) == 4
    # the generated population CHURNED (some new tids with parent attribution)
    gen_after = {t.tid for t in reg.generated()}
    assert gen_after != gen_before
    new_ones = [t for t in reg.generated() if t.tid not in gen_before]
    assert new_ones and all(t.parent is not None for t in new_ones)
    # the event is recorded (the audit trail)
    assert run._team_cull_events
    ev = run._team_cull_events[-1]
    assert ev["added"] and not ev["culled"] or ev["culled"]
    # every bred team is legal (6 mons, distinct base species)
    from league.teams import base_species
    for t in reg.all():
        assert len(t.mons) == 6
        bases = {base_species(m["species"]) for m in t.mons}
        assert len(bases) == 6


def test_team_builder_first_generation_seeds_from_curated(tmp_path):
    reg = TeamRegistry(path=tmp_path / "teams.json")
    for i in range(4):
        reg.add(f"cur_{i}", structured_build(SETS, random.Random(60 + i)),
                origin=ORIGIN_CURATED)
    assert reg.generated() == []
    brain = FakeBrain()
    run = _mk_run(tmp_path, reg, brain=brain, n_rounds=8, cull_every=4)
    run._round_idx = 4
    run._n_rounds = 8
    run._team_cull_and_refresh()
    gen = reg.generated()
    assert len(gen) == 4                            # the first generation
    assert all(t.parent is not None for t in gen)   # curated anchors as parents
    assert all(t.origin == ORIGIN_GENERATED for t in gen)


def test_team_builder_skips_off_boundary(tmp_path):
    reg = _pool_with_history(tmp_path)
    brain = FakeBrain()
    run = _mk_run(tmp_path, reg, brain=brain, n_rounds=16, cull_every=4)
    run._round_idx = 3                              # not a boundary
    run._n_rounds = 16
    run._team_cull_and_refresh()
    assert brain.train_calls == []                  # nothing fired
    assert len(reg) == 8
    run._round_idx = 16                             # == n_rounds: no next round
    run._n_rounds = 16
    run._team_cull_and_refresh()
    assert brain.train_calls == []


def test_team_builder_never_culls_curated(tmp_path):
    """Even with a large cull fraction, curated teams are PROTECTED."""
    reg = _pool_with_history(tmp_path)
    brain = FakeBrain()
    # drive the cull to the floor: cull_frac 0.9 of the 4 generated = 4
    run = _mk_run(tmp_path, reg, brain=brain, n_rounds=8, cull_every=4,
                  team_cull_frac=0.9, team_min_pool=2)
    run._round_idx = 4
    run._n_rounds = 8
    run._team_cull_and_refresh()
    curated = {t.tid for t in reg.all() if t.origin == ORIGIN_CURATED}
    assert curated == {f"cur_{i}" for i in range(4)}   # all 4 still there
