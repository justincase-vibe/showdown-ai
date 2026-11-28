"""Team registry + generator tests (league/teams.py).

Covers the handoff spec for the league's living team pool:
  * registry add / save / load round-trip
  * ELO math (mirrors test_registry.py::test_elo_magnitudes_standard)
  * curated teams are PROTECTED (remove raises), generated are cullable
  * bottom_generated / top_generated never surface curated teams
  * mutation validity: 6 mons, no base-species dupes, EXACTLY ONE slot
    changed per op (a no-op re-pick of the same set/item is a bug),
    legality by construction vs bot/data/movesets.json
  * structured_build: 6 distinct base species, verified sets only
  * seed_curated: exactly the 28 human teams from bot/team_pools.json, idempotent
  * determinism under a fixed seed
  * thread-safety of update_result (the league runs concurrent battles)

Run:  python -m pytest league/tests/test_teams.py -k "not live" -q
"""
from __future__ import annotations

import json
import random
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.teams import (
    ELO_INITIAL,
    ORIGIN_CURATED,
    ORIGIN_GENERATED,
    OP_SWAP_ITEM,
    OP_SWAP_MON,
    OP_SWAP_SET,
    Team,
    TeamRegistry,
    base_species,
    load_sets,
    mutate,
    new_generated_tid,
    resolve_set,
    seed_curated,
    structured_build,
)

#: shared data (read-only; these files are the verified competitive-set source)
SETS = load_sets("gen9ou")
POOLS = json.loads((ROOT / "bot" / "team_pools.json").read_text(encoding="utf-8"))
CURATED_TEAMS: list = POOLS["gen9ou"]["teams"]


@pytest.fixture
def reg(tmp_path):
    return TeamRegistry(path=tmp_path / "teams.json")


def _sample_mons(seed: int = 7) -> list:
    """A 6-mon team in the exchange format (all from verified sets)."""
    return structured_build(SETS, random.Random(seed))


# ═══════════════════════════════════════════════════════════════════════════
#  registry: add / get / round-trip
# ═══════════════════════════════════════════════════════════════════════════

def test_add_and_get(reg):
    mons = _sample_mons()
    reg.add("gen_a_r0", mons, origin=ORIGIN_GENERATED, parent=None, created_round=0)
    t = reg.get("gen_a_r0")
    assert t.tid == "gen_a_r0"
    assert t.origin == ORIGIN_GENERATED
    assert t.mons == mons
    assert t.elo == ELO_INITIAL
    assert t.wins == t.losses == t.draws == 0


def test_add_updates_in_place(reg):
    mons = _sample_mons(1)
    reg.add("gen_a_r0", mons, origin=ORIGIN_GENERATED)
    mons2 = _sample_mons(2)
    reg.add("gen_a_r0", mons2, origin=ORIGIN_GENERATED)
    assert len(reg) == 1
    assert reg.get("gen_a_r0").mons == mons2


def test_persistence_round_trip(tmp_path):
    """add + result + reload -> every field survives (mons included)."""
    path = tmp_path / "teams.json"
    reg1 = TeamRegistry(path=path)
    mons_a = _sample_mons(11)
    mons_b = _sample_mons(12)
    reg1.add("gen_a", mons_a, origin=ORIGIN_GENERATED)
    reg1.add("gen_b", mons_b, origin=ORIGIN_GENERATED, parent="gen_a", created_round=4)
    reg1.update_result("gen_a", "gen_b", winner="gen_a")

    reg2 = TeamRegistry(path=path)
    assert len(reg2) == 2
    assert set(reg2.tids()) == {"gen_a", "gen_b"}
    ta, tb = reg2.get("gen_a"), reg2.get("gen_b")
    assert ta.mons == mons_a and tb.mons == mons_b
    assert ta.wins == 1 and tb.losses == 1
    assert ta.elo > ELO_INITIAL and tb.elo < ELO_INITIAL
    assert ta.elo + tb.elo == 2 * ELO_INITIAL          # zero-sum survives reload
    assert tb.parent == "gen_a"
    assert tb.created_round == 4
    assert ta.origin == ORIGIN_GENERATED


def test_from_dict_ignores_unknown_fields():
    """Forward-compat: a future teams.json with extra keys must still load."""
    d = Team(tid="x", mons=[{"species": "Mew"}]).to_dict()
    d["some_future_field"] = 42
    t = Team.from_dict(d)
    assert t.tid == "x"
    assert not hasattr(t, "some_future_field")


def test_origin_counts_and_generated(reg):
    reg.add("cur_00", _sample_mons(1), origin=ORIGIN_CURATED)
    reg.add("gen_a", _sample_mons(2), origin=ORIGIN_GENERATED)
    reg.add("gen_b", _sample_mons(3), origin=ORIGIN_GENERATED)
    assert reg.origin_counts() == {"curated": 1, "generated": 2}
    assert {t.tid for t in reg.generated()} == {"gen_a", "gen_b"}


# ═══════════════════════════════════════════════════════════════════════════
#  ELO (mirrors league/tests/test_registry.py conventions exactly)
# ═══════════════════════════════════════════════════════════════════════════

def test_elo_magnitudes_standard(reg):
    """Exact standard ELO magnitudes (K=32), pinned against the formula.
    Mirrors test_registry.py::test_elo_magnitudes_standard (same formula,
    same scenarios, same expected values).

    A=1500 vs B=1700: E_A = 1/(1+10^(200/400)) = 0.24025...
      A wins -> A +24, B -24
    """
    reg.add("a", _sample_mons(1), elo=1500)
    reg.add("b", _sample_mons(2), elo=1700)
    reg.update_result("a", "b", winner="a")
    assert reg.get("a").elo == 1524
    assert reg.get("b").elo == 1676


def test_elo_magnitudes_underdog_win(reg):
    """An underdog win must pay the upset premium (K*(1-E_underdog)).
    Mirrors test_registry.py::test_elo_magnitudes_underdog_win."""
    reg.add("a", _sample_mons(1), elo=1500)
    reg.add("b", _sample_mons(2), elo=1700)
    reg.update_result("a", "b", winner="b")          # b (favorite) wins
    assert reg.get("a").elo == 1492                    # -8, not -24
    assert reg.get("b").elo == 1708

    reg.add("c", _sample_mons(3), elo=1700)
    reg.update_result("a", "c", winner="a")          # a (underdog) wins
    # a is now 1492 vs c's 1700 -> E_a=0.2319 -> upset premium round(32*0.768)=+25
    assert reg.get("a").elo == 1492 + 25
    assert reg.get("c").elo == 1700 - 25


def test_elo_draw_unequal_ratings(reg):
    """A draw between UNEQUAL ratings shifts ELO toward the favorite's
    expectation (standard ELO: score 0.5 for both)."""
    reg.add("a", _sample_mons(1), elo=1500)
    reg.add("b", _sample_mons(2), elo=1700)
    reg.update_result("a", "b", winner=None)
    assert reg.get("a").elo == 1508                    # +8
    assert reg.get("b").elo == 1692                    # -8


def test_elo_zero_sum_and_counters(reg):
    reg.add("a", _sample_mons(1))
    reg.add("b", _sample_mons(2))
    reg.update_result("a", "b", winner="a")
    reg.update_result("a", "b", winner="b")
    reg.update_result("a", "b", winner=None)
    ta, tb = reg.get("a"), reg.get("b")
    assert ta.elo + tb.elo == 2 * ELO_INITIAL
    assert (ta.wins, ta.losses, ta.draws) == (1, 1, 1)
    assert (tb.wins, tb.losses, tb.draws) == (1, 1, 1)


def test_elo_convergence(reg):
    reg.add("strong", _sample_mons(1))
    reg.add("weak", _sample_mons(2))
    for _ in range(10):
        reg.update_result("strong", "weak", winner="strong")
    assert reg.get("strong").elo > ELO_INITIAL + 50
    assert reg.get("weak").elo < ELO_INITIAL - 50


def test_standings_sorts_by_elo_and_filters_origin(reg):
    reg.add("cur_00", _sample_mons(1), origin=ORIGIN_CURATED)
    reg.add("gen_hi", _sample_mons(2), origin=ORIGIN_GENERATED)
    reg.add("gen_lo", _sample_mons(3), origin=ORIGIN_GENERATED)
    reg.update_result("gen_hi", "gen_lo", winner="gen_hi")
    s_all = reg.standings()
    assert len(s_all) == 3
    assert s_all[0]["tid"] == "gen_hi"
    assert s_all[2]["tid"] == "gen_lo"
    s_gen = reg.standings(origin=ORIGIN_GENERATED)
    assert {r["tid"] for r in s_gen} == {"gen_hi", "gen_lo"}
    assert all(r["origin"] == ORIGIN_GENERATED for r in s_gen)


# ═══════════════════════════════════════════════════════════════════════════
#  culling: curated PROTECTED, generated cullable
# ═══════════════════════════════════════════════════════════════════════════

def test_remove_generated_ok(reg):
    reg.add("gen_a", _sample_mons(1), origin=ORIGIN_GENERATED)
    reg.remove("gen_a")
    assert len(reg) == 0
    with pytest.raises(KeyError):
        reg.get("gen_a")


def test_remove_curated_raises(reg):
    """User ruling: the human-built baseline is never culled by the builder."""
    reg.add("cur_00", _sample_mons(1), origin=ORIGIN_CURATED)
    with pytest.raises(ValueError, match="PROTECTED"):
        reg.remove("cur_00")
    assert len(reg) == 1                      # still there


def test_remove_missing_raises(reg):
    with pytest.raises(KeyError, match="not in pool"):
        reg.remove("ghost")


def test_bottom_top_generated_exclude_curated(reg):
    reg.add("cur_00", _sample_mons(1), origin=ORIGIN_CURATED, elo=1000)
    reg.add("gen_a", _sample_mons(2), origin=ORIGIN_GENERATED, elo=1400)
    reg.add("gen_b", _sample_mons(3), origin=ORIGIN_GENERATED, elo=1500)
    reg.add("gen_c", _sample_mons(4), origin=ORIGIN_GENERATED, elo=1600)
    bottom = reg.bottom_generated(2)
    assert [t.tid for t in bottom] == ["gen_a", "gen_b"]
    top = reg.top_generated(1)
    assert [t.tid for t in top] == ["gen_c"]
    # a curated team with the WORST elo in the pool is never a cull candidate
    assert all(t.origin == ORIGIN_GENERATED for t in bottom + top)


# ═══════════════════════════════════════════════════════════════════════════
#  generator: load_sets / resolve_set
# ═══════════════════════════════════════════════════════════════════════════

def test_load_sets_shape():
    assert len(SETS) == 108                                   # gen9ou species
    assert sum(1 for v in SETS.values() if len(v) >= 2) == 57  # species w/ ≥2 sets
    for species, lst in SETS.items():
        assert species and lst, species
        for s in lst:
            assert isinstance(s["moves"], list) and 1 <= len(s["moves"]) <= 4
            assert all(isinstance(mv, str) and mv for mv in s["moves"])
            assert isinstance(s["evs"], dict)
            assert s["item"] is None or isinstance(s["item"], str)


def test_resolve_set_exchange_format():
    """resolve_set output must be in the team_pools exchange format that
    bot/teams.py::_to_tb consumes (species/moves/item/ability/nature/evs list6)."""
    species = "Clefable"
    s = SETS[species][0]
    mon = resolve_set(species, s)
    assert mon["species"] == species
    assert isinstance(mon["moves"], list) and 1 <= len(mon["moves"]) <= 4
    assert isinstance(mon["evs"], list) and len(mon["evs"]) == 6
    assert all(isinstance(v, int) for v in mon["evs"])
    assert mon["evs"][0] == s["evs"].get("hp", 0)
    assert mon["ivs"] == [31] * 6
    # the curated pools omit ivs — _to_tb tolerates both, and every key the
    # curated pools use must be present here
    curated_keys = {"species", "item", "ability", "moves", "nature", "evs"}
    assert curated_keys <= set(mon.keys())


def test_base_species():
    assert base_species("Rotom-Wash") == "Rotom"
    assert base_species("Deoxys-Speed") == "Deoxys"
    assert base_species("Garchomp") == "Garchomp"


# ═══════════════════════════════════════════════════════════════════════════
#  generator: mutate
# ═══════════════════════════════════════════════════════════════════════════

def _bases(mons):
    return {base_species(m["species"]) for m in mons}


def _assert_legal_team(mons):
    """6 mons, distinct base species, every mon from verified set data."""
    assert len(mons) == 6
    assert len(_bases(mons)) == 6, f"base-species duplicate in {sorted(_bases(mons))}"
    for m in mons:
        sp = m["species"]
        assert sp in SETS, f"species {sp!r} not in movesets data"
        assert 1 <= len(m["moves"]) <= 4
        assert len(m["evs"]) == 6


def _changed_slots(old, new):
    return [i for i in range(6) if old[i] != new[i]]


@pytest.mark.parametrize("op", [OP_SWAP_MON, OP_SWAP_SET, OP_SWAP_ITEM])
def test_mutate_exactly_one_slot_changed(op):
    """CONTRACT: a mutation changes EXACTLY one slot (the rest of the team is
    preserved) — a no-op re-pick of the same set/item is a bug (it was observed
    at 58% for swap_item and 80% for swap_set before the fix)."""
    for seed in range(20):
        team = _sample_mons(seed)
        try:
            new, used = mutate(team, SETS, random.Random(seed), op=op)
        except RuntimeError:
            continue                                 # inapplicable for this team
        assert used == op
        _assert_legal_team(new)
        changed = _changed_slots(team, new)
        assert len(changed) == 1, (
            f"{op} seed={seed}: {len(changed)} slots changed — must be exactly 1"
        )
        i = changed[0]
        # the other five mons are untouched (object identity: mutation must not
        # rewrite preserved slots)
        for j in range(6):
            if j != i:
                assert new[j] is team[j]


def test_mutate_swap_mon_replaces_with_another_species():
    team = _sample_mons(3)
    new, op = mutate(team, SETS, random.Random(3), op=OP_SWAP_MON)
    i = _changed_slots(team, new)[0]
    assert new[i]["species"] != team[i]["species"]
    assert base_species(new[i]["species"]) not in _bases(team)
    assert op == OP_SWAP_MON


def test_mutate_swap_set_keeps_species_changes_set():
    team = _sample_mons(3)
    new, op = mutate(team, SETS, random.Random(3), op=OP_SWAP_SET)
    i = _changed_slots(team, new)[0]
    assert new[i]["species"] == team[i]["species"]
    assert new[i] != team[i]                          # and actually a different set


def test_mutate_swap_item_keeps_species_changes_item():
    team = _sample_mons(3)
    new, op = mutate(team, SETS, random.Random(3), op=OP_SWAP_ITEM)
    i = _changed_slots(team, new)[0]
    assert new[i]["species"] == team[i]["species"]
    assert new[i]["item"] != team[i]["item"]
    # everything else on that mon is preserved
    for k in ("moves", "ability", "nature", "evs"):
        assert new[i][k] == team[i][k]


def test_mutate_wrong_length_raises():
    with pytest.raises(ValueError, match="6 mons"):
        mutate(_sample_mons()[:5], SETS, random.Random(0))


def test_mutate_all_ops_inapplicable_raises():
    """A team where NO op is applicable (tiny set data: 6 single-set species
    sharing one item) must fail loudly, not silently return the input team."""
    tiny = {f"Sp{i}": [{"name": "only", "moves": ["Tackle"], "item": "TheOneItem",
                        "ability": "Ab", "nature": "Neutral", "evs": {"hp": 4}}]
            for i in range(6)}
    mons = [resolve_set(f"Sp{i}", tiny[f"Sp{i}"][0]) for i in range(6)]
    for seed in range(5):
        with pytest.raises(RuntimeError, match="no applicable op"):
            mutate(mons, tiny, random.Random(seed))


def test_mutate_deterministic_under_seed():
    team = _sample_mons(9)
    r1 = mutate(team, SETS, random.Random(123))
    r2 = mutate(team, SETS, random.Random(123))
    assert r1 == r2                                   # (mons, op) identical
    r3 = mutate(team, SETS, random.Random(124))
    assert r3[1] in (OP_SWAP_MON, OP_SWAP_SET, OP_SWAP_ITEM)


def test_mutate_works_on_curated_pool_teams():
    """Curated pool teams (no ivs/tera_type, different EVs than movesets) must
    be mutable too — every op either applies exactly one change or degrades."""
    for seed in range(10):
        team = [dict(m) for m in CURATED_TEAMS[0]]
        for op in (OP_SWAP_MON, OP_SWAP_SET, OP_SWAP_ITEM):
            try:
                new, used = mutate(team, SETS, random.Random(seed * 31 + [1, 2, 3][[OP_SWAP_MON, OP_SWAP_SET, OP_SWAP_ITEM].index(op)]), op=op)
            except RuntimeError:
                continue
            assert len(_changed_slots(team, new)) == 1
            _assert_legal_team(new)


# ═══════════════════════════════════════════════════════════════════════════
#  generator: structured_build
# ═══════════════════════════════════════════════════════════════════════════

def test_structured_build_distinct_and_legal():
    for seed in range(25):
        mons = structured_build(SETS, random.Random(seed))
        _assert_legal_team(mons)


def test_structured_build_deterministic():
    a = structured_build(SETS, random.Random(5))
    b = structured_build(SETS, random.Random(5))
    assert a == b


def test_structured_build_rejects_tiny_data():
    tiny = {f"S{i}": [{"name": "x", "moves": ["Tackle"], "item": None,
                       "ability": None, "nature": None, "evs": {}}]
            for i in range(5)}
    with pytest.raises(RuntimeError, match="fewer than 6"):
        structured_build(tiny, random.Random(0))


# ═══════════════════════════════════════════════════════════════════════════
#  seeding the curated baseline
# ═══════════════════════════════════════════════════════════════════════════

def test_seed_curated_loads_all_28(tmp_path):
    reg = TeamRegistry(path=tmp_path / "teams.json")
    n = seed_curated(reg, "gen9ou")
    assert n == 28
    assert len(reg) == 28
    assert reg.origin_counts() == {"curated": 28}
    assert [t.tid for t in reg.all()] == [f"cur_{i:02d}" for i in range(28)]
    for i, t in enumerate(reg.all()):
        assert t.origin == ORIGIN_CURATED
        assert t.parent is None
        assert t.mons == CURATED_TEAMS[i]           # exact pools content


def test_seed_curated_idempotent(tmp_path):
    path = tmp_path / "teams.json"
    reg1 = TeamRegistry(path=path)
    seed_curated(reg1, "gen9ou")
    reg2 = TeamRegistry(path=path)
    assert seed_curated(reg2, "gen9ou") == 28
    assert len(reg2) == 28                          # no duplicates
    assert reg2.get("cur_00").mons == CURATED_TEAMS[0]


def test_seed_curated_missing_pools_raises(tmp_path):
    reg = TeamRegistry(path=tmp_path / "teams.json")
    with pytest.raises(RuntimeError, match="no team pools"):
        seed_curated(reg, "gen9ou", pools_path=tmp_path / "nope.json")


# ═══════════════════════════════════════════════════════════════════════════
#  generated tids
# ═══════════════════════════════════════════════════════════════════════════

def test_new_generated_tid_unique_and_readable(reg):
    reg.add("gen_cur_05_r12_abcd", _sample_mons(1), origin=ORIGIN_GENERATED)
    seen = set()
    for i in range(50):
        tid = new_generated_tid(reg, parent="gen_cur_05_r12", rnd=12)
        assert tid.startswith("gen_")
        assert tid not in reg and tid not in seen
        seen.add(tid)
    # parent tag is carried into the tid (attribution readability; the
    # implementation compresses gen_/cur_ prefixes to keep tids short)
    assert "c05_r12" in new_generated_tid(reg, parent="gen_cur_05_r12", rnd=3)


# ═══════════════════════════════════════════════════════════════════════════
#  concurrency (the league runs battles on worker threads)
# ═══════════════════════════════════════════════════════════════════════════

def test_concurrent_updates_consistent(tmp_path):
    """Mirror of test_registry.py::test_concurrent_updates_consistent for teams:
    zero-sum ELO conserved, every result counted once, file stays valid JSON."""
    path = tmp_path / "teams.json"
    reg = TeamRegistry(path=path)
    n_teams = 6
    tids = [f"t{i}" for i in range(n_teams)]
    mons_cache = [_sample_mons(i) for i in range(n_teams)]
    for tid, mons in zip(tids, mons_cache):
        reg.add(tid, mons)
    total_initial_elo = n_teams * ELO_INITIAL

    n_threads, per_thread = 8, 25
    total_battles = n_threads * per_thread
    barrier = threading.Barrier(n_threads)
    errors: list = []

    def worker(tid: int) -> None:
        try:
            barrier.wait()
        except Exception as e:
            errors.append(e)
            return
        try:
            for i in range(per_thread):
                a = tids[(tid + i) % n_teams]
                b = tids[(tid + i + 1) % n_teams]
                r = (tid * per_thread + i) % 3
                winner = a if r == 0 else (b if r == 1 else None)
                reg.update_result(a, b, winner=winner)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"worker errors: {errors}"
    assert sum(reg.get(tid).elo for tid in tids) == total_initial_elo
    total_w = sum(reg.get(tid).wins for tid in tids)
    total_l = sum(reg.get(tid).losses for tid in tids)
    total_d = sum(reg.get(tid).draws for tid in tids)
    assert total_w == total_l
    assert total_d % 2 == 0
    assert total_w + total_d // 2 == total_battles
    data = json.loads(path.read_text(encoding="utf-8"))
    assert {t["tid"] for t in data["teams"]} == set(tids)
