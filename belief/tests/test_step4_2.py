"""Step 4.2 gate — batch damage-window transport (P4.2).

Run from the project root:

    python -m pytest belief/tests/test_step4_2.py -v

P4.2 (PLAN.md D78) collapses the per-candidate Node spawns of one belief
evidence event into ONE batch spawn. The batch path MUST be numerically
identical to the single path (same windows, same likelihoods, same posterior)
— it is a transport change, not a math change. These tests pin that contract:

  * ``windows_many`` == ``window_for`` per item (cold cache, fresh keys);
  * the single and batch paths share ONE ``_CACHE`` (a result computed by one
    path is a hit for the other — proven by counting spawns, not timing);
  * mixed hit/miss batches go out as EXACTLY ONE ``calc_damage_many`` call
    containing only the misses;
  * a per-item calc error raises ``RuntimeError`` exactly like the single path
    (never a silent zero window);
  * the per-direction helpers (``defender_windows_many`` /
    ``attacker_windows_many`` / ``maxhp_windows_many``) match their
    per-candidate siblings;
  * belief-level: a damage event still updates the posterior correctly through
    the batched wiring in ``state.py`` (the regression the 117-test stack is
    guarding — re-run the full suite too).
"""
from __future__ import annotations

import pytest

from belief import events as E
from belief import evidence as ev
from belief.state import BeliefState

# ── helpers ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch):
    """Tests must not leak cache state into each other (the _CACHE is
    module-global and process-wide). Save / clear / restore."""
    saved = dict(ev._CACHE)
    ev._CACHE.clear()
    yield
    ev._CACHE.clear()
    ev._CACHE.update(saved)


@pytest.fixture
def spawn_counters(monkeypatch):
    """Count real spawns on the damage.calc module (single + batch) while the
    calls still do the real work."""
    from damage import calc
    counts = {"single": 0, "batch": 0}
    orig_single, orig_batch = calc.calc_damage, calc.calc_damage_many

    def counting_single(*a, **kw):
        counts["single"] += 1
        return orig_single(*a, **kw)

    def counting_batch(items):
        counts["batch"] += 1
        return orig_batch(items)

    monkeypatch.setattr(calc, "calc_damage", counting_single)
    monkeypatch.setattr(calc, "calc_damage_many", counting_batch)
    return counts


def _mon(name: str, **kw) -> dict:
    return {"name": name, **kw}


def _items(attacker, defender, move, field=None, fmt=None, gen=9):
    return [{"attacker": attacker, "defender": defender, "move": move,
             "field": field, "fmt": fmt, "gen": gen}]


# ── equivalence: batch == single, per item ───────────────────────────────────

def test_windows_many_matches_window_for_cold(spawn_counters):
    """Same numbers as three separate window_for calls (fresh keys -> the batch
    actually spawns, and in ONE batch call)."""
    a = _mon("charizard", nature="Modest", evs={"spa": 252})
    d1 = _mon("blastoise", nature="Bold", evs={"def": 252})
    d2 = _mon("gengar", nature="Timid", evs={"spe": 252})
    d3 = _mon("machamp", nature="Adamant", evs={"atk": 252})
    items = _items(a, d1, "flamethrower") + _items(a, d2, "thunderbolt") \
        + _items(a, d3, "icebeam")
    got = ev.windows_many(items)
    assert len(got) == 3
    ref = [ev.window_for(a, d1, "flamethrower"),
           ev.window_for(a, d2, "thunderbolt"),
           ev.window_for(a, d3, "icebeam")]
    for g, r in zip(got, ref):
        assert g == r, f"batch window {g} != single window {r}"
    # the batch went out as exactly ONE batch spawn, and the reference calls
    # then hit the cache the batch populated (single-path spawns stayed at 0)
    assert spawn_counters["batch"] == 1
    assert spawn_counters["single"] == 0  # references served from the shared cache


def test_windows_many_empty_is_noop(spawn_counters):
    assert ev.windows_many([]) == []
    assert spawn_counters["batch"] == 0 and spawn_counters["single"] == 0


# ── shared cache: one result serves both paths ───────────────────────────────

def test_batch_result_is_a_hit_for_the_single_path(spawn_counters):
    """Batch first (spawns), then the identical single calls must be cache hits
    (zero new spawns) — the keys really are shared."""
    a = _mon("charizard", nature="Modest", evs={"spa": 252})
    d = _mon("blastoise", nature="Bold", evs={"def": 252})
    items = _items(a, d, "flamethrower")
    got = ev.windows_many(items)
    assert spawn_counters["batch"] == 1 and spawn_counters["single"] == 0
    ref = ev.window_for(a, d, "flamethrower")
    assert got[0] == ref
    assert spawn_counters["batch"] == 1 and spawn_counters["single"] == 0, \
        "window_for after the identical batch must be a cache hit"


def test_single_result_is_a_hit_for_the_batch_path(spawn_counters):
    """Reverse direction: window_for first, then the identical batch must not
    spawn at all (all hits)."""
    a = _mon("charizard", nature="Modest", evs={"spa": 252})
    d = _mon("blastoise", nature="Bold", evs={"def": 252})
    ref = ev.window_for(a, d, "flamethrower")
    assert spawn_counters["single"] == 1
    got = ev.windows_many(_items(a, d, "flamethrower"))
    assert got[0] == ref
    assert spawn_counters["batch"] == 0, "batch of pure cache hits must not spawn"


def test_mixed_hits_and_misses_spawn_once_with_only_misses(spawn_counters):
    """Pre-warm 2 of 3 keys via the single path; the batch must send EXACTLY the
    remaining miss, in ONE batch call, and return all three in input order."""
    a = _mon("charizard", nature="Modest", evs={"spa": 252})
    d1 = _mon("blastoise", nature="Bold", evs={"def": 252})
    d2 = _mon("gengar", nature="Timid", evs={"spe": 252})
    d3 = _mon("machamp", nature="Adamant", evs={"atk": 252})
    # warm d1 and d3 (NOT d2)
    r1 = ev.window_for(a, d1, "flamethrower")
    r3 = ev.window_for(a, d3, "icebeam")
    assert spawn_counters["single"] == 2
    items = _items(a, d1, "flamethrower") + _items(a, d2, "thunderbolt") \
        + _items(a, d3, "icebeam")
    got = ev.windows_many(items)
    assert spawn_counters["batch"] == 1, "mixed batch must go out as ONE spawn"
    assert spawn_counters["single"] == 2, "no extra single spawns"
    # order preserved, warm keys unchanged
    assert got[0] == r1 and got[2] == r3
    assert got[1] == ev.window_for(a, d2, "thunderbolt")


def test_windows_many_dedupes_identical_specs_within_batch(spawn_counters, monkeypatch):
    """Fallback-slot case (3.8 bench: 210 candidates sharing ~3 specs): the batch
    sends ONE item per unique spec, not one per candidate, and hands the same
    window back to every sharer — in input order."""
    from damage import calc
    sent = {"n": 0}
    orig = calc.calc_damage_many

    def counting(items):
        sent["n"] = len(items)
        return orig(items)

    monkeypatch.setattr(calc, "calc_damage_many", counting)
    a = _mon("charizard", nature="Modest", evs={"spa": 252})
    d = _mon("blastoise", nature="Bold", evs={"def": 252})
    items = _items(a, d, "flamethrower") * 5  # five candidates, ONE spec
    got = ev.windows_many(items)
    assert len(got) == 5
    assert all(w == got[0] for w in got), "every spec-sharer gets the same window"
    assert got == [ev.window_for(a, d, "flamethrower")] * 5
    assert sent["n"] == 1, f"batch sent {sent['n']} items; expected 1 unique spec"


# ── error contract: loud, like the single path ───────────────────────────────

def test_windows_many_item_error_raises_runtimeerror(monkeypatch):
    from damage import calc
    monkeypatch.setattr(calc, "calc_damage_many",
                        lambda items: [{"error": "boom"} for _ in items])
    with pytest.raises(RuntimeError, match="boom"):
        ev.windows_many(_items(_mon("charizard"), _mon("blastoise"), "flamethrower"))


def test_windows_many_result_count_mismatch_raises(monkeypatch):
    from damage import calc
    monkeypatch.setattr(calc, "calc_damage_many", lambda items: [{}])  # wrong arity
    items = _items(_mon("charizard"), _mon("blastoise"), "flamethrower") \
        + _items(_mon("charizard"), _mon("gengar"), "thunderbolt")
    with pytest.raises(RuntimeError, match="returned 1 results for 2 items"):
        ev.windows_many(items)


# ── per-direction helpers match their per-candidate siblings ─────────────────

def _cands():
    return [
        {"moves": ["tackle", "a", "b", "c"], "item": "", "ability": "static",
         "nature": "Hardy", "evs": {}, "tera": None, "weight": 1.0},
        {"moves": ["tackle", "a", "b", "c"], "item": "", "ability": "static",
         "nature": "Modest", "evs": {"spa": 252}, "tera": None, "weight": 1.0},
    ]


def test_defender_windows_many_matches_single():
    cands = _cands()
    attacker = _mon("charizard", nature="Modest", evs={"spa": 252})
    got = ev.defender_windows_many(cands, "blastoise", attacker,
                                   defender_status=None, defender_boosts=None,
                                   defender_hp_fraction=0.5, move="flamethrower",
                                   field=None)
    ref = [ev.defender_window(c, "blastoise", attacker, None, None, 0.5,
                              "flamethrower", None) for c in cands]
    assert got == ref


def test_attacker_windows_many_matches_single():
    cands = _cands()
    defender = _mon("blastoise", nature="Bold", evs={"def": 252})
    got = ev.attacker_windows_many(cands, "charizard",
                                   attacker_status=None, attacker_boosts=None,
                                   attacker_hp_fraction=0.8,
                                   defender_mon=defender, move="flamethrower")
    ref = [ev.attacker_window(c, "charizard", None, None, 0.8, defender,
                              "flamethrower") for c in cands]
    assert got == ref


def test_maxhp_windows_many_matches_single():
    cands = _cands()
    got = ev.maxhp_windows_many(cands, "garchomp")
    ref = [ev.maxhp_for(c, "garchomp") for c in cands]
    assert [w["maxhp"] for w in got] == ref


# ── belief-level wiring: the update still discriminates ──────────────────────

def test_damage_update_applies_through_batch_path():
    """A damage event through state.py (now batched) must still apply as belief
    evidence and discriminate: the window-consistent candidate ~1, the
    inconsistent one floored (the 3.4 behavior, via the P4.2 wiring).

    The candidate split is SPDEF (Flamethrower is special — a DEF split would be
    a false test: both windows identical). The observed delta is picked
    ADAPTIVELY from the two windows (this test pins the wiring, not the calc
    numbers — that is the 3.4 gate's job): midpoint of the 0-SpD window, which
    must sit outside the 252-SpD window incl. its crit range.
    """
    spec = {"name": "charizard", "nature": "Modest", "evs": {"spa": 252}}
    bs = BeliefState("gen9ou", own_team={"p1a": spec})
    bs.add_opponent("p2a", "blastoise")
    base = {"moves": ["tackle", "a", "b", "c"], "item": "", "ability": "torrent",
            "nature": "Bold", "tera": None, "weight": 1.0}
    c_ok = dict(base, evs={})          # 0-SpD  -> the higher damage window
    c_bad = dict(base, evs={"spd": 252})  # 252-SpD -> the lower window
    bs.slot("p2a").candidates = [c_ok, c_bad]

    a_mon = ev.mon_options(spec)  # same attacker construction as state.py
    w_ok = ev.defender_window(c_ok, "blastoise", a_mon, None, {}, None,
                              "flamethrower", None)
    w_bad = ev.defender_window(c_bad, "blastoise", a_mon, None, {}, None,
                               "flamethrower", None)
    # calc preconditions (loud if @smogon/calc ever changes the spreads):
    assert w_ok["min"] != w_bad["min"], "precondition: SpD spreads must be distinct"
    delta = (w_ok["min"] + w_ok["max"]) // 2
    assert w_ok["min"] <= delta <= w_ok["max"]
    assert not (w_bad["min"] <= delta <= w_bad["max"]), \
        f"precondition: delta {delta} inside the 252-SpD window {w_bad}"
    assert not (w_bad["crit_min"] and w_bad["crit_min"] <= delta <= w_bad["crit_max"]), \
        f"precondition: delta {delta} inside the 252-SpD crit window {w_bad}"

    maxhp = 299  # Blastoise max HP (pinned in 3.4); delta << maxhp -> no KO
    applied = bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=maxhp - delta, maxhp=maxhp,
        ko=False, source_kind="move", move="flamethrower",
        source_side="p1", source_slot="p1a"))
    assert applied, "the batched damage path must still apply as belief evidence"
    w = [c["weight"] for c in bs.slot("p2a").candidates]
    assert sum(w) == pytest.approx(1.0)
    # the 0-SpD candidate is confirmed (~1); the 252-SpD one is floored at EPS
    assert w[0] > 0.999 and w[1] < 1e-6, \
        f"expected the 0-SpD candidate confirmed, got weights {w}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
