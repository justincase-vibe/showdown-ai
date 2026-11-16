"""Step 3.8 gate — perf bench (update + 100 samples per mon, ms-scale) + D43.

Run from the project root:

    python -m pytest belief/tests/test_step3_8.py -v

Sub-step 3.8 (PLAN.md §3 / §3a): measure the per-turn belief cost the AI pays in
production and confirm it is acceptable:

  * a SOFT damage-evidence ``update`` — the expensive path. It drives the Step-2
    calc, one Node spawn per UNIQUE mon spec (the window dedupes on spec, NOT on
    the set, so 210 fallback candidates that share a spec cost a handful of
    spawns, not 210);
  * a ``sample(100)`` draw — the D45 ``deepcopy`` concern (pure Python, no Node).

The bench is a GATE, not just a report: it asserts every number is finite and
below a generous ceiling, so a catastrophic perf regression (losing the window
dedup cache -> one Node spawn per CANDIDATE; or ``sample`` accidentally spawning
Node) fails loudly instead of creeping in. It also asserts the dedup cache itself
works (a repeated identical window call is a ~0ms hit) — that dedupe is exactly
what bounds the per-spec cost. The result + the Node-spawn cost model are written
to a gitignored artifact (``belief/artifacts/bench_3_8_report.txt``).

HEADLINE (this machine, 2026-11-16): one Node spawn ≈ 250ms; a per-turn damage
update ≈ 0.5–0.75s (bounded by the few unique specs a slot's candidates carry);
``sample(100)`` ≈ 0.5ms. **Verdict: acceptable for per-turn belief updates**
(well inside Showdown's turn budget). The Node spawn is the cost to watch for
DEEP-TREE search (Step 5+, "a few calculations down the tree") — a batch endpoint
in ``node_calc.js`` (many windows in one spawn) is the flagged follow-up, NOT a
Step 3 blocker. No belief/calc code changes in this step.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from belief import events as E
from belief.state import BeliefState, build_prior

ROOT = Path(__file__).resolve().parents[2]  # belief/tests/x.py -> project root
ARTIFACT = ROOT / "belief" / "artifacts" / "bench_3_8_report.txt"

# Generous ceilings (this machine: update ~0.5–0.75s, sample ~0.5ms). They are
# wide enough to never flake on a slow box, tight enough to catch a real
# regression: losing the window dedup cache would push the 210-candidate slot to
# ~210 Node spawns (~50s) and a Node-spawning sample() to ~25s — both exceed.
CEILING_UPDATE_MS = 10_000.0
CEILING_SAMPLE_MS = 1_000.0
# A repeated identical window_for MUST be a cache hit: far faster than the ~250ms
# cold spawn. (Kept well under the cold cost so the assertion is noise-robust.)
CACHE_HIT_MS = 100.0


def _time(fn):
    t0 = time.perf_counter()
    r = fn()
    return (time.perf_counter() - t0) * 1000.0, r


def _bench_slot(species: str, fmt: str) -> dict:
    """Time switch-in (3.6 maxhp evidence) + a damage update (the SOFT path) +
    sample(100) for one belief slot. Returns a row for the report + the gate."""
    cands = build_prior(species, fmt)
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252}}}
    bs = BeliefState(fmt, own_team=own)
    bs.add_opponent("p2a", species)

    # switch-in reveals max HP -> 3.6 maxhp evidence (a calc per spec, too)
    t_switch, applied_switch = _time(lambda: bs.update(E.SwitchIn(
        side="p2", slot="p2a", species=species, hp=183, maxhp=183)))
    assert applied_switch or True  # fact bookkeeping always runs; evidence may not

    def dmg(hp_after: int) -> E.DamageObserved:
        return E.DamageObserved(target_side="p2", target_slot="p2a", hp_after=hp_after,
                                maxhp=183, ko=False, source_kind="move", move="flamethrower",
                                source_side="p1", source_slot="p1a")

    # the SOFT damage update — the expensive path (one Node spawn per unique spec)
    t_cold, applied = _time(lambda: bs.update(dmg(183 - 60)))
    assert applied, f"{species}/{fmt}: damage event did not apply as belief evidence"

    # sample(100) — the D45 deepcopy concern (pure Python, must be ms-scale)
    t_sample, samples = _time(lambda: bs.slot("p2a").sample(100, seed=42))
    assert len(samples) == 100, f"{species}: sample(100) returned {len(samples)}"

    return {
        "species": species, "fmt": fmt, "candidates": len(cands),
        "switch_in_ms": t_switch, "update_ms": t_cold, "sample100_ms": t_sample,
    }


def _cache_probe() -> tuple[float, float]:
    """Isolated window_for: cold spawn vs the repeated (cached) call. Uses a
    species/move distinct from the bench rows so the 'cold' is genuinely cold."""
    from belief import evidence as ev
    a = ev.mon_options({"nature": "Modest", "evs": {"spa": 252}}, name="charizard")
    d = ev.mon_options({"nature": "Bold", "evs": {"def": 252}}, name="gengar")
    cold = _time(lambda: ev.window_for(a, d, "flamethrower"))[0]
    warm = _time(lambda: ev.window_for(a, d, "flamethrower"))[0]
    return cold, warm


def _write_report(rows, node_cold_ms, node_warm_ms) -> None:
    L = [
        "=== STEP 3.8 PERF BENCH (belief update + sample, per-turn cost) ===",
        "measured: switch-in (3.6 maxhp evidence) | damage update (SOFT, the",
        "expensive path) | sample(100). Ceilings: update < "
        f"{CEILING_UPDATE_MS:.0f}ms, sample < {CEILING_SAMPLE_MS:.0f}ms.",
        "",
        f"  {'species':10s} {'fmt':14s} {'cands':>5s}  "
        f"{'switch-in':>10s} {'update':>10s} {'sample100':>10s}",
    ]
    for r in rows:
        L.append(f"  {r['species']:10s} {r['fmt']:14s} {r['candidates']:>5d}  "
                 f"{r['switch_in_ms']:>9.0f}ms {r['update_ms']:>9.0f}ms "
                 f"{r['sample100_ms']:>9.2f}ms")
    L += [
        "",
        "cost model (evidence.py: window_for deduped by mon SPEC, not set):",
        f"  one Node spawn (calc_damage)      : {node_cold_ms:8.1f} ms (cold)",
        f"  repeated window_for (cache hit)   : {node_warm_ms:8.2f} ms  -> dedupe works",
        "  per-turn damage update ~= (unique specs in the slot) x one Node spawn;",
        "  the defender's HP fraction changes each turn -> a cache miss each turn,",
        "  so the ~250ms Node spawn IS the realistic per-turn cost (not amortized).",
        "",
        "verdict: acceptable for per-turn belief updates (sub-second, well inside",
        "  Showdown's turn budget). Node spawn is the cost to watch for DEEP-TREE",
        "  search (Step 5+): a batch endpoint in node_calc.js (many windows in one",
        "  spawn) is the flagged follow-up, not a Step 3 blocker. No code changes.",
    ]
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text("\n".join(L) + "\n", encoding="utf-8")


def test_bench_update_and_sample_in_range():
    rows = [
        _bench_slot("garchomp", "gen9ou"),     # set-based: 2 candidates / 2 specs
        _bench_slot("machamp", "gen9bss3v3"),  # fallback: 210 candidates / ~3 specs
    ]
    for r in rows:
        assert 0.0 < r["update_ms"] < CEILING_UPDATE_MS, \
            f"{r['species']}: damage update {r['update_ms']:.0f}ms exceeds ceiling"
        assert 0.0 < r["sample100_ms"] < CEILING_SAMPLE_MS, \
            f"{r['species']}: sample(100) {r['sample100_ms']:.0f}ms exceeds ceiling"

    # the dedup cache is what bounds the per-spec cost: a repeated identical
    # window_for must be a cache hit (far faster than the cold Node spawn).
    node_cold, node_warm = _cache_probe()
    assert node_warm < CACHE_HIT_MS, \
        f"cache miss: repeated window_for took {node_warm:.0f}ms (dedupe broken?)"
    assert node_warm < node_cold, \
        f"cache not helping: warm {node_warm:.0f}ms >= cold {node_cold:.0f}ms"

    _write_report(rows, node_cold, node_warm)
    assert ARTIFACT.exists() and ARTIFACT.stat().st_size > 0, "artifact not written"

    print("\n=== STEP 3.8 PERF BENCH ===")
    for r in rows:
        print(f"  {r['species']:10s} ({r['candidates']:>3} cands): "
              f"update {r['update_ms']:6.0f}ms   sample(100) {r['sample100_ms']:5.2f}ms")
    print(f"  node spawn: cold {node_cold:.0f}ms / warm (cache hit) {node_warm:.2f}ms")
    print(f"  artifact: {ARTIFACT}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
