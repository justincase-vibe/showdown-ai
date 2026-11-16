"""Step 3.7 gate — whole-battle fuzz bug-catcher.

Run from the project root:

    python -m pytest belief/tests/test_step3_7.py -v

Sub-step 3.7 (PLAN.md §3 / §3a): run N≈12 **seeded fork battles** (the harness
``scratch/step3_7_fuzz_battles.js`` emits a team-spec JSON header + the raw log
per battle), feed EVERY event through ``BeliefState.update()``, and after EVERY
update assert the hard belief invariants hold (no empty belief, weights sum to 1,
all weights > 0). ``Unknown`` events (lines the adapter could not map) are
counted and reported — the observable bug-catcher. A human-readable per-turn
report (top candidates / top move / entropy per p2 slot) is written to a
gitignored artifact (``belief/artifacts/fuzz_3_7_report.txt``).

The point is NOT to force every mechanic (3.4/3.6 unit-test those) — it is to
prove the belief UPDATE machinery stays well-formed over REAL multi-turn logs
(real switch-ins, KOs, re-switches, statuses, boosts) and to surface any unmapped
line (Unknown) or crash as a loud, locatable failure.
"""
from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from belief import events as E
from belief.log_events import LogAdapter
from belief.state import BeliefState

ROOT = Path(__file__).resolve().parents[2]  # belief/tests/x.py -> project root
FMT = "gen9bss3v3"
HARNESS = ROOT / "scratch" / "step3_7_fuzz_battles.js"
ARTIFACT = ROOT / "belief" / "artifacts" / "fuzz_3_7_report.txt"
MIN_BATTLES = 10            # plan range N≈10–20; the harness currently emits 12
WEIGHT_TOL = 1e-6


# ═══════════════════════════ harness I/O ═══════════════════════════

def _run_fuzz_battles():
    """Run the node harness; return ``[{'header': {...}, 'log': [lines]}, ...]``.

    The header is the SINGLE source of truth for the teams (the harness owns them),
    so the gate builds ``own_team`` without duplicating the team definition."""
    proc = subprocess.run(["node", str(HARNESS)], cwd=str(ROOT),
                          capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        raise AssertionError(f"fuzz harness failed (rc={proc.returncode})\n{proc.stderr}")
    battles = []
    cur = None
    for raw in proc.stdout.splitlines():
        s = raw.strip()
        if s.startswith("=== FUZZ BATTLE") and s.endswith("START ==="):
            cur = {"header": None, "log": []}
            battles.append(cur)
        elif s.startswith("=== FUZZ BATTLE") and s.endswith("END ==="):
            cur = None
        elif s.startswith("=== FUZZ DONE"):
            continue
        elif cur is not None and s.startswith("FUZZ_HEADER:"):
            cur["header"] = json.loads(s[len("FUZZ_HEADER:"):].strip())
        elif cur is not None and raw.startswith("|"):
            cur["log"].append(raw)
    return [b for b in battles if b["header"] is not None and b["log"]]


def _slot_specs(side: str, team) -> dict:
    """Team array (index = active order) -> ``{slot: spec}``; p1->p1a/b/c, p2->p2a/b/c."""
    prefix = side[:2]
    return {f"{prefix}{'abc'[i]}": spec for i, spec in enumerate(team)}


# ═══════════════════════════ report helpers ═══════════════════════════

def _entropy(weights) -> float:
    """Shannon entropy (log2) over the positive posterior masses — a belief spread
    measure: high = still uncertain, low = narrowed."""
    return -sum(w * math.log2(w) for w in weights if w > 0.0)


def _fmt_candidate(c: dict) -> str:
    moves = "/".join(c.get("moves", [])[:4]) or "-"
    return f"[{moves} | {c.get('item') or '-'} | {c.get('ability') or '-'}]"


def _assert_invariants(bs: BeliefState, where: str) -> None:
    """The hard belief invariants after every update: never-empty, normalized to
    1, every weight strictly positive (the EPS floor guarantees the last two)."""
    for slot in bs.slots:
        sb = bs.slot(slot)
        assert len(sb.candidates) >= 1, f"{where}: {slot} has an EMPTY belief"
        total = sum(c["weight"] for c in sb.candidates)
        assert abs(total - 1.0) < WEIGHT_TOL, f"{where}: {slot} weights sum to {total!r}"
        assert all(c["weight"] > 0.0 for c in sb.candidates), \
            f"{where}: {slot} has a non-positive weight"


def _run_battle(bs: BeliefState, battle: dict, bi: int):
    """Feed one battle's full event stream through ``update()``, asserting the
    invariants after each. Returns ``(unknown_events, report_lines)``."""
    events = LogAdapter().feed_lines(battle["log"])
    unknown = [e for e in events if isinstance(e, E.Unknown)]
    report_lines = []
    for ev in events:
        if isinstance(ev, E.Unknown):
            continue  # counted + reported; no evidence to apply
        try:
            bs.update(ev)
        except Exception as e:  # a crash is exactly what the bug-catcher must surface
            pytest.fail(f"battle {bi}: update({ev.type}) raised "
                        f"{type(e).__name__}: {e}")
        _assert_invariants(bs, f"battle {bi} after {ev.type}")
        if isinstance(ev, E.Turn):
            for slot in bs.slots:
                sb = bs.slot(slot)
                top = max(sb.candidates, key=lambda c: c["weight"])
                mp = sb.move_probabilities()
                top_move = max(mp.items(), key=lambda kv: kv[1])[0] if mp else "-"
                report_lines.append(
                    f"  b{bi} turn {ev.n:>2} {slot} [{sb.species}] "
                    f"top={top['weight']:.3f} {_fmt_candidate(top)} "
                    f"| topmove={top_move}({mp.get(top_move, 0.0):.3f}) "
                    f"| entropy={_entropy(sb.posterior):.3f}")
    return unknown, report_lines


# ═══════════════════════════ the gate ═══════════════════════════

def test_fuzz_invariants_and_report():
    battles = _run_fuzz_battles()
    assert len(battles) >= MIN_BATTLES, \
        f"expected >= {MIN_BATTLES} battles, got {len(battles)}"

    all_report = [f"=== STEP 3.7 FUZZ — {len(battles)} seeded BSS battles ===",
                  "(top candidate per p2 slot, its top move, and posterior entropy "
                  "per turn; Unknown events = the observable bug-catcher)"]
    total_unknown = 0
    any_p2 = False
    for bi, battle in enumerate(battles):
        own = _slot_specs("p1", battle["header"]["p1"])
        bs = BeliefState(FMT, own_team=own)
        unknown, report_lines = _run_battle(bs, battle, bi)
        total_unknown += len(unknown)
        any_p2 = any_p2 or bool(report_lines)
        all_report.append(f"\n--- battle {bi}  seed={battle['header']['seed']} ---")
        all_report.extend(report_lines or ["  (no p2 slot registered)"])
        for e in unknown:
            all_report.append(f"  !! UNKNOWN line: {e.raw!r}")

    # write the artifact (gitignored, regenerated each run)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text("\n".join(all_report) + "\n", encoding="utf-8")
    assert ARTIFACT.exists() and ARTIFACT.stat().st_size > 0, "artifact not written"

    # the fuzz actually exercised belief state (switch-ins happened)
    assert any_p2, "no p2 slot was ever registered — the fuzz exercised no belief"
    # the report is non-trivial (real per-turn entries, not just headers)
    per_turn = [l for l in all_report if "entropy=" in l]
    assert len(per_turn) >= MIN_BATTLES, f"report too thin: {len(per_turn)} per-turn lines"

    print(f"\n=== STEP 3.7 FUZZ: {len(battles)} battles, "
          f"{total_unknown} Unknown event(s) ===")
    print(f"  artifact: {ARTIFACT}")
    for l in all_report:
        if "!! UNKNOWN" in l:
            print("  ", l)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
