"""C4 gate eval (tools/gate_eval.py): hermetic pins (no server, no torch).

The live run (``run_eval_battles`` via the shared battle loop) needs a
server pool — that is the CPU/GPU smoke, run manually. These pin the PURE
parts: the Wilson lower-CI math, the decision-rule verdict, and the
per-opponent pooling. The reference Wilson values were verified by hand
against the closed-form formula (z=1.959963984540054).

Run:  python -m pytest tools/tests/test_gate_eval.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.gate_eval import (  # noqa: E402
    GATE_CONTINUE, GATE_PASS, PRIMARY_OPP, apply_validity, build_validity,
    gate_verdict, min_mode_bound, pool_rows, wilson_lower, wilson_upper,
    _entry_stats, _parse_subjects,
)


# ── wilson_lower ────────────────────────────────────────────────────────────

def test_wilson_no_data_is_nan():
    assert math.isnan(wilson_lower(0, 0))
    assert math.isnan(wilson_lower(5, -1))


def test_wilson_zero_wins_is_zero():
    assert wilson_lower(0, 10) == 0.0


def test_wilson_extreme_small_n():
    # 1/10 falls below zero before clipping -> 0.0 (Wilson is conservative)
    assert wilson_lower(1, 10) == 0.0
    # verified closed-form values (z = 1.959963984540054)
    assert abs(wilson_lower(5, 10) - 0.007366) < 1e-4
    assert abs(wilson_lower(9, 10) - 0.330068) < 1e-4
    assert abs(wilson_lower(10, 10) - 0.422416) < 1e-4


def test_wilson_gate_scale_values():
    # the values the gate verdict actually lives on (n ~ 400)
    assert abs(wilson_lower(200, 400) - 0.393211) < 1e-4
    assert abs(wilson_lower(280, 400) - 0.593088) < 1e-4
    # monotonic in wins
    assert wilson_lower(150, 407) < wilson_lower(200, 407) < wilson_lower(280, 407)


def test_wilson_upper_symmetric():
    assert abs(wilson_upper(200, 400) - (1.0 - 0.393211)) < 1e-4
    assert wilson_upper(10, 10) == 1.0
    # 0/10: the upper bound is POSITIVE (true p could be > 0) — the classic
    # reason Wilson beats the naive interval
    assert 0.0 < wilson_upper(0, 10) < 1.0
    assert math.isnan(wilson_upper(5, 0))
    # interval brackets the point estimate
    assert wilson_lower(70, 100) < 0.70 < wilson_upper(70, 100)


# ── gate_verdict (the M1 decision rule) ─────────────────────────────────────

def test_verdict_thresholds():
    assert GATE_PASS == 0.40
    assert GATE_CONTINUE == 0.25
    assert gate_verdict(0.40) == "PASS"
    assert gate_verdict(0.55) == "PASS"
    assert gate_verdict(0.3999) == "CONTINUE"
    assert gate_verdict(0.25) == "CONTINUE"
    assert gate_verdict(0.2499) == "STOP"
    assert gate_verdict(0.0) == "STOP"


def test_verdict_no_data():
    assert gate_verdict(float("nan")) == "NO_DATA"
    assert gate_verdict(None) == "NO_DATA"


# ── per-mode stats + pooled verdict (C1-C5 spec) ────────────────────────────

def test_primary_opp_is_plain_strategy():
    assert PRIMARY_OPP == "strategy"


def test_min_mode_bound_takes_worst():
    assert abs(min_mode_bound({"greedy": 0.50, "sampled": 0.42}) - 0.42) < 1e-12
    assert abs(min_mode_bound({"greedy": 0.30}) - 0.30) < 1e-12


def test_min_mode_bound_empty_or_nan():
    assert math.isnan(min_mode_bound({}))
    assert math.isnan(min_mode_bound({"greedy": float("nan")}))


def test_min_mode_bound_ignores_nan_entries():
    # one mode errored (nan) -> the verdict falls back to the real bound
    assert abs(min_mode_bound({"greedy": 0.41, "sampled": float("nan")})
               - 0.41) < 1e-12


def _row(w, l, d=0):
    return {"eval_w": w, "eval_l": l, "eval_d": d, "eval_crashed": 0,
            "eval_wr": w / (w + l + d) if (w + l + d) else float("nan")}


def test_entry_stats_per_mode_and_pooled():
    pairs = {
        "strategy": {
            "greedy": _row(180, 20),
            "sampled": _row(120, 80),
        },
        "strategy_posterior": {"greedy": _row(90, 110)},
    }
    st = _entry_stats(pairs)
    assert set(st["per_mode_lower_ci"]["strategy"]) == {"greedy", "sampled"}
    assert st["per_mode_lower_ci"]["strategy"]["greedy"] \
        > st["per_mode_lower_ci"]["strategy"]["sampled"]
    p = st["pooled"]["strategy"]
    assert p["wins"] == 300 and p["n"] == 400
    assert 0.0 <= p["lower_ci_95"] < 0.75 < p["upper_ci_95"] <= 1.0
    # verdict bound = WORST pooled bound of strategy + strategy_posterior
    assert st["verdict_bound"] == st["pooled"]["strategy_posterior"]["lower_ci_95"]
    assert st["base_verdict"] == gate_verdict(st["verdict_bound"])
    assert st["mode_diff_warnings"]  # 0.45 vs 0.60 differ by 15 pts > 10


def test_entry_stats_mode_diff_warning_threshold():
    pairs = {"strategy": {"greedy": _row(55, 45), "sampled": _row(60, 40)}}
    assert _entry_stats(pairs)["mode_diff_warnings"] == []


def test_entry_stats_skips_error_rows():
    pairs = {"strategy": {"greedy": _row(50, 50), "sampled": {"error": "boom"}}}
    st = _entry_stats(pairs)
    assert "strategy" in st["pooled"]
    assert set(st["per_mode_lower_ci"]["strategy"]) == {"greedy"}


# ── validity rules (C1/C2 based) ─────────────────────────────────────────────

def _entries_with_c1(c1_wr_n, c2_wr_n=None):
    def _pooled(w, n):
        return {"wins": w, "losses": n - w, "draws": 0, "crashed": 0,
                "n": n, "win_rate": w / n,
                "lower_ci_95": wilson_lower(w, n),
                "upper_ci_95": wilson_upper(w, n)}
    entries = {}
    if c1_wr_n is not None:
        entries["control:random_init"] = {
            "stats": {"pooled": {"strategy": _pooled(*c1_wr_n)}}}
    if c2_wr_n is not None:
        entries["control:random"] = {
            "stats": {"pooled": {"strategy": _pooled(*c2_wr_n)}}}
    return entries


def test_validity_c2_too_strong_invalidates():
    # C2 random bot beats strategy 30% > 25% -> YARDSTICK_INVALID
    v = build_validity(_entries_with_c1((80, 400), (120, 400)))
    assert v["yardstick_invalid"] is True
    assert any("C2" in r for r in v["reasons"])


def test_validity_c1_upper_bound_invalidates():
    # C1 random-init at 50% (n=400): upper bound ~0.58 >= 0.40 -> invalid
    v = build_validity(_entries_with_c1((200, 400)))
    assert v["yardstick_invalid"] is True
    assert any("UPPER" in r for r in v["reasons"])


def test_validity_clean_when_floor_is_weak():
    # floor at 20% (n=400): upper bound ~0.27 < 0.40, C2 20% <= 25% -> valid
    v = build_validity(_entries_with_c1((80, 400), (80, 400)))
    assert v["yardstick_invalid"] is False
    assert v["c1"] is not None and v["c2"] is not None


def test_apply_validity_caps_pass_when_intervals_overlap():
    # C1 floor at 20% (n=400, upper ~0.27); net at 32.5% (n=400, lower ~0.22):
    # margin 12.5 pts >= 10 BUT the 95% intervals overlap -> cap at CONTINUE
    v = build_validity(_entries_with_c1((80, 400), (80, 400)))
    net = {"win_rate": 130 / 400, "lower_ci_95": wilson_lower(130, 400),
           "upper_ci_95": wilson_upper(130, 400)}
    out = apply_validity("PASS", net, v)
    assert out.startswith("CONTINUE (capped")


def test_apply_validity_pass_survives_with_clear_margin():
    # C1 floor at 20% (n=400, upper ~0.27); net at 50% (n=400, lower ~0.39):
    # margin 30 pts and NON-overlapping intervals -> PASS stands
    v = build_validity(_entries_with_c1((80, 400), (80, 400)))
    net = {"win_rate": 0.5, "lower_ci_95": wilson_lower(200, 400),
           "upper_ci_95": wilson_upper(200, 400)}
    assert apply_validity("PASS", net, v) == "PASS"


def test_apply_validity_invalid_yardstick_overrides():
    v = build_validity(_entries_with_c1((200, 400)))
    net = {"win_rate": 0.9, "lower_ci_95": 0.8, "upper_ci_95": 0.95}
    assert apply_validity("PASS", net, v).startswith("YARDSTICK_INVALID")


def test_c1_falls_back_to_c2_row():
    # no random_init entry; the C2 row (identical net here) fills C1
    v = build_validity(_entries_with_c1(None, (80, 400)))
    assert v["c1"] is not None and v["c2"] is not None


def test_parse_subjects_plain_and_override():
    out = _parse_subjects("random_init,heuristic,strategy:strategy,strategy_posterior:strategy")
    assert out == [("random_init", None), ("heuristic", None),
                   ("strategy", ["strategy"]),
                   ("strategy_posterior", ["strategy"])]
    assert _parse_subjects(None) == []


# ── pool_rows (reporting-only pool of play modes for one opponent) ──────────

def test_pool_rows_sums_and_never_counts_draws_as_wins():
    rows = [
        {"eval_w": 100, "eval_l": 100, "eval_d": 5, "eval_crashed": 1},
        {"eval_w": 50, "eval_l": 150, "eval_d": 2, "eval_crashed": 0},
    ]
    p = pool_rows(rows)
    assert p["wins"] == 150 and p["losses"] == 250 and p["draws"] == 7
    assert p["n"] == 407 and p["crashed"] == 1
    assert abs(p["win_rate"] - 150 / 407) < 1e-12
    assert 0.0 <= p["lower_ci_95"] <= 1.0


def test_pool_rows_empty():
    p = pool_rows([])
    assert p["n"] == 0
    assert math.isnan(p["win_rate"])
    assert math.isnan(p["lower_ci_95"])


# ── summarize_eval_records (the shared row contract, refactored out of the
#    trainer) ────────────────────────────────────────────────────────────────

def test_summarize_eval_records_contract():
    from gui.m1_gate_main import summarize_eval_records
    records = [
        {"a1_won": True, "a2_won": None, "crashed": False},   # win
        {"a1_won": True, "a2_won": None, "crashed": False},   # win
        {"a1_won": None, "a2_won": True, "crashed": False},   # loss
        {"a1_won": None, "a2_won": None, "crashed": False},   # draw (stall)
        {"a1_won": True, "a2_won": True, "crashed": False},   # double K.O. = draw
        {"a1_won": None, "a2_won": None, "crashed": True},    # crashed
    ]
    row = summarize_eval_records(records)
    assert row["eval_w"] == 2 and row["eval_l"] == 1
    assert row["eval_d"] == 2 and row["eval_crashed"] == 1
    assert abs(row["eval_wr"] - 2 / 5) < 1e-12


def test_summarize_eval_records_empty_is_nan():
    from gui.m1_gate_main import summarize_eval_records
    row = summarize_eval_records([])
    assert row["eval_w"] == 0 and row["eval_crashed"] == 0
    assert math.isnan(row["eval_wr"])


# ── the C3 wrapper must keep its exact row contract (6.6a) ──────────────────

def test_eval_round_m1_contract_shape():
    """_eval_round_m1 returns eval_* keys ONLY (plus eval_opp) — the additive
    contract the trainer's metrics row depends on. Pinned with fakes; no
    server, no torch (run_eval_battles is monkeypatched)."""
    import gui.m1_gate_main as m1

    fake_row = {"eval_w": 3, "eval_l": 1, "eval_d": 0, "eval_crashed": 0,
                "eval_wr": 0.75, "eval_battles": 4}
    calls = {}

    def fake_run_eval_battles(net, opp_ref, fmt, vocabs, device, n_battles,
                              max_steps, n_servers, per_server, port_base,
                              seed_base, deterministic=True, battle_hook=None,
                              stop_check=None, label="eval"):
        calls.update(dict(net=net, opp_ref=opp_ref, n_battles=n_battles,
                          seed_base=seed_base, deterministic=deterministic))
        return dict(fake_row)

    orig = m1.run_eval_battles
    m1.run_eval_battles = fake_run_eval_battles
    try:
        class _FakePool(dict):
            def names(self):
                return list(self.keys())

        class _FakeTrainer:
            net = "NET"
            round = 3

        pool = _FakePool()
        pool["strategy"] = "STRATEGY_REF"
        row = m1._eval_round_m1(_FakeTrainer(), pool, "gen9ou", "vocabs",
                                "cpu", "strategy", 4, 200, 6, 2, 49980,
                                None, lambda: False)
        # exact contract: additive eval_* keys + eval_opp, nothing else
        assert set(row.keys()) == {"eval_opp", "eval_w", "eval_l", "eval_d",
                                   "eval_crashed", "eval_wr", "eval_battles"}
        assert row["eval_opp"] == "strategy"
        # the C3 wrapper stays GREEDY and uses the round-disjoint seed space
        assert calls["deterministic"] is True
        assert calls["seed_base"] == 1_000_000 + 3 * 1000
        assert calls["n_battles"] == 4
    finally:
        m1.run_eval_battles = orig
