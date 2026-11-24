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
    GATE_CONTINUE, GATE_PASS, gate_verdict, pool_rows, wilson_lower,
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


# ── pool_rows (both play modes pooled for one opponent) ─────────────────────

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
