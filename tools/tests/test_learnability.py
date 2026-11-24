"""Hermetic tests for tools/learnability_test.py (the (a) learnability gate).

Only the PURE decision logic is tested here (stop rules + verdict + contract
constants) — the battle orchestration itself is validated by the --smoke run
and the shared loops' own test suites (tools/tests/test_gate_eval.py for the
eval loop, net/tests/test_ppo.py for the PPO core under test). No server, no
GPU, fast.
"""
from __future__ import annotations

from gui.m1_gate_main import DEG_ENTROPY_FLOOR, DEG_KL_HARD
from tools.learnability_test import (EVAL_MODE_OFFSET, EVAL_PORT_BASE, PASS_WR,
                                     SWITCH_SHARE_FLOOR, SWITCH_SHARE_STREAK,
                                     TRAIN_PORT_BASE, check_stop_rules,
                                     decide_verdict)


# ── stop rules ────────────────────────────────────────────────────────────────
def test_stop_entropy_collapse_is_immediate():
    reason = check_stop_rules({"entropy": DEG_ENTROPY_FLOOR - 0.05,
                               "kl_approx": 0.0}, float("nan"), 0)
    assert reason is not None and "entropy" in reason
    # above the floor -> no stop
    assert check_stop_rules({"entropy": DEG_ENTROPY_FLOOR + 0.1}, float("nan"), 0) is None


def test_stop_kl_blowup_is_immediate():
    reason = check_stop_rules({"entropy": 1.0, "kl_approx": DEG_KL_HARD + 0.2},
                              0.5, 0)
    assert reason is not None and "KL" in reason
    assert check_stop_rules({"entropy": 1.0, "kl_approx": DEG_KL_HARD * 0.5},
                            0.5, 0) is None


def test_stop_switch_share_needs_full_streak():
    low = SWITCH_SHARE_FLOOR - 0.01
    # current round low + (STREAK-1) prior low rounds = STREAK low rounds -> stop
    reason = check_stop_rules({}, low, SWITCH_SHARE_STREAK - 1)
    assert reason is not None and "switch" in reason
    # one short of the streak -> no stop yet
    assert check_stop_rules({}, low, SWITCH_SHARE_STREAK - 2) is None
    # healthy share never stops, regardless of prior streak bookkeeping
    assert check_stop_rules({}, 0.5, 99) is None
    # a round with no actions at all (nan) never triggers the share rule
    assert check_stop_rules({}, float("nan"), SWITCH_SHARE_STREAK + 5) is None


def test_stop_rules_tolerate_skipped_update_metrics():
    # a round whose PPO update was skipped has no entropy/kl keys at all
    assert check_stop_rules({}, 0.5, 0) is None
    assert check_stop_rules({"skipped": True, "reason": "too_few (1)"},
                            float("nan"), 0) is None


def test_stop_rules_non_finite_metrics_are_ignored():
    assert check_stop_rules({"entropy": float("nan"), "kl_approx": float("inf")},
                            float("nan"), 9) is None


# ── verdict ───────────────────────────────────────────────────────────────────
def _evals(rows):
    """rows: list of (round, sampled_wr, greedy_wr)"""
    out = []
    for rnd, swr, gwr in rows:
        out.append({"round": rnd,
                    "sampled": {"eval_wr": swr, "eval_w": 9, "eval_l": 1,
                                "eval_d": 0},
                    "greedy": {"eval_wr": gwr, "eval_w": 9, "eval_l": 1,
                               "eval_d": 0}})
    return out


def test_verdict_pass_when_sampled_reaches_bar():
    v = decide_verdict(_evals([(0, 0.50, 0.55), (2, 0.70, 0.72),
                               (4, 0.90, 0.88)]), max_rounds=15)
    assert v["verdict"] == "PASS"
    assert v["best_sampled"]["round"] == 4 and v["best_sampled"]["wr"] == 0.90


def test_verdict_fail_when_bar_never_reached():
    v = decide_verdict(_evals([(0, 0.50, 0.50), (2, 0.55, 0.55),
                               (4, 0.60, 0.61)]), max_rounds=15)
    assert v["verdict"] == "FAIL"


def test_verdict_stopped_overrides_fail():
    v = decide_verdict(_evals([(0, 0.50, 0.50), (2, 0.52, 0.52)]), max_rounds=15,
                       stopped=True, stop_reason="entropy collapse (x)")
    assert v["verdict"] == "STOPPED_DEGENERATE"
    assert v["stop_reason"] == "entropy collapse (x)"


def test_verdict_sampled_is_the_decision_mode():
    # greedy at 0.95 does NOT pass the gate if sampled is 0.70 — sampled is
    # the net's natural play mode and the spec's criterion (documented in
    # decide_verdict).
    v = decide_verdict(_evals([(4, 0.70, 0.95)]), max_rounds=15)
    assert v["verdict"] == "FAIL"
    assert v["best_sampled"]["wr"] == 0.70


def test_verdict_ignores_rounds_beyond_max():
    v = decide_verdict(_evals([(16, 0.99, 0.99)]), max_rounds=15)
    assert v["verdict"] == "FAIL" and v["best_sampled"] is None


def test_verdict_skips_non_finite_rows():
    v = decide_verdict([{"round": 2,
                         "sampled": {"eval_wr": float("nan")},
                         "greedy": {"eval_wr": 0.9}}], max_rounds=15)
    assert v["verdict"] == "FAIL" and v["best_sampled"] is None


# ── contract constants ────────────────────────────────────────────────────────
def test_contract_constants_match_the_spec():
    assert PASS_WR == 0.85                    # spec: "exceed 85%"
    assert SWITCH_SHARE_FLOOR == 0.03         # M1 decision: "< 3%"
    assert SWITCH_SHARE_STREAK == 3           # M1 decision: "for 3 rounds"
    # thresholds are the M1 gate guard's own constants (no drift)
    assert DEG_ENTROPY_FLOOR > 0.0 and DEG_KL_HARD > DEG_ENTROPY_FLOOR
    # the eval pool must sit on a DISJOINT port base (the training pool stays
    # live during evals — a collision would kill the eval's servers)
    assert EVAL_PORT_BASE != TRAIN_PORT_BASE
    assert EVAL_PORT_BASE - TRAIN_PORT_BASE >= 4   # >= max servers in play
    # sampled/greedy eval seeds must stay disjoint for the default eval size
    assert EVAL_MODE_OFFSET > 100
