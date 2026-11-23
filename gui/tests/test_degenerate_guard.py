"""Degenerate-training guard (``gui.m1_gate_main._degenerate_check``) — hermetic.

Pins each WARN rule (entropy collapse, KL blow-up, value-head collapse,
no-progress, flat yardstick) plus the healthy no-warning case. The function is
a pure predicate over recent metric rows, so this test needs NO torch and NO
battle engine. It is the "observable bug-catcher" the user asked for: a dead
net must produce a loud, greppable warning — not a silent garbage checkpoint.

Run:  python -m pytest gui/tests/test_degenerate_guard.py -v
"""
from __future__ import annotations

from gui import m1_gate_main as m1
from gui.m1_gate_main import _degenerate_check


def _healthy(**over):
    """A metric row that triggers NO warnings by default."""
    row = {
        "entropy": 1.5,
        "kl_approx": 0.1,
        "value_std": 0.4,
        "a1_wins": 2,
        "a1_losses": 8,
        "eval_wr": 0.2,
    }
    row.update(over)
    return row


def test_empty_history_no_warnings():
    assert _degenerate_check([]) == []


def test_healthy_row_no_warnings():
    assert _degenerate_check([_healthy()]) == []


def test_entropy_collapse_warns():
    warns = _degenerate_check([_healthy(entropy=0.1)])
    assert any("entropy collapse" in w for w in warns), warns


def test_kl_elevated_warns():
    warns = _degenerate_check([_healthy(kl_approx=0.6)])
    assert any("KL elevated" in w for w in warns), warns


def test_kl_hard_warns():
    warns = _degenerate_check([_healthy(kl_approx=1.5)])
    assert any("KL blow-up" in w for w in warns), warns


def test_kl_just_below_threshold_no_warn():
    # 0.49 is just under DEG_KL_WARN (0.50) -> no KL warning
    warns = _degenerate_check([_healthy(kl_approx=0.49)])
    assert not any("KL" in w for w in warns), warns


def test_value_head_collapse_warns():
    warns = _degenerate_check([_healthy(value_std=0.001)])
    assert any("value-head collapse" in w for w in warns), warns


def test_no_progress_five_zero_win_rounds_warns():
    rows = [_healthy(a1_wins=0, a1_losses=9)
            for _ in range(m1.DEG_NO_PROGRESS_ROUNDS)]
    warns = _degenerate_check(rows)
    assert any("no progress" in w for w in warns), warns


def test_no_progress_not_triggered_with_a_win():
    rows = [_healthy(a1_wins=0, a1_losses=9) for _ in range(4)]
    rows.append(_healthy(a1_wins=1, a1_losses=9))  # a win breaks the streak
    warns = _degenerate_check(rows)
    assert not any("no progress" in w for w in warns), warns


def test_flat_yardstick_three_zero_evals_warns():
    rows = [_healthy(eval_wr=0.0) for _ in range(m1.DEG_FLAT_YARDSTICK_EVALS)]
    warns = _degenerate_check(rows)
    assert any("flat yardstick" in w for w in warns), warns


def test_flat_yardstick_not_triggered_on_nonzero():
    rows = [_healthy(eval_wr=0.0) for _ in range(2)] + [_healthy(eval_wr=0.1)]
    warns = _degenerate_check(rows)
    assert not any("flat yardstick" in w for w in warns), warns


def test_nan_metrics_skipped_not_crash():
    # non-finite entropy/kl/value_std must be ignored (no false warn, no crash)
    warns = _degenerate_check([_healthy(
        entropy=float("nan"), kl_approx=float("nan"), value_std=float("nan"))])
    assert warns == [], warns


def test_thresholds_are_sane_ordering():
    # guard against accidental threshold inversions that would make the rules
    # vacuous (e.g. HARD below WARN) or over-eager.
    assert m1.DEG_KL_WARN < m1.DEG_KL_HARD
    assert m1.DEG_ENTROPY_FLOOR > 0.0
    assert m1.DEG_VALUE_STD_FLOOR > 0.0
    assert m1.DEG_NO_PROGRESS_ROUNDS >= 2
    assert m1.DEG_FLAT_YARDSTICK_EVALS >= 2
