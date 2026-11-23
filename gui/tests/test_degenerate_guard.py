"""Degenerate-training guard (``gui.m1_gate_main._degenerate_check``) — hermetic.

Pins each WARN rule and the healthy no-warning case, plus the KEY regression:
a young net's naturally-volatile early rounds (high KL step, low value spread)
must NOT cry-wolf — only *sustained* soft signals (or a single severe step)
fire. The function is a pure predicate over recent metric rows, so this test
needs NO torch and NO battle engine. It is the "observable bug-catcher": a dead
net must produce a loud, greppable warning — not a silent garbage checkpoint,
and not a false alarm at round 2.

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


# ── entropy (immediate) ──────────────────────────────────────────────────────

def test_entropy_collapse_warns():
    warns = _degenerate_check([_healthy(entropy=0.1)])
    assert any("entropy collapse" in w for w in warns), warns


# ── KL: severe single-step is immediate, soft elevation must persist ─────────

def test_kl_blowup_single_round_warns():
    warns = _degenerate_check([_healthy(kl_approx=1.5)])
    assert any("KL blow-up" in w for w in warns), warns


def test_kl_elevated_single_round_no_warn():
    # A single 0.6 KL step is a big update, not degeneration -> no alarm.
    warns = _degenerate_check([_healthy(kl_approx=0.6)])
    assert not any("KL" in w for w in warns), warns


def test_kl_sustained_two_rounds_warns():
    rows = [_healthy(kl_approx=0.6), _healthy(kl_approx=0.7)]
    warns = _degenerate_check(rows)
    assert any("KL sustained" in w for w in warns), warns


def test_kl_just_below_threshold_no_warn():
    warns = _degenerate_check([_healthy(kl_approx=0.49), _healthy(kl_approx=0.49)])
    assert not any("KL" in w for w in warns), warns


# ── value-head: soft collapse must persist (warmup is naturally low) ─────────

def test_value_head_collapse_single_round_no_warn():
    # A single low-spread round at round 2 is warmup, not collapse -> no alarm.
    warns = _degenerate_check([_healthy(value_std=0.015)])
    assert not any("value-head" in w for w in warns), warns


def test_value_head_collapse_sustained_warns():
    rows = [_healthy(value_std=0.01), _healthy(value_std=0.012)]
    warns = _degenerate_check(rows)
    assert any("value-head collapse" in w for w in warns), warns


# ── no progress / flat yardstick (window-based) ──────────────────────────────

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


# ── robustness ───────────────────────────────────────────────────────────────

def test_nan_metrics_skipped_not_crash():
    warns = _degenerate_check([_healthy(
        entropy=float("nan"), kl_approx=float("nan"), value_std=float("nan"))])
    assert warns == [], warns


def test_real_round2_case_no_false_positive():
    """The EXACT 2-round GPU test that fired a false alarm before: the net was
    IMPROVING (eval 0.1 -> 0.3, entropy healthy 1.40 -> 1.48), so rounds 1-2
    must produce NO warning under the sustained-signal design."""
    rows = [
        {"entropy": 1.403, "kl_approx": 0.355, "value_std": 0.0242,
         "a1_wins": 1, "a1_losses": 9, "eval_wr": 0.1},
        {"entropy": 1.480, "kl_approx": 0.797, "value_std": 0.0152,
         "a1_wins": 2, "a1_losses": 7, "eval_wr": 0.3},
    ]
    assert _degenerate_check(rows) == [], _degenerate_check(rows)


def test_thresholds_are_sane_ordering():
    # guard against accidental inversions that would make rules vacuous or
    # over-eager.
    assert m1.DEG_KL_WARN < m1.DEG_KL_HARD
    assert m1.DEG_ENTROPY_FLOOR > 0.0
    assert m1.DEG_VALUE_STD_FLOOR > 0.0
    assert m1.DEG_SUSTAINED >= 2
    assert m1.DEG_NO_PROGRESS_ROUNDS >= 2
    assert m1.DEG_FLAT_YARDSTICK_EVALS >= 2
