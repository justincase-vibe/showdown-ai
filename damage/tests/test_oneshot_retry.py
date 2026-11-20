"""B — the one-shot Node fallback retries a TRANSIENT spawn failure exactly once.

The 3200-battle league crash (0xC0000409) is spawn churn; the pool (C) is the
primary path, but the one-shot spawn is the FALLBACK (POKE_CALC_ONESHOT, or a
pool that keeps dying). Before this fix it only retried a *timeout*; a *crash*
(non-zero exit) on the first attempt raised immediately. Damage calc is a pure
function, so we now retry ONCE on a timeout OR a crash, while a genuine calc
ERROR (exit-0 JSON with an ``error`` field, e.g. an illegal species) is NOT
retried and raises exactly once — the one-shot contract, unchanged.

Run from the project root:

    python -m pytest damage/tests/test_oneshot_retry.py -v
"""
from __future__ import annotations

import json
import subprocess as _real_sub

import pytest

from damage import calc

GARCH = {"name": "Garchomp", "nature": "Adamant", "evs": {"atk": 252}}
TITAN = {"name": "Tyranitar", "nature": "Bold", "evs": {"def": 252}}
REQ = {"type": "damage", "gen": 9, "attacker": GARCH, "defender": TITAN,
       "move": {"name": "earthquake"}, "field": None}
GOOD = json.dumps({"min": 100, "max": 120})


def _cp(returncode: int, stdout: str = "", stderr: str = ""):
    return _real_sub.CompletedProcess(args=["node"], returncode=returncode,
                                      stdout=stdout, stderr=stderr)


class _FakeSub:
    """Stands in for the ``subprocess`` module inside ``damage.calc``.

    Script a list of outcomes (one per call); record the call count so we can
    assert retry behaviour. ``TimeoutExpired`` is the REAL class so that
    ``except subprocess.TimeoutExpired`` in calc.py still resolves to it."""

    TimeoutExpired = _real_sub.TimeoutExpired

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def run(self, *args, **kwargs):
        idx = self.calls
        self.calls += 1
        if idx >= len(self.outcomes):
            raise AssertionError(f"subprocess.run called {idx + 1} times (scripted {len(self.outcomes)})")
        outcome = self.outcomes[idx]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# ── single-request fallback (_node_oneshot) ─────────────────────────────────

def test_crash_then_success_retries_once(monkeypatch):
    fake = _FakeSub([_cp(1, stderr="boom"), _cp(0, stdout=GOOD)])
    monkeypatch.setattr(calc, "subprocess", fake)
    d = calc._node_oneshot(REQ)
    assert d == {"min": 100, "max": 120}
    assert fake.calls == 2  # retried exactly once


def test_crash_twice_raises_loudly(monkeypatch):
    fake = _FakeSub([_cp(1, stderr="boom"), _cp(2, stderr="boom again")])
    monkeypatch.setattr(calc, "subprocess", fake)
    with pytest.raises(RuntimeError):
        calc._node_oneshot(REQ)
    assert fake.calls == 2  # retried once, then gave up (never swallowed)


def test_timeout_then_success_retries_once(monkeypatch):
    fake = _FakeSub([_real_sub.TimeoutExpired(cmd="node", timeout=120),
                     _cp(0, stdout=GOOD)])
    monkeypatch.setattr(calc, "subprocess", fake)
    d = calc._node_oneshot(REQ)
    assert d == {"min": 100, "max": 120}
    assert fake.calls == 2


def test_calc_error_is_not_retried(monkeypatch):
    """A real calc ERROR is exit-0 JSON with an ``error`` field — it must raise
    exactly once and NOT be retried (the one-shot contract, unchanged)."""
    fake = _FakeSub([_cp(0, stdout=json.dumps({"error": "illegal species"}))])
    monkeypatch.setattr(calc, "subprocess", fake)
    with pytest.raises(RuntimeError, match="illegal species"):
        calc._node_oneshot(REQ)
    assert fake.calls == 1  # NOT retried


# ── batch fallback (_node_batch_oneshot) ────────────────────────────────────

def test_batch_crash_then_success_retries_once(monkeypatch):
    fake = _FakeSub([_cp(1, stderr="boom"),
                     _cp(0, stdout=json.dumps([{"min": 1, "max": 2}]))])
    monkeypatch.setattr(calc, "subprocess", fake)
    out = calc._node_batch_oneshot([REQ])
    assert out == [{"min": 1, "max": 2}]
    assert fake.calls == 2


def test_batch_crash_twice_raises(monkeypatch):
    fake = _FakeSub([_cp(1, stderr="boom"), _cp(3, stderr="boom again")])
    monkeypatch.setattr(calc, "subprocess", fake)
    with pytest.raises(RuntimeError):
        calc._node_batch_oneshot([REQ])
    assert fake.calls == 2
