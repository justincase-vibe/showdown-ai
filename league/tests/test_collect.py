"""D103 Step 2 — ``league.collect.TrajectoryCollector`` (FAST + offline).

Pins the step shape the PPO update (step 3) will consume — the SAME shape
``net.trainer.play_and_collect`` produces, so the league feeds the identical
``net.ppo`` machinery:

  * only real actions are stored (a pass/wait is NOT a transition);
  * ``old_logprob``/``old_value`` are the decision-time numbers, ``old_value``
    RAW (0,1) — the 2V-1 reward-space map stays in the GAE step;
  * the terminal reward (+1/−1/0) lands on the LAST step only (``gae()``'s
    sparse-terminal model);
  * finalize is one-shot; recording after finalize is a loud error.

Run:  python -m pytest league/tests/test_collect.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.collect import TrajectoryCollector
from net.agent import Decision


_UNSET = object()   # distinguishes "action not given" from an explicit pass/wait


def _decision(action_index: int = 0, logprob: float = -0.5,
              value: float = 0.6, action=_UNSET) -> Decision:
    if action is _UNSET:
        action = {"kind": "move", "move": "tackle"}
    return Decision(action=action, action_index=action_index,
                    logprob=logprob, value=value, entropy=0.3,
                    prob=np.zeros(2))


def _obs():
    return {"f": np.zeros((2, 2), dtype=np.float32)}


# ── the step shape ─────────────────────────────────────────────────────────

def test_record_stores_the_decision_numbers_raw():
    c = TrajectoryCollector()
    obs = _obs()
    c.record(obs, _decision(action_index=2, logprob=-1.25, value=0.42))
    s = c.steps[0]
    assert s.obs is obs                    # the EXACT obs the agent saw (no copy)
    assert s.action_index == 2
    assert s.old_logprob == pytest.approx(-1.25)
    assert s.old_value == pytest.approx(0.42)   # RAW (0,1) — GAE maps it (step 3)
    assert s.reward == 0.0 and s.done is False


def test_record_skips_pass_wait_turns():
    """A pass/wait (``d.action is None``) is not a PPO transition — even if a
    sloppy caller hands one in, it must not inject a phantom step."""
    c = TrajectoryCollector()
    c.record(_obs(), _decision(action=None, action_index=-1, logprob=0.0,
                               value=0.5))
    assert len(c) == 0


# ── finalization: the sparse-terminal reward model ─────────────────────────

def test_finalize_win_puts_terminal_reward_on_last_step_only():
    c = TrajectoryCollector()
    c.record(_obs(), _decision())
    c.record(_obs(), _decision(action_index=1))
    c.record(_obs(), _decision(action_index=2))
    c.finalize(+1.0)
    assert len(c) == 3
    assert c.steps[0].reward == 0.0 and c.steps[0].done is False
    assert c.steps[1].reward == 0.0 and c.steps[1].done is False
    assert c.steps[2].reward == pytest.approx(1.0) and c.steps[2].done is True


def test_finalize_loss_and_draw():
    for outcome, want in ((-1.0, -1.0), (0.0, 0.0)):
        c = TrajectoryCollector()
        c.record(_obs(), _decision())
        c.finalize(outcome)
        assert c.steps[0].reward == pytest.approx(want)
        assert c.steps[0].done is True


def test_empty_trajectory_finalizes_to_nothing():
    """A side that never acted (e.g. fainted before its first turn)
    contributes zero steps — the engine's buffer.extend([]) is a no-op."""
    c = TrajectoryCollector()
    c.finalize(+1.0)
    assert len(c) == 0 and c.steps == [] and c.finalized


# ── one-shot finalization (no double-booking of an outcome) ────────────────

def test_double_finalize_raises_loudly():
    c = TrajectoryCollector()
    c.record(_obs(), _decision())
    c.finalize(+1.0)
    with pytest.raises(RuntimeError, match="twice"):
        c.finalize(-1.0)


def test_record_after_finalize_raises_loudly():
    c = TrajectoryCollector()
    c.finalize(+1.0)
    with pytest.raises(RuntimeError, match="after finalize"):
        c.record(_obs(), _decision())
