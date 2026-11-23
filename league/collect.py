"""D103 Step 2 — per-battle trajectory collection into a bot's buffer.

ONE collector per (battle, side). The driver calls :meth:`record` at the
decision seam; the engine calls :meth:`finalize` with the decided outcome
when the battle ends, and extends the bot's ``RolloutBuffer`` with the steps.

The step shape is the SAME as ``net.trainer.play_and_collect`` (no parallel
semantics — the league feeds the same ``net.ppo`` machinery):

* only REAL actions are stored — a pass/wait (``d.action is None``) is not a
  PPO transition and is skipped (defensively, so a sloppy caller can't
  inject a phantom transition);
* ``old_logprob`` / ``old_value`` are the decision-time numbers; ``old_value``
  stays RAW (0,1) — the ``2V-1`` reward-space mapping belongs to the GAE
  step (``SelfPlayTrainer.apply_gae``), exactly as in the solo trainer;
* the terminal reward (winner +1 / loser −1 / draw −1, 2026-11-23 ruling)
  lands on the LAST step
  only — ``gae()``'s sparse-terminal model; all earlier steps stay 0.0;
* a battle that ERRORED (infra) is DROPPED, never finalized: a partial
  trajectory with no decided outcome is poison for PPO (the engine discards
  the whole collector — see ``league.engine.LeagueRun._on_battle``).

Threading: one collector is owned by ONE battle (one worker thread); no
locking needed (the same discipline as ``DecisionGuard``).
"""
from __future__ import annotations

from typing import Any, List, Optional

from net.ppo import RolloutStep

__all__ = ["TrajectoryCollector"]


class TrajectoryCollector:
    """One battle, one side: the bot's decisions in play order, GAE-ready."""

    def __init__(self) -> None:
        self._steps: List[RolloutStep] = []
        self._finalized: Optional[float] = None

    # ── collection (the driver's seam) ──────────────────────────────────
    def record(self, obs: Any, d: Any) -> None:
        """Store ONE decision (``d``: a ``net.agent.Decision``-shaped object).

        A pass/wait (``d.action is None``) is skipped — nothing happened, so
        there is no transition to learn from."""
        if self._finalized is not None:
            raise RuntimeError(
                "TrajectoryCollector.record after finalize — a battle's "
                "trajectory is closed once its outcome is attached")
        if d is None or d.action is None:
            return
        self._steps.append(RolloutStep(
            obs=obs,
            action_index=int(d.action_index),
            old_logprob=float(d.logprob),
            old_value=float(d.value),          # RAW (0,1) — GAE maps it (step 3)
            reward=0.0,
            done=False,
        ))

    # ── finalization (the engine's seam, at battle end) ─────────────────
    def finalize(self, outcome: float, gamma: float = 0.99,
                 lam: float = 0.95) -> "TrajectoryCollector":
        """Attach the decided outcome to the trajectory AND GAE it.

        ``outcome``: +1 (win) / −1 (loss) / −1 (draw, 2026-11-23 ruling)
        from the bot's own POV.
        Lands on the LAST step (sparse-terminal, what ``gae()`` expects) with
        ``done=True``; earlier steps keep reward 0.0. Empty trajectories
        (a side never acted) finalize to nothing — the buffer is untouched.

        GAE-at-finalize (D103 step 3): this is the ONE place a battle's
        trajectory is known to be COMPLETE, so each step's ``advantage``/
        ``ret`` are filled here via the shared ``net.ppo.fill_gae`` (the same
        2V-1-mapped sparse-terminal GAE the solo trainer uses). The buffer is
        then extended with already-GAE'd steps, and the round-end update never
        re-GAEs across the whole buffer (which would be wrong — the
        intermediate battle ends are not terminals). ``gamma``/``lam`` come
        from the league config (defaults mirror ``TrainConfig``)."""
        if self._finalized is not None:
            raise RuntimeError(
                "TrajectoryCollector.finalize called twice — the outcome was "
                f"already attached ({self._finalized:+.0f})")
        self._finalized = float(outcome)
        if self._steps:
            self._steps[-1].reward = float(outcome)
            self._steps[-1].done = True
            from net.ppo import fill_gae
            fill_gae(self._steps, gamma=gamma, lam=lam)
        return self

    # ── introspection ───────────────────────────────────────────────────
    @property
    def steps(self) -> List[RolloutStep]:
        """The collected steps (read-only view; the engine extends a buffer
        with them AFTER :meth:`finalize`)."""
        return self._steps

    @property
    def finalized(self) -> bool:
        return self._finalized is not None

    def __len__(self) -> int:
        return len(self._steps)
