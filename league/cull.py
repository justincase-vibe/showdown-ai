"""Cull selection (D103 step 4) — the PURE decision of WHICH bots to retire.

The league's population loop: every K rounds the bottom X% by ELO is retired
and replaced by FRESH random bots (diversity / local-minima escape — the
user's "constantly cycle new agents"). This module owns ONLY the selection
(torch-free, roster-records-only, trivially testable); the engine performs
the roster surgery + bot minting (``engine.LeagueRun._cull_and_refresh``)
because that needs the live ``LearningBot``s (torch).

Policy (pinned by ``league/tests/test_cull.py``):
  * the bottom ``frac`` of the roster by ELO, ascending (ties broken by
    NAME — deterministic regardless of roster order);
  * at least 1 bot is retired when the roster is above the floor (a cull
    that would retire 0 bots is a no-op cull — pointless churn);
  * the roster never drops below ``min_roster`` (and never below 2 — the
    league's hard minimum, regardless of the knob);
  * ``frac <= 0`` (disabled) or an empty roster -> ``[]`` (no cull).
"""
from __future__ import annotations

import math
from typing import Any, List, Sequence

__all__ = ["select_cull"]


def select_cull(bots: Sequence[Any], frac: float, min_roster: int) -> List[str]:
    """The names of the bots to retire: the bottom ``frac`` of ``bots``
    by ELO.

    ``bots`` — any objects with ``.name`` (str) and ``.elo`` (int/float)
    (``registry.Bot`` records or a test stand-in). Order-independent.

    Returns ``[]`` when nothing may be culled: ``frac <= 0``, an empty
    roster, or culling would drop the roster below the floor
    (``max(2, min_roster)``)."""
    if frac is None or len(bots) == 0:
        return []
    try:
        x = float(frac)
    except (TypeError, ValueError):
        return []
    if x <= 0.0 or not math.isfinite(x):
        return []
    n = len(bots)
    floor = max(2, int(min_roster))
    k = max(1, math.floor(n * x))          # at least 1 when culling at all
    k = min(k, n - floor)                  # never below the floor
    if k <= 0:
        return []
    ordered = sorted(bots, key=lambda b: (int(b.elo), str(b.name)))
    return [b.name for b in ordered[:k]]
