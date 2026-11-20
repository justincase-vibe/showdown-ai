"""D103 Step 4 (4.1) — ``league.cull.select_cull``: the PURE cull selection.

FAST + offline + torch-free: pins the population loop's selection policy —
bottom X% by ELO, deterministic (order-independent, ties by name), the
roster floor (never below ``min_roster``, never below 2), the disable
switch (``frac <= 0``), and the new ``LeagueConfig`` knobs.

Run:  python -m pytest league/tests/test_cull.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.cull import select_cull


class _Bot:
    """A roster-record stand-in (just ``.name`` + ``.elo``)."""

    def __init__(self, name: str, elo: int):
        self.name = name
        self.elo = elo


def _bots(*elos: int) -> list:
    return [_Bot(f"bot{i:02d}", elo) for i, elo in enumerate(elos)]


# ── the selection policy ────────────────────────────────────────────────────

def test_selects_bottom_frac_by_elo():
    # 8 bots, 25% -> the 2 weakest (1300, 1400), roster order shuffled
    bots = _bots(1400, 1500, 1600, 1700, 1300, 1450, 1550, 1650)
    assert select_cull(bots, 0.25, min_roster=4) == ["bot04", "bot00"]


def test_at_least_one_retired_when_above_floor():
    # 4 bots, 25% -> floor(1.0)=1 (a 0-bot cull would be pointless churn)
    bots = _bots(1500, 1500, 1500, 1600)
    assert len(select_cull(bots, 0.25, min_roster=2)) == 1


def test_never_drops_below_min_roster():
    # 4 bots at the floor -> nothing may be retired
    bots = _bots(1400, 1500, 1600, 1700)
    assert select_cull(bots, 0.25, min_roster=4) == []


def test_floor_is_at_least_two_regardless_of_knob():
    # 3 bots, min_roster=1 -> the league's hard minimum (2) still holds
    bots = _bots(1400, 1500, 1600)
    culled = select_cull(bots, 0.5, min_roster=1)
    assert len(culled) == 1            # 3 - 1 = 2 remain


def test_frac_zero_or_negative_disables():
    bots = _bots(1400, 1500, 1600, 1700)
    assert select_cull(bots, 0, min_roster=2) == []
    assert select_cull(bots, -0.1, min_roster=2) == []
    assert select_cull(bots, float("nan"), min_roster=2) == []
    assert select_cull(bots, None, min_roster=2) == []


def test_empty_roster():
    assert select_cull([], 0.25, min_roster=4) == []


def test_order_independent():
    a = _bots(1500, 1400, 1600, 1550, 1450, 1700)
    b = list(reversed(a))
    assert select_cull(a, 0.33, min_roster=2) == select_cull(b, 0.33, min_roster=2)


def test_ties_broken_by_name_deterministically():
    # two 1400s (beta/omega) then two 1500s (alpha/zeta): 4 bots, 50% -> 2
    tie = [_Bot("zeta", 1500), _Bot("alpha", 1500),
           _Bot("omega", 1400), _Bot("beta", 1400)]
    assert select_cull(tie, 0.5, min_roster=2) == ["beta", "omega"]


# ── the LeagueConfig knobs (defaults + from_dict round-trip) ───────────────

def test_league_config_cull_knobs_default_and_roundtrip():
    from league.engine import LeagueConfig
    cfg = LeagueConfig()
    assert (cfg.cull_every, cfg.cull_frac, cfg.min_roster) == (4, 0.25, 4)
    assert cfg.refresh_dir is None
    d = LeagueConfig.from_dict(
        {"cull_every": 0, "cull_frac": 0.5, "min_roster": 2,
         "refresh_dir": "some/dir"})
    assert (d.cull_every, d.cull_frac, d.min_roster) == (0, 0.5, 2)
    assert d.refresh_dir == "some/dir"
