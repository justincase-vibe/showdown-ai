"""D92 Step 1 — the standalone observation builder (``net.env.build_obs``).

These pin the DRY refactor of ``ShowdownEnv.embed_battle`` into a free function
so a poke-env ``Player`` (human-vs-bot play) can build the net's observation
directly from a live battle, and the new ``my_side`` POV auto-detector.

All fast + offline: they use a lightweight fake battle (no fork, no connection,
no GPU). The rich encoder path (real species/moves) is already covered by the
S4.1/S4.2 gates — this only proves the refactor preserved the pipeline and that
``my_side`` detects p1 vs p2 correctly.
"""
from __future__ import annotations

import numpy as np

from net.env import _zero_obs, build_obs, my_side
from net.tests.test_s41 import make_ctx, make_vocabs


# ── a minimal fake poke-env battle (just enough for build_obs/build_action_list) ─
class _FakeMove:
    def __init__(self, mid: str):
        self.id = mid


class _FakeMon:
    def __init__(self, species: str, moves=()):
        self.species = species
        self.base_species = species
        self.moves = {i: _FakeMove(m) for i, m in enumerate(moves)}


class _FakeBattle:
    def __init__(self, log="" , team=None, active=None, avail_moves=(),
                 switches=(), fs=False, trapped=False, tera=False, wait=False):
        self._log = log
        self.team = team or {}
        self.active_pokemon = active
        self.available_moves = [_FakeMove(m) for m in avail_moves]
        self.available_switches = [_FakeMon(s) for s in switches]
        self.force_switch = fs
        self.trapped = trapped
        self.can_tera = tera
        self._wait = wait

    def _build_replay_log(self):
        if isinstance(self._log, Exception):
            raise self._log
        return self._log


def _vocabs():
    return make_vocabs(make_ctx())


# ── build_obs: shape + finiteness (the refactor preserved the pipeline) ────
def test_build_obs_shape_matches_zero_obs():
    obs = build_obs(_FakeBattle(), "gen9ou", _vocabs(), "p1")
    z = _zero_obs()
    assert set(obs.keys()) == set(z.keys())
    for k, arr in obs.items():
        assert arr.shape == z[k].shape, f"shape drift on {k}"
        assert np.isfinite(arr.astype(float)).all(), f"non-finite on {k}"


def test_build_obs_with_active_moves_still_valid_shape():
    # a real-ish decision moment: an active mon with moves + a bench switch.
    b = _FakeBattle(
        active=_FakeMon("garchomp", moves=["outrage", "earthquake"]),
        avail_moves=["outrage", "earthquake"],
        switches=["tornadustherian"],
        tera=True,
    )
    obs = build_obs(b, "gen9ou", _vocabs(), "p1")
    z = _zero_obs()
    assert set(obs.keys()) == set(z.keys())
    for arr in obs.values():
        assert np.isfinite(arr.astype(float)).all()


def test_build_obs_defensive_on_garbage_log():
    # a belief/encode failure must degrade to a valid zero obs, never raise.
    obs = build_obs(_FakeBattle(log=RuntimeError("boom")), "gen9ou", _vocabs(), "p1")
    assert set(obs.keys()) == set(_zero_obs().keys())
    for arr in obs.values():
        assert np.isfinite(arr.astype(float)).all()


# ── my_side: the POV auto-detector (a p2 bot must swap) ────────────────────
_P1 = ["|poke|p1a|Garchomp|M|", "|poke|p1b|Tornadus|M|"]
_P2 = ["|poke|p2a|Landorus|M|", "|poke|p2b|Talonflame|M|"]
_ALL = _P1 + _P2


def test_my_side_detects_p1():
    b = _FakeBattle(team={"p1a": _FakeMon("garchomp"), "p1b": _FakeMon("tornadus")})
    assert my_side(b, _ALL) == "p1"


def test_my_side_detects_p2():
    b = _FakeBattle(team={"p2a": _FakeMon("landorus"), "p2b": _FakeMon("talonflame")})
    assert my_side(b, _ALL) == "p2"


def test_my_side_defaults_to_p1_when_team_unknown():
    assert my_side(_FakeBattle(), _ALL) == "p1"
    assert my_side(_FakeBattle(), []) == "p1"


def test_my_side_never_raises_on_malformed_lines():
    assert my_side(_FakeBattle(team={"p1a": _FakeMon("garchomp")}),
                   ["|poke|malformed", "|nonsense|line|"]) == "p1"
