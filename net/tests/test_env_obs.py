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
import pytest

from net.env import _zero_obs, build_ctx, build_obs, my_side
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


def test_build_ctx_returns_snapshot_or_degrades_never_raises():
    # build_ctx is the belief half of build_obs (ONE pipeline, ONE source of
    # truth — the ctx-based brain in fitness.strategy_agent reuses it). It
    # returns the get_context() snapshot (top-level keys always present) or
    # None on a broken log — never raises (the caller degrades).
    b = _FakeBattle(active=_FakeMon("garchomp", moves=["outrage"]),
                    avail_moves=["outrage"])
    ctx = build_ctx(b, "gen9ou", "p1")
    assert isinstance(ctx, dict)
    for key in ("turn", "fainted", "active", "field", "own", "opponent"):
        assert key in ctx
    assert build_ctx(_FakeBattle(log=RuntimeError("boom")), "gen9ou", "p1") is None


def test_build_obs_raises_when_vocabs_is_none():
    # A missing vocabs is a WIRING error, not a transient data gap: it must fail
    # LOUD (ValueError), NOT silently degrade to a blind zero-obs every turn (the
    # A3 self-check bug — heuristic vs random ran 3/3 with both sides blind).
    # Contrast: a VALID vocabs + a garbage log still degrades gracefully (above).
    with pytest.raises(ValueError, match="is required"):
        build_obs(_FakeBattle(), "gen9ou", None, "p1")


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


def test_my_side_detects_real_fork_gendered_previews():
    """2026-11-23 draw-bug regression: the fork's ``|poke|`` lines carry a
    display name + gender ("Garchomp, M") while ``battle.team`` carries the
    species id ("garchomp"). The pre-fix raw string compare matched NOTHING
    against real fork logs and silently defaulted every p2 player to p1 —
    its observation was built with the opponent's team as its own, its
    active resolved to 0 HP, and it switched every turn (the stall/draw).
    The pre-fix tests masked this because their fake lines had no gender."""
    lines = ["|poke|p1a|Garchomp, M|", "|poke|p1b|Tornadus, M|",
             "|poke|p2a|Landorus, M|", "|poke|p2b|Talonflame, F|"]
    b1 = _FakeBattle(team={"p1a": _FakeMon("garchomp"),
                           "p1b": _FakeMon("tornadus")})
    b2 = _FakeBattle(team={"p2a": _FakeMon("landorus"),
                           "p2b": _FakeMon("talonflame")})
    assert my_side(b1, lines) == "p1"
    assert my_side(b2, lines) == "p2"          # the pre-fix code said "p1"


def test_my_side_majority_wins_when_teams_share_species():
    """Species Clause is per-team — both teams may field Ceruledge (the
    2026-11-23 stall seeds had exactly this). The majority match must still
    win, not the shared species."""
    lines = ["|poke|p1a|Garchomp, M|", "|poke|p1b|Ceruledge, F|",
             "|poke|p2a|Ceruledge, M|", "|poke|p2b|Tyranitar, F|"]
    b1 = _FakeBattle(team={"p1a": _FakeMon("garchomp"),
                           "p1b": _FakeMon("ceruledge")})
    b2 = _FakeBattle(team={"p2a": _FakeMon("ceruledge"),
                           "p2b": _FakeMon("tyranitar")})
    assert my_side(b1, lines) == "p1"
    assert my_side(b2, lines) == "p2"


# ── _stop_ps_client: clean per-battle websocket teardown ────────────────────
# Each PokeEnv spins up its OWN event-loop thread, and each of its two agents
# owns a PSClient whose listen() coroutine runs on it; poke-env only reaps
# those at atexit. Without a per-battle stop each battle leaks a socket +
# coroutine (and, when a server is killed first, the ConnectionResetError
# teardown noise). These pin the best-effort helper ShowdownEnv.close() calls.

class _FakeWS:
    def __init__(self, raise_on_close: bool = False):
        self.closed = False
        self._raise = raise_on_close

    async def close(self):
        if self._raise:
            raise RuntimeError("boom")
        self.closed = True


class _FakeClient:
    """Mimic the poke-env PSClient surface _stop_ps_client uses:
    websocket, loop, and the async stop_listening() (which closes the
    websocket)."""
    def __init__(self, ws, loop):
        self.websocket = ws
        self.loop = loop

    async def stop_listening(self):
        # mirrors PSClient.stop_listening -> _stop_listening -> websocket.close()
        await self.websocket.close()


def _running_loop():
    import asyncio
    import threading
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop, t


def test_stop_ps_client_closes_the_websocket():
    from net.env import _stop_ps_client
    loop, t = _running_loop()
    try:
        ws = _FakeWS()
        assert _stop_ps_client(_FakeClient(ws, loop)) is True
        assert ws.closed is True
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)


def test_stop_ps_client_none_is_a_noop():
    from net.env import _stop_ps_client
    assert _stop_ps_client(None) is False


def test_stop_ps_client_missing_websocket_is_a_noop():
    from net.env import _stop_ps_client

    class _NoWS:
        websocket = None
        loop = None

    assert _stop_ps_client(_NoWS()) is False


def test_stop_ps_client_missing_loop_is_a_noop():
    from net.env import _stop_ps_client
    assert _stop_ps_client(_FakeClient(_FakeWS(), None)) is False


def test_stop_ps_client_swallows_a_close_error():
    """A failing close must NOT raise into the caller (teardown must never fail
    the battle it is tearing down)."""
    from net.env import _stop_ps_client
    loop, t = _running_loop()
    try:
        assert _stop_ps_client(_FakeClient(_FakeWS(raise_on_close=True), loop)) is False
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join(timeout=5)
