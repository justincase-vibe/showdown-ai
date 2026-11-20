"""D92 Step 2 — the human-vs-bot player (``net.human_play``).

Fast + offline (no fork, no GPU): they pin the **net -> legal-order** decision
path (``decide_order``) and the **checkpoint load** the user will actually fight
with (their real 50-battle BC net). The live "play a battle in the browser" path
needs the fork server and is exercised by the manual GUI flow (D92 step 3) —
instantiating a real ``Player`` would open a websocket, so it is intentionally
NOT unit-tested here.
"""
from __future__ import annotations

import asyncio
import threading
import types
from pathlib import Path

import pytest

import net.human_play as hp
from net.agent import PokeAgent, make_random_net
from net.env import build_action_list, build_obs
from net.tests.test_env_obs import _FakeBattle, _FakeMon
from net.tests.test_s41 import make_ctx, make_vocabs


def _vocabs():
    return make_vocabs(make_ctx())


def _agent():
    # seed=1234 pins the WEIGHTS: an unseeded random net depends on the global
    # torch RNG state left by earlier code, so unseeded assertions on concrete
    # decisions are order-dependent flakes.
    return PokeAgent(make_random_net(_vocabs(), seed=1234), seed=1234)


def test_decide_order_returns_legal_move_pe():
    v = _vocabs()
    b = _FakeBattle(
        team={"p1a": _FakeMon("garchomp", moves=["outrage", "earthquake"]),
              "p1b": _FakeMon("tornadustherian")},
        active=_FakeMon("garchomp", moves=["outrage", "earthquake"]),
        avail_moves=["outrage", "earthquake"],
        switches=["tornadustherian"],
        tera=True,
    )
    pe = hp.decide_order(_agent(), b, "gen9ou", v, greedy=True)
    # 1 = the one bench switch (p1b — poke-env indexes switches into the WHOLE
    # team array, where the active p1a is 0); 6/7 = the two moves; 22/23 = tera
    # on those moves. (The old expectation of 0 was stale — a switch to p1b is
    # team index 1, not 0.)
    assert pe in (1, 6, 7, 22, 23), pe


def test_decide_order_is_deterministic_when_greedy():
    v = _vocabs()
    b = _FakeBattle(active=_FakeMon("garchomp", moves=["outrage", "earthquake"]),
                    avail_moves=["outrage", "earthquake"], tera=True)
    a = _agent()  # one fixed net -> argmax must be stable across calls
    assert hp.decide_order(a, b, "gen9ou", v, greedy=True) == \
        hp.decide_order(a, b, "gen9ou", v, greedy=True)


def test_decide_order_pass_when_no_legal_action():
    v = _vocabs()
    b = _FakeBattle(active=None, avail_moves=[], switches=[])   # a wait turn
    assert hp.decide_order(_agent(), b, "gen9ou", v, greedy=True) == hp.PASS


def test_load_real_bc_checkpoint_via_play_path():
    """The strongest end-to-end proof: the user's ACTUAL 50-battle BC net loads
    through the same path the browser bot uses, and produces a usable agent."""
    ckpt = Path(__file__).resolve().parents[2] / "scratch" / "2026-11-19_bc_50b.pt"
    if not ckpt.exists():
        pytest.skip("the 50-battle BC checkpoint is not present (scratch/)")
    try:
        from net.vocab_db import build_vocabs
        vocabs = build_vocabs()
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"legality DB / vocabs not available: {exc}")
    net, agent = hp._load_checkpoint(str(ckpt), vocabs, "cpu")
    assert sum(p.numel() for p in net.parameters()) > 1_000_000   # the ~7.2M net
    b = _FakeBattle(active=_FakeMon("garchomp", moves=["outrage"]),
                    avail_moves=["outrage"], tera=True)
    d = agent.act(build_obs(b, "gen9ou", vocabs, "p1"),
                  build_action_list(b), deterministic=True)
    assert d is not None


def test_fork_reachable_returns_bool():
    # a closed port reports False (does not raise); 8000 reflects the fork's state.
    assert hp._fork_reachable(port=1) is False
    assert isinstance(hp._fork_reachable(port=8000), bool)


# ── convenience flow (2026-11-20): open the browser + auto-challenge YOU ────

def test_open_in_browser_returns_bool(monkeypatch):
    import webbrowser
    monkeypatch.setattr(webbrowser, "open", lambda url: True)
    assert hp._open_in_browser() is True
    monkeypatch.setattr(webbrowser, "open", lambda url: False)
    assert hp._open_in_browser() is False
    # a crashing browser (headless edge) must not raise — best-effort contract
    def _boom(url):
        raise RuntimeError("no display")
    monkeypatch.setattr(webbrowser, "open", _boom)
    assert hp._open_in_browser() is False


class _FakeClient:
    """Just enough of poke_env's PSClient: a real event loop (run in its own
    thread) + the logged_in Event + challenge(). Mirrors the contract
    ``send_challenge`` uses — nothing more."""

    def __init__(self, logged_in: bool = True):
        self.loop = asyncio.new_event_loop()
        self.logged_in = asyncio.Event()
        if logged_in:
            self.logged_in.set()
        self.challenges = []

    async def challenge(self, username, format_, packed_team):
        self.challenges.append((username, format_, packed_team))


def _client_with_running_loop(logged_in: bool = True):
    client = _FakeClient(logged_in=logged_in)

    def _run():
        asyncio.set_event_loop(client.loop)
        client.loop.run_forever()

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return client, t


def test_send_challenge_sends_exactly_one_challenge():
    client, _t = _client_with_running_loop()
    player = types.SimpleNamespace(ps_client=client, next_team="PACKED_TEAM")
    hp.send_challenge(player, "ttvJustinCase", "gen9ou")
    # the exact contract: ONE challenge, to you, in the configured format, with
    # the bot's next team
    assert client.challenges == [("ttvJustinCase", "gen9ou", "PACKED_TEAM")]
    client.loop.call_soon_threadsafe(client.loop.stop)


def test_send_challenge_raises_when_bot_not_logged_in():
    # no hang, no silent drop — the caller falls back to the manual /challenge
    client, _t = _client_with_running_loop(logged_in=False)
    player = types.SimpleNamespace(ps_client=client, next_team="PACKED_TEAM")
    with pytest.raises(Exception):
        hp.send_challenge(player, "ttvJustinCase", "gen9ou", timeout=0.3)
    assert client.challenges == []
    client.loop.call_soon_threadsafe(client.loop.stop)
