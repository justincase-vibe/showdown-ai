"""D92 Step 2 — the human-vs-bot player (``net.human_play``).

Fast + offline (no fork, no GPU): they pin the **net -> legal-order** decision
path (``decide_order``) and the **checkpoint load** the user will actually fight
with (their real 50-battle BC net). The live "play a battle in the browser" path
needs the fork server and is exercised by the manual GUI flow (D92 step 3) —
instantiating a real ``Player`` would open a websocket, so it is intentionally
NOT unit-tested here.
"""
from __future__ import annotations

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
    return PokeAgent(make_random_net(_vocabs()), seed=1234)


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
    # 0 = the one bench switch; 6/7 = the two moves; 22/23 = tera on those moves.
    assert pe in (0, 6, 7, 22, 23), pe


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
