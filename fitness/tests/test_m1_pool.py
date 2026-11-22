"""M1 gate (C1) — the opponent pool, hermetic.

Pins: the pool resolves the 4 standing references + self-snapshots in order;
each reference's ``make_agent`` returns the right KIND of agent (net ones are
PokeAgents; the two ctx brains carry ``decides_from_battle``; the posterior
variant is the ONLY one with ``posterior=True``); the snapshot net loads
LAZILY and is CACHED (one load, one frozen net shared across battles).

Run:  python -m pytest fitness/tests/test_m1_pool.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402  (only the TEST builds nets; the module stays torch-free)

from fitness.ladder import Ladder  # noqa: E402
from fitness.m1_pool import (  # noqa: E402
    M1_CORE_NAMES, build_m1_pool, snapshot_reference,
)


def _vocabs():
    from net.vocab_db import build_vocabs
    return build_vocabs()


def test_pool_is_a_ladder_with_the_4_core_names():
    ladder = build_m1_pool(fmt="gen9ou", vocabs=_vocabs())
    assert isinstance(ladder, Ladder)
    names = ladder.names()
    # order preserved: the 4 standing refs first, in canonical order
    assert names[:4] == list(M1_CORE_NAMES), names
    assert set(names) == set(M1_CORE_NAMES)


def test_core_reference_weights():
    ladder = build_m1_pool(fmt="gen9ou", vocabs=_vocabs())
    assert ladder.get("random").weight == 1.0          # the floor
    assert ladder.get("heuristic").weight == 2.0
    assert ladder.get("strategy").weight == 2.0
    assert ladder.get("strategy_posterior").weight == 2.0


def test_agent_kinds_match_the_interface():
    ladder = build_m1_pool(fmt="gen9ou", vocabs=_vocabs())
    from net.agent import PokeAgent
    # net opponents (PokeAgent)
    assert isinstance(ladder.get("random").agent(seed=0), PokeAgent)
    # ctx-brain opponents: decide from the BATTLE, not the encoded obs
    strat = ladder.get("strategy").agent(seed=0)
    post = ladder.get("strategy_posterior").agent(seed=0)
    assert getattr(strat, "decides_from_battle", False) is True
    assert getattr(post, "decides_from_battle", False) is True
    # the posterior variant is the ONLY informed-calc one (the plain yardstick
    # must stay known-info-only — 6.6a)
    assert getattr(post, "posterior", None) is True
    assert getattr(strat, "posterior", None) is False
    # the OLD heuristic is PokeAgent-compatible (has .act)
    assert hasattr(ladder.get("heuristic").agent(seed=0), "act")


def test_snapshot_reference_lazy_and_cached(tmp_path):
    """A self-snapshot: the net loads ONCE and is shared across battles."""
    from net.model import PokeNet
    from net.agent import PokeAgent
    v = _vocabs()
    ckpt = tmp_path / "self_round1.pt"
    # BC-format checkpoint (state_for_save payload) — load_file reads it directly
    torch.save(PokeNet(v).state_for_save(), ckpt)

    ref = snapshot_reference(ckpt, v, device="cpu", name="self_1")
    a1 = ref.agent(seed=1)
    a2 = ref.agent(seed=2)
    assert isinstance(a1, PokeAgent) and isinstance(a2, PokeAgent)
    # the SAME frozen net object backs both battles (cached -> one disk read)
    assert a1.net is a2.net
    assert ref.name == "self_1"


def test_pool_includes_snapshots_in_order(tmp_path):
    from net.model import PokeNet
    from net.agent import PokeAgent
    v = _vocabs()
    c1, c2 = tmp_path / "s1.pt", tmp_path / "s2.pt"
    torch.save(PokeNet(v).state_for_save(), c1)
    torch.save(PokeNet(v).state_for_save(), c2)
    ladder = build_m1_pool(fmt="gen9ou", vocabs=v, snapshot_paths=[str(c1), str(c2)])
    names = ladder.names()
    assert names[:4] == list(M1_CORE_NAMES)
    assert names[4:6] == ["self_1", "self_2"], names
    # a snapshot agent is a PokeAgent (the learner's own frozen copy)
    assert isinstance(ladder.get("self_1").agent(seed=0), PokeAgent)


def test_pool_ctx_brains_are_decide_drivable():
    """C1b: the trainer drives ``decides_from_battle`` opponents via
    ``.decide(battle, fmt, vocabs)`` (the NEW branch in ``play_vs_checkpoint``).
    Confirm the pool's strategy / strategy_posterior agents return a LEGAL action
    (or a pass) from a real fake battle — the EXACT call the C3 loop will make, so
    the full pool is usable WITHOUT a real battle. (The net / old-heuristic ``.act``
    path is already covered by ``league.tests.test_driver`` + the ``heuristic``
    reference tests.)"""
    from net.tests.test_env_obs import _FakeBattle, _FakeMon
    from net.env import build_action_list
    v = _vocabs()
    ladder = build_m1_pool(fmt="gen9ou", vocabs=v)
    b = _FakeBattle(active=_FakeMon("garchomp", moves=["outrage", "earthquake"]),
                    avail_moves=["outrage", "earthquake"], tera=True)
    acts = build_action_list(b)
    assert acts, "the fake battle must offer legal actions"
    for name in ("strategy", "strategy_posterior"):
        agent = ladder.get(name).agent(seed=0)
        assert getattr(agent, "decides_from_battle", False) is True
        got = agent.decide(battle=b, fmt="gen9ou", vocabs=v)
        assert got is None or got in acts, (name, got, acts)
