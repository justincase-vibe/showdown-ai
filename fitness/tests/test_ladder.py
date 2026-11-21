"""Ladder + reference tests (A1). Hermetic: no server, CPU-only nets/agents."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fitness.ladder import Reference, Ladder, random_reference


@pytest.fixture
def vocabs():
    from net.vocab_db import build_vocabs

    return build_vocabs()


def test_ladder_container():
    ladder = Ladder()
    assert len(ladder) == 0
    assert ladder.names() == []

    ref = random_reference(None, device="cpu")
    ladder.add(ref)
    assert len(ladder) == 1
    assert "random" in ladder
    assert "nope" not in ladder
    assert ladder.names() == ["random"]
    assert ladder.get("random") is ref
    assert [r.name for r in ladder] == ["random"]

    with pytest.raises(KeyError, match="unknown reference"):
        ladder.get("ghost")


def test_ladder_replaces_by_name():
    ladder = Ladder()
    ladder.add(Reference(name="x", make_agent=lambda seed: None, weight=1.0))
    ladder.add(Reference(name="x", make_agent=lambda seed: None, weight=2.0))
    assert len(ladder) == 1
    assert ladder.get("x").weight == 2.0


def test_random_reference_makes_fresh_agent(vocabs):
    from net.agent import PokeAgent

    ref = random_reference(vocabs, device="cpu")
    assert ref.name == "random"
    agent = ref.agent(seed=123)
    assert isinstance(agent, PokeAgent)
    # a fresh agent per battle (RNG isolation — same object is never reused)
    agent2 = ref.agent(seed=123)
    assert agent is not agent2
    # the agent exposes the decision seam the harness/driver rely on
    assert hasattr(agent, "act") and hasattr(agent, "act_greedy")


def test_reference_agent_seed_passthrough(vocabs):
    """A seeded reference gives the SAME net weights for the same seed
    (reproducible), and DIFFERENT nets for different seeds (diversity)."""
    import torch

    ref = random_reference(vocabs, device="cpu")
    a = ref.agent(seed=7)
    b = ref.agent(seed=7)
    c = ref.agent(seed=8)
    same = all(torch.equal(p1, p2) for p1, p2 in zip(a.net.parameters(), b.net.parameters()))
    diff = any(not torch.equal(p1, p2) for p1, p2 in zip(a.net.parameters(), c.net.parameters()))
    assert same, "same seed should give identical weights"
    assert diff, "different seeds should give different weights"
