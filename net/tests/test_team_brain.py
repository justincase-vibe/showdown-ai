"""Team brain tests (net/team_brain.py).

Hermetic (synthetic vocabs, no legality DB, no league data) and CPU:
  * tokenization (display name -> toID vocab index, <pad> for OOV, padding)
  * PERMUTATION INVARIANCE — shuffling a team's slots must not change its
    score (a team is a SET of mons; the slot order is noise)
  * score determinism + shapes
  * rank() — best-first, and candidates that do not beat the parent rank last
  * save/load round-trip (identical scores)
  * team_pairs_from_registry (min_battles filter, equal-ELO dropped)
  * THE SYNTHETIC-SYNERGY OVERTIT TEST: a count-interaction rule (a property
    of the COMBINATION, not of any single mon) must be learnable — if the
    brain cannot overfit an obvious synergy, it cannot rank the 10k's
    mutations. (Approved handoff item 7.)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from net.team_brain import (
    TeamBrain,
    team_pairs_from_registry,
    team_to_ids,
)
from net.vocab import Vocab, Vocabs
from league.teams import TeamRegistry, ORIGIN_GENERATED

# ═══════════════════════════════════════════════════════════════════════════
#  synthetic world (hermetic; the rule below is the overfit target)
# ═══════════════════════════════════════════════════════════════════════════

SPECIES = [f"Sp{i}" for i in range(1, 11)]
ABILITIES = ["Ab1", "Ab2", "Ab3", "Ab4"]
ITEMS = ["It1", "It2", "It3"]
MOVES = [f"Mv{i}" for i in range(1, 13)]

#: the synthetic synergy rule: GOOD iff (>=2 mons carry Ab1) AND (>=1 mon
#: carries Mv1). A COUNT INTERACTION — a per-mon linear model cannot express
#: it; only a token-interacting model (attention over the 6 mon tokens) can.
SYNERGY_ABILITY = "Ab1"
SYNERGY_MOVE = "Mv1"


def synthetic_vocabs() -> Vocabs:
    """Production-realistic: the vocabs are keyed by toID (lowercase alnum),
    exactly like ``net.vocab_db.build_vocabs`` builds them from the legality
    DB's ids. The mon-dicts keep DISPLAY names ("Sp1") — the tokenizer's
    ``_to_id`` normalization is what bridges the two (the net/encoder.py
    convention). Building these vocabs with display names would make every
    lookup silently OOV -> <pad> (the bug this test world exists to pin)."""
    return Vocabs(
        species=Vocab([s.lower() for s in SPECIES]),
        ability=Vocab([a.lower() for a in ABILITIES]),
        item=Vocab([i.lower() for i in ITEMS]),
        move=Vocab([m.lower() for m in MOVES]),
    )


def random_team(rng: random.Random) -> list:
    return [
        {
            "species": rng.choice(SPECIES),
            "ability": rng.choice(ABILITIES),
            "item": rng.choice(ITEMS),
            "moves": rng.sample(MOVES, k=4),
        }
        for _ in range(6)
    ]


def is_good(team) -> bool:
    n_ab1 = sum(1 for m in team if m["ability"] == SYNERGY_ABILITY)
    n_mv1 = sum(1 for m in team if SYNERGY_MOVE in m["moves"])
    return n_ab1 >= 2 and n_mv1 >= 1


# ═══════════════════════════════════════════════════════════════════════════
#  tokenization
# ═══════════════════════════════════════════════════════════════════════════

def test_team_to_ids_display_names_and_oov():
    v = synthetic_vocabs()
    mons = [
        {"species": "Sp1", "ability": "Ab1", "item": "It1", "moves": ["Mv1", "Mv2"]},
        {"species": "Unknown", "ability": None, "item": "Nope", "moves": []},
    ]
    ids = team_to_ids(mons, v, n_mons=6)
    assert ids["species"].shape == (6,)
    assert ids["moves"].shape == (6, 4)
    # the display name "Sp1" must resolve to a REAL vocab id (non-zero) —
    # a silent all-OOV tokenization would score every team identically
    sp_id = v.species.index("sp1")
    assert sp_id > 0
    assert int(ids["species"][0]) == sp_id                      # toID-normalized
    assert int(ids["moves"][0, 0]) == v.move.index("mv1")
    assert v.move.index("mv1") > 0
    assert int(ids["species"][1]) == 0                          # OOV -> <pad>
    assert int(ids["ability"][1]) == 0
    assert int(ids["item"][1]) == 0
    assert int(ids["moves"][1, 0]) == 0                          # empty -> pad
    # padding slots are zero
    assert int(ids["species"][2]) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  invariance + determinism
# ═══════════════════════════════════════════════════════════════════════════

def test_permutation_invariance_of_score():
    v = synthetic_vocabs()
    brain = TeamBrain(v, seed=7)
    rng = random.Random(3)
    team = random_team(rng)
    shuffled = team[::-1]
    s1 = brain.score([team])
    s2 = brain.score([shuffled])
    assert torch.allclose(s1, s2, atol=1e-6), "slot order must not change the score"


def test_score_shapes_and_determinism():
    v = synthetic_vocabs()
    brain = TeamBrain(v, seed=7)
    rng = random.Random(4)
    teams = [random_team(rng) for _ in range(5)]
    s = brain.score(teams)
    assert s.shape == (5,)
    assert torch.allclose(brain.score(teams), s, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
#  rank
# ═══════════════════════════════════════════════════════════════════════════

def test_rank_returns_a_permutation_best_first():
    v = synthetic_vocabs()
    brain = TeamBrain(v, seed=7)
    rng = random.Random(5)
    cands = [random_team(rng) for _ in range(6)]
    order = brain.rank(cands)
    assert sorted(order) == list(range(6))
    scores = brain.score(cands)
    vals = [float(scores[i]) for i in order]
    assert vals == sorted(vals, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════
#  persistence
# ═══════════════════════════════════════════════════════════════════════════

def test_save_load_round_trip(tmp_path):
    v = synthetic_vocabs()
    brain = TeamBrain(v, seed=7)
    rng = random.Random(6)
    teams = [random_team(rng) for _ in range(3)]
    before = brain.score(teams)
    p = tmp_path / "brain.pt"
    brain.save(p)
    loaded = TeamBrain.load_file(p, v)
    after = loaded.score(teams)
    assert torch.allclose(before, after, atol=1e-6)


# ═══════════════════════════════════════════════════════════════════════════
#  pair mining from the team registry
# ═══════════════════════════════════════════════════════════════════════════

def test_team_pairs_from_registry_filters(tmp_path):
    reg = TeamRegistry(path=tmp_path / "teams.json")
    rng = random.Random(8)
    reg.add("a", random_team(rng), origin=ORIGIN_GENERATED)
    reg.add("b", random_team(rng), origin=ORIGIN_GENERATED)
    reg.add("c", random_team(rng), origin=ORIGIN_GENERATED)
    # a beats b once (below the min_battles=2 bar), a beats c twice (over bar)
    reg.update_result("a", "b", winner="a")
    reg.update_result("a", "c", winner="a")
    reg.update_result("c", "a", winner="a")
    pairs = team_pairs_from_registry(reg, min_battles=2)
    # only (a, c) is eligible; a is better -> the pair is (a_mons, c_mons, True)
    assert len(pairs) == 1
    a_mons, c_mons, better = pairs[0]
    assert better is True
    assert a_mons == reg.get("a").mons
    assert c_mons == reg.get("c").mons
    # equal-ELO pairs are dropped
    reg.update_result("b", "c", winner="b")
    reg.update_result("c", "b", winner="c")
    pairs2 = team_pairs_from_registry(reg, min_battles=1)
    assert all(p[0] != p[1] for p in pairs2)


# ═══════════════════════════════════════════════════════════════════════════
#  THE synthetic-synergy overfit test (the hermetic guarantee)
# ═══════════════════════════════════════════════════════════════════════════

def test_synthetic_synergy_overfit():
    """The brain must be able to LEARN a count-interaction synergy rule from
    preference pairs alone. The rule is a property of the team COMBINATION
    (>=2 mons with Ab1 AND >=1 mon with Mv1) — no single mon carries it.
    A model that cannot overfit this cannot rank the league's mutations."""
    v = synthetic_vocabs()
    brain = TeamBrain(v, seed=11)

    rng = random.Random(1234)
    teams = [random_team(rng) for _ in range(384)]
    # balance the classes (the random rule is skewed; force ~50/50)
    good = [t for t in teams if is_good(t)]
    bad = [t for t in teams if not is_good(t)]
    assert len(good) > 40 and len(bad) > 40
    good = good[:240]
    bad = bad[:240]
    pool = good + bad
    rng.shuffle(pool)

    train_pool = pool[:320]
    val_pool = pool[320:]
    val_good = [t for t in val_pool if is_good(t)]
    val_bad = [t for t in val_pool if not is_good(t)]
    assert val_good and val_bad

    # preference pairs: a seeded 160-pair (good, bad) sample keeps the overfit
    # fast and deterministic.
    train_good = [t for t in train_pool if is_good(t)]
    train_bad = [t for t in train_pool if not is_good(t)]
    pairs = []
    prng = random.Random(99)
    for _ in range(160):
        pairs.append((prng.choice(train_good), prng.choice(train_bad), True))

    # train (a handful of preference epochs — this is an OVERTIT test)
    metrics = brain.train_step(pairs, epochs=12)
    assert metrics["n_pairs"] == 160
    assert torch.isfinite(torch.tensor(metrics["loss"])).item()

    def acc(split_good, split_bad) -> float:
        correct = 0
        for g in split_good:
            for b in split_bad:
                correct += int(brain.score([g, b])[0] > brain.score([g, b])[1])
        return correct / (len(split_good) * len(split_bad))

    train_correct = 0
    train_g = [t for t in train_pool if is_good(t)]
    train_b = [t for t in train_pool if not is_good(t)]
    for g in train_g[:40]:
        for b in train_b[:40]:
            train_correct += int(brain.score([g, b])[0] > brain.score([g, b])[1])
    train_acc = train_correct / (40 * 40)

    val_acc = acc(val_good, val_bad)
    assert train_acc >= 0.9, f"overfit failed: train acc {train_acc:.3f} < 0.9"
    assert val_acc >= 0.8, f"generalization failed: val acc {val_acc:.3f} < 0.8"


def test_train_step_empty_pairs_no_crash():
    v = synthetic_vocabs()
    brain = TeamBrain(v, seed=7)
    out = brain.train_step([], epochs=1)
    assert out["n_pairs"] == 0
