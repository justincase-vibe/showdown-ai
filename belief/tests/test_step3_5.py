"""Step 3.5 gate — sampling + move probabilities + correlation preservation.

Run from the project root:

    python -m pytest belief/tests -v

Sub-step 3.5 (PLAN.md §3 / §3a "3.5 — sampling"): give the belief a way to turn
the posterior into a decision. Two views over the SAME posterior:

  * ``SlotBelief.sample(n, seed)`` — a SEEDABLE multinomial draw of FULL sets.
    Full sets (not per-move marginals) is the whole point: within-set move
    CORRELATIONS are preserved by construction — a Wish set you draw already
    carries its Protect (user invariant 3.3/3.5: "Wish means 90% Protect", and
    moves are correlated, not independent). Seeded ⇒ reproducible; fresh
    ``random.Random(seed)`` ⇒ no RNG state leaks across calls or slots; returns
    COPIES ⇒ mutating a sample cannot corrupt the live belief.
  * ``SlotBelief.move_probabilities()`` — posterior MASS over the sets
    CONTAINING each move (membership, order-blind per the 3.3 moveset-is-a-set
    invariant). Every value is in [0,1]; they sum to the posterior-expected
    number of moves (4 for 4-move sets), NOT 1 — a mon carries several moves.
    PP is unobservable in the log, so it is deliberately ignored.

The spec test is ``test_correlation_preserved_wish_protect``: with a 0.9 Wish
set that also runs Protect and a 0.1 Wish set that doesn't, P(Protect|Wish) must
be 0.9 — the correlation survives the sampling, it is not averaged away.
"""
from __future__ import annotations

import pytest

from belief.state import SlotBelief

SLOT = "p2a"
SPECIES = "blastoise"


def sb(cands) -> SlotBelief:
    """A slot belief over a controlled candidate set (SlotBelief normalizes)."""
    return SlotBelief(SLOT, SPECIES, cands)


def cand(moves, weight=1.0, **kw):
    base = {"moves": list(moves), "item": "", "ability": "x", "nature": "Hardy",
            "evs": {}, "tera": None, "weight": weight}
    base.update(kw)
    return base


# ── sample(n, seed) ──────────────────────────────────────────────────────────

def test_sample_reproducible_with_seed():
    cands = [cand(["a", "b", "c", "d"], 0.5), cand(["w", "x", "y", "z"], 0.5)]
    s1 = sb(list(cands))
    s2 = sb(list(cands))
    a = s1.sample(50, seed=12345)
    b = s2.sample(50, seed=12345)
    assert a == b, "same seed + same posterior must give identical samples"
    # stateless across calls on the same object
    assert s1.sample(50, seed=12345) == a
    # a different seed gives (statistically) a different draw
    c = s1.sample(50, seed=99999)
    assert c != a


def test_sample_matches_posterior():
    # three candidates, target mass 0.5 / 0.3 / 0.2 (normalized from 5/3/2)
    cands = [cand(["markA", "b", "c", "d"], 5.0),
             cand(["markB", "b", "c", "d"], 3.0),
             cand(["markC", "b", "c", "d"], 2.0)]
    s = sb(cands)
    n = 20000
    draws = s.sample(n, seed=1)
    frac = {m: sum(1 for d in draws if m in d["moves"]) / n
            for m in ("markA", "markB", "markC")}
    target = {"markA": 0.5, "markB": 0.3, "markC": 0.2}
    for m, t in target.items():
        assert abs(frac[m] - t) <= 0.03, f"{m}: {frac[m]:.4f} vs target {t}"
    assert abs(sum(frac.values()) - 1.0) <= 0.05


def test_sample_returns_copies():
    cands = [cand(["a", "b", "c", "d"], 1.0)]
    s = sb(cands)
    before = [c["weight"] for c in s.candidates]
    before_moves = [list(c["moves"]) for c in s.candidates]
    sample = s.sample(1)[0]
    sample["moves"].append("EVIL")
    sample["weight"] = -1.0
    assert [c["weight"] for c in s.candidates] == before
    assert [list(c["moves"]) for c in s.candidates] == before_moves


def test_sample_negative_raises():
    s = sb([cand(["a", "b", "c", "d"], 1.0)])
    with pytest.raises(ValueError):
        s.sample(-1)


def test_sample_respects_ruled_out():
    # one candidate is effectively ruled out (mass ~0) — it is essentially
    # never drawn even though it is present (the EPS floor keeps it non-empty).
    s = sb([cand(["alive", "b", "c", "d"], 1.0),
            cand(["dead", "b", "c", "d"], 1e-6)])
    draws = s.sample(1000, seed=0)
    dead = sum(1 for d in draws if "dead" in d["moves"])
    assert dead == 0, f"ruled-out candidate drawn {dead}/1000 times"


# ── move_probabilities() ─────────────────────────────────────────────────────

def test_move_probabilities_membership_orderblind():
    # identical posterior, one candidate's move ORDER differs — membership mass
    # must be identical (moveset-is-a-set invariant).
    A = cand(["wish", "protect", "foulplay", "struggle"], 0.9)
    B = cand(["wish", "earthquake", "surf", "slash"], 0.1)
    A_reordered = cand(["struggle", "foulplay", "protect", "wish"], 0.9)
    p1 = sb([A, B]).move_probabilities()
    p2 = sb([A_reordered, B]).move_probabilities()
    assert p1 == p2


def test_move_probabilities_range_and_sum():
    # disjoint 4-move sets → every value in [0,1], sum ≈ 4 (expected moves)
    s = sb([cand(["a", "b", "c", "d"], 0.5),
            cand(["w", "x", "y", "z"], 0.5)])
    p = s.move_probabilities()
    for v in p.values():
        assert 0.0 <= v <= 1.0
    assert abs(sum(p.values()) - 4.0) <= 1e-9


# ── the spec test: correlation is preserved, not averaged away ───────────────

def test_correlation_preserved_wish_protect():
    # setA (0.9): Wish+Protect  ·  setB (0.1): Wish, no Protect
    A = cand(["wish", "protect", "foulplay", "struggle"], 0.9)
    B = cand(["wish", "earthquake", "surf", "slash"], 0.1)
    s = sb([A, B])
    p = s.move_probabilities()
    assert abs(p["wish"] - 1.0) <= 1e-9      # every set has Wish
    assert abs(p["protect"] - 0.9) <= 1e-9   # only the 0.9 set has Protect
    # P(Protect | Wish) = mass(Wish ∧ Protect) / mass(Wish)
    mp = s.mass(lambda c: "wish" in c["moves"] and "protect" in c["moves"])
    mw = s.mass(lambda c: "wish" in c["moves"])
    assert abs(mp / mw - 0.9) <= 1e-9


def test_sample_preserves_correlation():
    A = cand(["wish", "protect", "foulplay", "struggle"], 0.9)
    B = cand(["wish", "earthquake", "surf", "slash"], 0.1)
    s = sb([A, B])
    draws = s.sample(2000, seed=7)
    with_wish = [d for d in draws if "wish" in d["moves"]]
    assert with_wish  # every set carries Wish, so this is all of them
    frac = sum(1 for d in with_wish if "protect" in d["moves"]) / len(with_wish)
    assert 0.82 <= frac <= 0.97, f"P(Protect|Wish) from samples = {frac:.4f}"


def test_marginals_are_not_independent():
    """The audit finding: move_probabilities() are MARGINALS, and multiplying them
    to get a joint is the independence error the user explicitly forbade ("the moves
    are correlated"). Pin that the CORRECT joint comes from the full sets (sample/mass),
    not from P(a)·P(b)."""
    # A (0.5): Wish+Protect  ·  B (0.5): no Wish, no Protect
    s = sb([cand(["wish", "protect", "a", "b"], 0.5),
            cand(["earthquake", "surf", "c", "d"], 0.5)])
    p = s.move_probabilities()
    joint = s.mass(lambda c: "wish" in c["moves"] and "protect" in c["moves"])
    naive = p["wish"] * p["protect"]
    assert abs(joint - 0.5) <= 1e-9       # true joint from the full sets
    assert abs(naive - 0.25) <= 1e-9      # the (wrong) independence product
    assert abs(joint - naive) > 0.2, "joint must differ from the naive product"
    # and sample() reproduces the CORRELATED joint, not the naive one:
    draws = s.sample(20000, seed=11)
    frac = sum(1 for d in draws if "wish" in d["moves"] and "protect" in d["moves"]) / len(draws)
    assert 0.45 <= frac <= 0.55, f"empirical P(Wish∧Protect) = {frac:.4f} (naive would be ~0.25)"
