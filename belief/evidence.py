"""Step 3.2 — shared evidence engine.

ONE module (user requirement: nothing written twice) that turns a single piece
of observed evidence into a per-candidate LIKELIHOOD and updates the posterior.
It is used by BOTH the set posterior (3.3–3.5) and the HP/EV window inference
(3.6), so the two never drift apart.

Design (PLAN.md §3, 3.2 + 3.3):
  * An evidence = a likelihood function ``candidate -> [0, 1]``.
  * HARD evidence (revealed move / ability / item, Tera, Choice-lock) is exact
    0/1 — a candidate either satisfies it or it doesn't.
  * SOFT evidence (damage window, speed order — implemented in 3.4 over the Step-2
    calc) is (0, 1]; the mechanism is the same, only the returned range differs.
  * "unknown is not disconfirmed": a candidate whose dimension is UNKNOWN
    (``item == ""`` / ``tera is None``) is CONSISTENT with a reveal of that
    dimension (likelihood 1) — absence of information is not evidence of a
    specific value. A candidate that pins a SPECIFIC value must match it.
  * EPS floor: no likelihood may be exactly 0 — ``apply_evidence`` floors every
    likelihood at ``EPS`` before multiplying, so the posterior can NEVER empty
    out (off-DB sets survive; the AI always has a normalizable belief to act on).
    After renormalization a HARD non-match reads ~EPS (≈0) while a match reads
    ~1 — so HARD evidence is effectively decisive without ever collapsing the
    distribution to all-zeros.

Pure math over the candidate dicts — no DB, no damage calc, no fork — so it is
trivially unit-testable and reusable.
"""
from __future__ import annotations

from typing import Callable, Dict, List

#: EPS floor — no evidence may fully rule out a candidate. Small enough that a
#: HARD non-match is ≈0 after renormalization; large enough to avoid float
#: underflow (we renormalize after every update, so no accumulation).
EPS = 1e-9

#: Choice items — using the SAME move twice means the item is locked to one of
#: these (a non-Choice item allows switching moves freely).
CHOICE_ITEMS = frozenset({"choiceband", "choicescarf", "choicespecs"})

#: A likelihood: candidate dict -> float in [0, 1].
Likelihood = Callable[[Dict], float]


def _floor(lik: float, eps: float) -> float:
    if lik < 0.0:
        raise ValueError(f"likelihood must be >= 0, got {lik}")
    return max(lik, eps)


def apply_evidence(candidates: List[Dict], likelihood: Likelihood,
                   eps: float = EPS) -> List[Dict]:
    """Update ``candidates`` weights IN PLACE with one piece of evidence.

    ``weight_i <- weight_i * max(likelihood(c_i), eps)``, then renormalize so
    the weights sum to exactly 1. Returns the same (mutated) list.

    The EPS floor guarantees the posterior never empties: even if the evidence
    is inconsistent with EVERY candidate (an off-DB set), all weights floor at
    ``eps`` and renormalize to a uniform, still-normalizable belief.
    """
    if not candidates:
        raise ValueError("apply_evidence: empty candidate list")
    for c in candidates:
        c["weight"] *= _floor(likelihood(c), eps)
    total = sum(c["weight"] for c in candidates)
    if total <= 0.0:  # defensive — impossible with a positive prior + eps floor
        n = len(candidates)
        for c in candidates:
            c["weight"] = 1.0 / n
        return candidates
    for c in candidates:
        c["weight"] /= total
    return candidates


# ── HARD evidence (exact 0/1; the eps floor is applied by apply_evidence) ──

def revealed_move(move: str) -> Likelihood:
    """The set contains the revealed move. Membership test — ORDER-BLIND (a
    moveset is a set: wish+protect+foulplay = foulplay+protect+wish)."""
    def lik(c: Dict) -> float:
        return 1.0 if move in c.get("moves", []) else 0.0
    return lik


def revealed_ability(ability: str) -> Likelihood:
    """The set's ability matches. (Ability is always a SPECIFIC value in every
    candidate — the fallback picks one of the species' abilities — so there is
    no "unknown" case to be lenient about.)"""
    def lik(c: Dict) -> float:
        return 1.0 if c.get("ability", "") == ability else 0.0
    return lik


def revealed_item(item: str) -> Likelihood:
    """The set's item is consistent with the reveal. A candidate with
    ``item == ""`` (unknown — the fallback models no item) is CONSISTENT
    (likelihood 1); a candidate pinning a SPECIFIC item must match it."""
    def lik(c: Dict) -> float:
        citem = c.get("item", "")
        if citem == "":
            return 1.0
        return 1.0 if citem == item else 0.0
    return lik


def tera_used(tera_type: str) -> Likelihood:
    """The set's Tera type is consistent with the reveal. ``tera is None`` means
    UNKNOWN (NOT "no Tera") -> consistent (likelihood 1); a candidate pinning a
    SPECIFIC Tera type must match it."""
    def lik(c: Dict) -> float:
        ctera = c.get("tera")
        if ctera is None:
            return 1.0
        return 1.0 if ctera == tera_type else 0.0
    return lik


def choice_lock() -> Likelihood:
    """The SAME move was used twice -> the item is locked to a Choice item.
    A candidate with an unknown item (``""``) is CONSISTENT (it could be a
    Choice item); a candidate pinning a SPECIFIC non-Choice item is ruled out;
    a candidate pinning a Choice item is confirmed."""
    def lik(c: Dict) -> float:
        citem = c.get("item", "")
        if citem == "":
            return 1.0
        return 1.0 if citem in CHOICE_ITEMS else 0.0
    return lik


#: name -> factory, so ``update()`` (3.3) can dispatch by evidence type.
HARD_EVIDENCE = {
    "revealed_move": revealed_move,
    "revealed_ability": revealed_ability,
    "revealed_item": revealed_item,
    "tera_used": tera_used,
    "choice_lock": choice_lock,
}
