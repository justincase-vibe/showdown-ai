"""PROOF: P0 fix — reflector moves no longer collapse the belief posterior.

Before the fix: Counter returned immune/[0,0] → all candidates floored to EPS
→ posterior collapsed to uniform, destroying prior evidence.

After the fix: the calc receives lastHitDamage from the attacker's facts,
returns a valid window (2× prior hit), and the posterior is preserved.

The correct test: a PRIOR hit narrows the candidates, then a Counter arrives.
Before: the Counter WIPED the narrowing (collapse). After: it's preserved.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from belief import events as E
from belief import evidence as ev
from belief.state import BeliefState, SlotBelief


def cand(**kw):
    base = {"moves": ["tackle", "a", "b", "c"], "item": "", "ability": "x",
            "nature": "Hardy", "evs": {}, "tera": None, "weight": 1.0}
    base.update(kw)
    return base


def install(bs, slot, species, cands):
    sb = SlotBelief(slot, species, cands)
    bs._slots[slot] = sb
    return sb


def main():
    # ── Setup: own Charizard (exactly known) + opponent Blastoise ──
    own = {"p1a": {"name": "charizard", "nature": "Modest", "evs": {"spa": 252},
                   "item": "lifeorb", "ability": "solarpower"}}

    def fresh():
        bs = BeliefState("gen9ou", own_team=own)
        sb = install(bs, "p2a", "blastoise",
                     [cand(evs={"spd": 0}), cand(evs={"spd": 252})])
        return bs, sb

    # ── Step 1: Prior move hit discriminates the two SpD candidates ──
    bs, sb = fresh()
    # delta 95 in 0-SpD [88,105], outside 252-SpD [70,84] (from existing test)
    bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=204, maxhp=299,
        ko=False, source_kind="move", move="flamethrower",
        source_side="p1", source_slot="p1a"))
    w_after_prior = [round(c["weight"], 4) for c in sb.candidates]
    print(f"After prior hit (discriminating): {w_after_prior}")
    assert w_after_prior[0] > 0.99, "Prior hit should narrow to candidate 0"

    # ── Step 2: Blastoise uses... well, Blastoise can't Counter.
    #    Let's use a mon that CAN: switch to Hitmonchan.
    #    Actually, the simplest correct test: use the existing defender test
    #    setup but with a REFLECTOR as the second event.
    #
    #    Since Counter is always 2× prior_hit (stat-independent), it can't
    #    discriminate between candidates. The BUG was that it DESTROYED the
    #    prior narrowing. The fix ensures it PRESERVES it (both get 1.0 →
    #    unchanged weights).
    #
    #    Let's verify: after the fix, the narrowed posterior is PRESERVED.
    bs, sb = fresh()
    bs.update(E.DamageObserved(
        target_side="p2", target_slot="p2a", hp_after=204, maxhp=299,
        ko=False, source_kind="move", move="flamethrower",
        source_side="p1", source_slot="p1a"))
    w_before_counter = [round(c["weight"], 4) for c in sb.candidates]

    # Now a Counter from the opponent (p2a) hits our mon (p1a)
    # Counter reflects 2 × last_hit = 2 × 95 = 190
    bs.update(E.DamageObserved(
        target_side="p1", target_slot="p1a", hp_after=109, maxhp=299,
        ko=False, source_kind="move", move="counter",
        source_side="p2", source_slot="p2a"))
    w_after_counter = [round(c["weight"], 4) for c in sb.candidates]
    print(f"After Counter (should PRESERVE narrowing): {w_after_counter}")

    # ── Verdict ──
    if w_after_counter[0] > 0.99:
        print("\n*** FIX CONFIRMED: Posterior narrowing preserved after Counter ***")
        print("    The calc received lastHitDamage and returned a valid window.")
        print("    Both candidates got likelihood 1.0 (Counter is stat-independent),")
        print("    so the prior evidence (Flamethrower narrowing) is intact.")
    elif all(0.4 < w < 0.6 for w in w_after_counter):
        print("\n*** BUG STILL PRESENT: Posterior collapsed to uniform ***")
        print("    The calc returned immune/[0,0] → all candidates floored to EPS.")
    else:
        print(f"\nUnexpected weights: {w_after_counter}")

    # ── Direct calc verification ──
    print("\n--- Direct calc check ---")
    from damage import calc
    att = {"name": "blastoise", "level": 50, "nature": "Hardy", "evs": {}}
    dfn = {"name": "charizard", "level": 50, "nature": "Modest", "evs": {"spa": 252}}
    r_with = calc.calc_damage(att, dfn, move={"name": "counter", "lastHitDamage": 95}, fmt="gen9ou")
    r_without = calc.calc_damage(att, dfn, move="counter", fmt="gen9ou")
    print(f"  With lastHitDamage=95:  immune={r_with.get('immune')}, range=[{r_with.get('min')},{r_with.get('max')}]")
    print(f"  Without lastHitDamage:  immune={r_without.get('immune')}, range=[{r_without.get('min')},{r_without.get('max')}]")


if __name__ == "__main__":
    main()
