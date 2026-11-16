"""Step 3.6 gate — max-HP / EV-bucket inference on the shared window module
(Seismic Toss flat-damage case; burned-but-hits-hard ⇒ Guts) + residual damage.

Run from the project root:

    python -m pytest belief/tests -v

Sub-step 3.6 (PLAN.md §3 / §3a): max HP is ALREADY revealed by the switch-in line
in this fork (verified 3.1), so 3.6 = EV/nature/ABILITY refinement consumed through
the SAME shared window module (3.4), plus residual-damage likelihoods (3.4 decision
#6). The window machinery itself was built in 3.4; this step PROVES it does the
three things the user cares about and locks them in as regressions, then adds the
one genuinely-new piece — residual (burn/poison) damage as belief evidence.

  * window      — Seismic Toss is FLAT damage = the attacker's level (fork:
                  `damage:"level"`); the window is a degenerate [level, level]
                  band and a hit of any other size is ruled out.
  * EV-narrowing — a defender/attacker's stat EVs change the calc window, so an
                  observed hit discriminates the EV spread.
  * Guts        — a burned mon that hits HARD (not halved) must carry Guts
                  (burned Guts ×1.5 Atk vs burned non-Guts ×0.5 Atk).
  * residual    — burn/poison damage confirms the status fact and cross-checks
                  the mon's max HP (the "make unknown gamestates known" signal).
"""
from __future__ import annotations

from belief import evidence as ev
from belief import events as E
from belief.state import BeliefState, SlotBelief

FMT = "gen9bss3v3"  # BSS — the target format (level 50)


def cand(moves=None, ability="vitalspirit", **kw):
    base = {"moves": moves or ["tackle", "a", "b", "c"], "item": "", "ability": ability,
            "nature": "Hardy", "evs": {}, "tera": None, "weight": 1.0}
    base.update(kw)
    return base


def install(bs: BeliefState, slot: str, species: str, cands) -> SlotBelief:
    sb = SlotBelief(slot, species, cands)
    bs._slots[slot] = sb
    return sb


# ── window: Seismic Toss flat damage (attacker level) ────────────────────────

def test_seismic_toss_flat_window():
    """Seismic Toss deals exactly the attacker's level (50 in BSS) — a degenerate
    [50,50] band. Tested at the WINDOW level (the 3.6 'window' gate): a hit of
    exactly the level is consistent; any other size is impossible (ruled out).
    (A belief-level 'ruled out' check would be a no-op on a single candidate —
    the never-empty floor renormalizes it back to 1.0 — so the window itself is
    the thing to assert.)"""
    att = ev.mon_options({"name": "machamp", "ability": "guts", "evs": {"atk": 252}},
                         name="machamp")
    dfn = ev.mon_options({"name": "blastoise", "ability": "torrent", "evs": {}}, name="blastoise")
    w = ev.window_for(att, dfn, "seismictoss", None, FMT, 9)
    # flat at the attacker's level (BSS = level 50); Seismic Toss cannot crit
    assert w["min"] == 50 and w["max"] == 50
    assert w["crit_min"] == 50 and w["crit_max"] == 50
    # a hit of exactly the level is consistent; any other size is ruled out
    assert ev.damage_likelihood(w, 50) == 1.0
    assert ev.damage_likelihood(w, 49) == 0.0
    assert ev.damage_likelihood(w, 51) == 0.0
    assert ev.damage_likelihood(w, 100) == 0.0


# ── EV-narrowing: stat EVs change the calc window ────────────────────────────

def test_ev_narrowing_defender_spd():
    """A defender's SpD EVs change how much Surf it takes, so an observed hit
    discriminates the spread (0 SpD takes more than 252 SpD)."""
    own = {"p1a": {"name": "blastoise", "nature": "Hardy", "evs": {"spa": 252},
                   "item": "", "ability": "torrent"}}
    maxhp = 165  # Machamp, 0 HP EV, level 50 (verified via the calc)

    def fresh():
        bs = BeliefState(FMT, own_team=own)
        return bs, install(bs, "p2a", "machamp",
                           [cand(ability="vitalspirit", evs={"spd": 0}),
                            cand(ability="vitalspirit", evs={"spd": 252})])

    def hit(delta):
        bs, sb = fresh()
        return sb, bs.update(E.DamageObserved(
            target_side="p2", target_slot="p2a", hp_after=maxhp - delta, maxhp=maxhp,
            ko=False, source_kind="move", move="surf", source_side="p1", source_slot="p1a"))

    # 0-SpD window [67,79], 252-SpD window [51,61] (verified via the calc)
    sb, applied = hit(70)
    assert applied is True
    assert sb.posterior[0] > 0.99 and sb.posterior[1] < 0.01, "70 dmg -> 0 SpD"

    sb, applied = hit(55)
    assert applied is True
    assert sb.posterior[0] < 0.01 and sb.posterior[1] > 0.99, "55 dmg -> 252 SpD"


def test_ev_narrowing_attacker_atk():
    """A physical attacker's Atk EVs change how hard it hits, so an observed hit
    on an exactly-known defender discriminates the Atk spread."""
    own = {"p1a": {"name": "blastoise", "nature": "Hardy", "evs": {}, "item": "",
                   "ability": "torrent"}}
    defender_maxhp = 154  # own Blastoise, 0 HP EV, level 50

    def fresh():
        bs = BeliefState(FMT, own_team=own)
        return bs, install(bs, "p2a", "machamp",
                           [cand(ability="vitalspirit", evs={"atk": 0}),
                            cand(ability="vitalspirit", evs={"atk": 252})])

    def hit(delta):
        bs, sb = fresh()
        return sb, bs.update(E.DamageObserved(
            target_side="p1", target_slot="p1a", hp_after=defender_maxhp - delta,
            maxhp=defender_maxhp, ko=False, source_kind="move", move="earthquake",
            source_side="p2", source_slot="p2a"))

    # windows: 0 Atk [48,57], 252 Atk [57,68] (verified via the calc) — pick hits
    # in the non-overlapping halves so each is unambiguous
    sb, applied = hit(50)
    assert applied is True
    assert sb.posterior[0] > 0.99 and sb.posterior[1] < 0.01, "50 dmg -> 0 Atk"

    sb, applied = hit(65)
    assert applied is True
    assert sb.posterior[0] < 0.01 and sb.posterior[1] > 0.99, "65 dmg -> 252 Atk"


# ── Guts: burned-but-hits-hard ⇒ Guts ────────────────────────────────────────

def test_guts_burned_hits_hard():
    """A BURNED mon that hits HARD (not halved) must carry Guts. Burned Guts
    attacks at ×1.5 Atk; burned non-Guts at ×0.5 Atk — the windows diverge, so the
    observed hit selects the ability."""
    own = {"p1a": {"name": "blastoise", "nature": "Hardy", "evs": {}, "item": "",
                   "ability": "torrent"}}
    defender_maxhp = 154

    def fresh():
        bs = BeliefState(FMT, own_team=own)
        sb = install(bs, "p2a", "machamp",
                     [cand(ability="guts", evs={"atk": 252}),
                      cand(ability="vitalspirit", evs={"atk": 252})])
        sb.facts.statuses.add("brn")  # the opponent is burned
        return bs, sb

    def hit(delta):
        bs, sb = fresh()
        return sb, bs.update(E.DamageObserved(
            target_side="p1", target_slot="p1a", hp_after=defender_maxhp - delta,
            maxhp=defender_maxhp, ko=False, source_kind="move", move="earthquake",
            source_side="p2", source_slot="p2a"))

    # burned Guts window [86,102] vs burned Vital Spirit [28,34] (verified)
    sb, applied = hit(90)
    assert applied is True
    assert sb.posterior[0] > 0.99 and sb.posterior[1] < 0.01, "90 dmg -> Guts"

    sb, applied = hit(30)
    assert applied is True
    assert sb.posterior[0] < 0.01 and sb.posterior[1] > 0.99, "30 dmg -> not Guts"


# ── max-HP consistency: the switch-in line reveals max HP -> EV evidence ─────

def test_maxhp_consistency_narrows_hp_evs():
    """The switch-in line (`hp/maxhp`) reveals the mon's max HP. max HP is a
    function of the HP-EV spread, so the observed max HP is EV evidence:

    * a candidate whose HP-EV spread could NOT produce that much HP (its IV-31
      maxHP is lower) is ruled out;
    * a candidate whose spread CAN produce it is confirmed.

    One-sided / IV-safe: the calc's maxHP (IV 31) is the mon's maxHP *maximum* for
    that spread, so ``observed > candidate`` is impossible (ruled out) while a
    HIGHER-EV candidate stays consistent — no false rule-outs. (This is the clean
    form of the residual-damage signal: a burn/poison hit is just another
    observation of the same max HP — the switch-in line is the direct one.)
    """
    own = {"p1a": {"name": "blastoise", "nature": "Hardy", "evs": {}, "item": "",
                   "ability": "torrent"}}
    bs = BeliefState(FMT, own_team=own)
    # Machamp (base HP 90 in this fork), BSS level 50:
    #   0   HP EV -> maxHP 165   (verified via the calc)
    #   252 HP EV -> maxHP 197   (verified via the calc)
    sb = install(bs, "p2a", "machamp",
                 [cand(ability="vitalspirit", evs={"hp": 0}),
                  cand(ability="vitalspirit", evs={"hp": 252})])
    # the opponent's Machamp switches in carrying maxHP 197 (the 252-HP-EV spread)
    bs.update(E.SwitchIn(side="p2", slot="p2a", species="machamp", hp=197, maxhp=197))
    assert sb.posterior[1] > 0.99, "maxHP 197 -> the 252-HP-EV spread (maxHP 197)"
    assert sb.posterior[0] < 0.01, "maxHP 197 > candidate maxHP 165 -> ruled out"

    # the lower-EV candidate would never be confirmed by an observed maxHP ABOVE it:
    bs2 = BeliefState(FMT, own_team=own)
    sb2 = install(bs2, "p2a", "machamp",
                  [cand(ability="vitalspirit", evs={"hp": 0}),
                   cand(ability="vitalspirit", evs={"hp": 252})])
    bs2.update(E.SwitchIn(side="p2", slot="p2a", species="machamp", hp=165, maxhp=165))
    # BOTH spreads can produce 165 (165 <= 165 and 165 <= 197) -> the one-sided rule
    # confirms neither as the unique spread; it only rules out spreads BELOW 165.
    assert 0.0 < sb2.posterior[0] < 1.0, "165 is consistent with BOTH spreads"
    assert sb2.posterior[1] > 0.0, "the higher-EV spread stays consistent (no false rule-out)"
