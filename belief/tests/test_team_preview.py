"""Step 1 of the bench-aware net — team-preview → the net sees the FULL team.

The opponent's bench was invisible to the net: in singles the field slots only
ever hold the active mon (p2a — the player picks the lead during the preview
phase, so preview order != field order), so the 5 bench mons — the very mons the
net must plan coverage against (Garchomp @80% vs a confirmed-slow Blastoise,
fast Fairy + Gliscor benched -> play coverage) — were never registered. This
locks in the fix:

  * the adapter maps ``|poke|`` team-preview lines to ``TeamPreview`` events
    (the opponent's 6 species, preview order p2a..p2f);
  * the belief accumulates the opponent team (``self.opp_team``);
  * ``get_context()`` places the active at its field slot (p2a) and the ALIVE
    bench at p2b..p2f (preview order, minus the active and any fainted mon);
  * the net therefore sees all 6 opponent species (active + bench) in the
    EXISTING ``opp_species`` slots — NO new dimension (the registry checksum is
    unchanged, checkpoints stay loadable);
  * the ACTIVE mon's damage/max-HP EV inference (3.6) is preserved — the
    preview's pre-registration must not clobber the active's posterior.

Run from the project root:

    CUDA_VISIBLE_DEVICES=-1 python -m pytest belief/tests/test_team_preview.py -v
"""
from __future__ import annotations

from belief import events as E
from belief.log_events import events_from_log
from belief.state import BeliefState, SlotBelief

FMT = "gen9ou"  # a real format with priors (build_prior works here)


def _feed(lines):
    """Build a belief from a raw log (adapter -> belief), exactly like
    ``net/env.build_obs`` does."""
    from belief.log_events import LogAdapter
    bs = BeliefState(FMT)
    adapter = LogAdapter()
    for line in lines:
        ev = adapter.feed_line(line)
        if ev is None or isinstance(ev, E.Unknown):
            continue
        bs.update(ev)
    return bs


# ── adapter: |poke| -> TeamPreview ────────────────────────────────────────────

def test_adapter_maps_poke_to_team_preview():
    evs = events_from_log([
        "|poke|p2|Kyurem|",
        "|poke|p2|Tinkaton, F|",
        "|poke|p2|Blaziken, M|",
    ])
    tp = [e for e in evs if e.type == "team_preview"]
    assert [e.side for e in tp] == ["p2", "p2", "p2"]
    assert [e.slot for e in tp] == ["p2a", "p2b", "p2c"]  # preview order
    assert [e.species for e in tp] == ["kyurem", "tinkaton", "blaziken"]
    # the phase marker |teampreview| (no payload) is still pure noise
    assert all(e.type != "team_preview"
               for e in events_from_log(["|teampreview|3"]))


def test_adapter_poke_slot_shape():
    # the rare explicit-slot shape also maps (side = the 2-char prefix)
    evs = events_from_log(["|poke| p2c 1 | Great Tusk |"])
    tp = [e for e in evs if e.type == "team_preview"]
    assert len(tp) == 1
    assert tp[0].slot == "p2c" and tp[0].species == "greattusk"


def test_adapter_poke_is_not_unknown_or_noise():
    # |poke| must now surface as a TeamPreview (NOT Unknown, NOT silently dropped)
    evs = events_from_log(["|poke|p2|Kyurem|"])
    assert [e.type for e in evs] == ["team_preview"]


# ── belief: registers the full opponent team ─────────────────────────────────

def test_belief_registers_full_opp_team():
    bs = _feed([
        "|poke|p2|Kyurem|", "|poke|p2|Tinkaton|", "|poke|p2|Blaziken|",
        "|poke|p2|Great Tusk|", "|poke|p2|Sinistcha|", "|poke|p2|Darkrai|",
    ])
    assert bs.opp_team == ["kyurem", "tinkaton", "blaziken", "greattusk",
                           "sinistcha", "darkrai"]


def test_belief_ignores_p1_previews():
    # the own side (p1) is fully known from the own_team spec — p1 previews are
    # inert (they must NOT leak into the opponent team comp)
    bs = _feed(["|poke|p1|Garchomp|", "|poke|p2|Kyurem|"])
    assert bs.opp_team == ["kyurem"]


# ── get_context: active + bench all visible ───────────────────────────────────

def test_context_shows_all_six_opp_species():
    # The lead is Blaziken (preview #3), NOT Kyurem (preview #1) — the player
    # picks the lead during the preview phase, so preview order != field order.
    # A naive "preview order -> field slot" mapping would misplace the active.
    bs = _feed([
        "|poke|p2|Kyurem|", "|poke|p2|Tinkaton|", "|poke|p2|Blaziken|",
        "|poke|p2|Great Tusk|", "|poke|p2|Sinistcha|", "|poke|p2|Darkrai|",
        "|start",
        "|switch|p2a: Blaziken|Blaziken, M|100/100",
    ])
    ctx = bs.get_context()
    opp = ctx["opponent"]
    # the active is at its field slot (p2a) with its real species + full HP
    assert opp["p2a"]["species"] == "blaziken"
    assert opp["p2a"]["hp_frac"] == 1.0
    # all six team species are present across the six slots (active + 5 bench)
    species = sorted(opp[slot]["species"]
                     for slot in ("p2a", "p2b", "p2c", "p2d", "p2e", "p2f"))
    assert species == sorted(["kyurem", "tinkaton", "blaziken",
                              "greattusk", "sinistcha", "darkrai"])
    # the bench is benched: present, not active, not fainted, hp 0
    for slot in ("p2b", "p2c", "p2d", "p2e", "p2f"):
        assert opp[slot]["hp_frac"] == 0.0
        assert opp[slot]["fainted"] is False


def test_context_no_preview_still_works():
    # a log without a team preview: the bench is empty (unchanged behavior) and
    # the active is still registered on switch-in (no regression for old logs)
    bs = _feed(["|start", "|switch|p2a: Blastoise|Blastoise, M|150/150"])
    ctx = bs.get_context()
    assert ctx["opponent"]["p2a"]["species"] == "blastoise"
    assert "p2b" not in ctx["opponent"]  # no preview -> no bench


def test_bench_excludes_fainted_mon():
    # a team mon that was active and fainted is GONE — it must not be offered as
    # switchable bench (the net would otherwise plan a switch into a corpse)
    bs = _feed([
        "|poke|p2|Garchomp|", "|poke|p2|Blastoise|", "|poke|p2|Tinkaton|",
        "|start",
        "|switch|p2a: Blastoise|Blastoise, M|150/150",
        "|faint|p2a: Blastoise",
        "|switch|p2a: Tinkaton|Tinkaton|100/100",
    ])
    ctx = bs.get_context()
    present = {ctx["opponent"][s]["species"] for s in
               ("p2a", "p2b", "p2c", "p2d", "p2e", "p2f")
               if s in ctx["opponent"]}
    assert "tinkaton" in present   # the new active
    assert "garchomp" in present   # the alive bench
    assert "blastoise" not in present  # fainted -> gone, NOT switchable


# ── the ACTIVE mon's EV inference is preserved after the preview ─────────────

def _atk_cand(atk):
    return {"moves": ["earthquake", "a", "b", "c"], "item": "",
            "ability": "vitalspirit", "nature": "Hardy", "evs": {"atk": atk},
            "tera": None, "weight": 1.0}


def test_preview_does_not_clobber_active_ev_inference():
    """The TeamPreview (which sets the opp team comp) must NOT disturb the
    ACTIVE mon's set posterior — its damage EV inference (3.6) still works
    after the preview has registered the team. (The preview handler only touches
    ``self.opp_team``; this pins that it can't leak into the active's belief.)"""
    from belief import events as E

    FMT36 = "gen9bss3v3"  # the 3.6 target (level 50), used by test_step3_6
    own = {"p1a": {"name": "blastoise", "nature": "Hardy", "evs": {},
                   "item": "", "ability": "torrent"}}
    defender_maxhp = 154  # own Blastoise, 0 HP EV, level 50 (test_step3_6)

    def fresh():
        bs = BeliefState(FMT36, own_team=own)
        # the preview registers the FULL opp team (the bench) ...
        bs.update(E.TeamPreview(side="p2", slot="p2a", species="machamp"))
        bs.update(E.TeamPreview(side="p2", slot="p2b", species="tinkaton"))
        assert bs.opp_team == ["machamp", "tinkaton"]
        # ... but must NOT clobber the active's set posterior:
        sb = SlotBelief("p2a", "machamp", [_atk_cand(0), _atk_cand(252)])
        bs._slots["p2a"] = sb
        return bs, sb

    def hit(delta):
        bs, sb = fresh()
        return sb, bs.update(E.DamageObserved(
            target_side="p1", target_slot="p1a", hp_after=defender_maxhp - delta,
            maxhp=defender_maxhp, ko=False, source_kind="move", move="earthquake",
            source_side="p2", source_slot="p2a"))

    # windows 0 Atk [48,57] vs 252 Atk [57,68] (test_step3_6) — the inference
    # still discriminates the Atk spread after the preview
    sb, applied = hit(50)
    assert applied is True
    assert sb.posterior[0] > 0.99 and sb.posterior[1] < 0.01, "50 dmg -> 0 Atk"
    sb, applied = hit(65)
    assert applied is True
    assert sb.posterior[0] < 0.01 and sb.posterior[1] > 0.99, "65 dmg -> 252 Atk"
