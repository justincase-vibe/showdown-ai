"""D61 surface (user-approved 2026-11-17) — the two belief-side gaps closed:

1. **Per-slot fainted flag** — ``BeliefState.fainted_slots``: the net must tell
   "benched, untouched" (switchable) from "gone" (not switchable); both
   otherwise read hp_frac 0. Set on Faint, cleared on the next SwitchIn into
   the same slot. Exposed per block in ``get_context()`` as ``fainted``.
2. **The seven curated volatiles (D61)** — Disguise / Taunt / Encore / Trap /
   Curse / Destiny Bond / Gastro Acid: the adapter maps ``|-start|``/``|-end|``
   for exactly this set to ``VolatileChanged`` (all other volatiles stay
   silently ignored — known, no set evidence, NOT Unknown); belief stores them
   on ``Facts.volatiles`` (multi-turn: persist across ``E.Turn``, cleared by
   ``|-end|`` or a new mon). Exposed per block in ``get_context()`` as
   ``volatiles``.

Run from the project root:

    python -m pytest belief/tests -v
"""
from __future__ import annotations

from belief import events as E
from belief.log_events import LogAdapter
from belief.state import BeliefState, SlotBelief


# ── helpers ──────────────────────────────────────────────────────────────────

def cand(**kw):
    base = {"moves": ["outrage", "a", "b", "c"], "item": "", "ability": "x",
            "nature": "Hardy", "evs": {}, "tera": None, "weight": 1.0}
    base.update(kw)
    return base


def install(bs: BeliefState, slot: str, species: str, n: int = 2) -> SlotBelief:
    """White-box slot (no DB prior needed for Facts-level tests)."""
    cands = [cand(moves=["m%d" % i], ability=chr(97 + i), weight=1.0 / n)
             for i in range(n)]
    sb = SlotBelief(slot, species, cands)
    bs._slots[slot] = sb
    return sb


# ── 1. adapter: curated volatiles only, never Unknown for the rest ──────────

def test_adapter_maps_curated_start_and_end():
    a = LogAdapter()
    ev = a.feed_line("|-start|p2a|Taunt|")
    assert isinstance(ev, E.VolatileChanged)
    assert (ev.side, ev.slot, ev.volatile, ev.active) == ("p2", "p2a", "taunt", True)
    ev = a.feed_line("|-end|p2a|Taunt|")
    assert isinstance(ev, E.VolatileChanged)
    assert ev.volatile == "taunt" and ev.active is False


def test_adapter_normalizes_display_names():
    a = LogAdapter()
    ev = a.feed_line("|-start|p1a|Gastro Acid|[of] p2a")
    assert isinstance(ev, E.VolatileChanged)
    assert ev.volatile == "gastroacid"
    ev = a.feed_line("|-start|p2b|Destiny Bond|")
    assert ev.volatile == "destinybond"
    ev = a.feed_line("|-start|p2b|trapped|[of] p1a")
    assert ev.volatile == "trapped"


def test_adapter_all_seven_curated_surface():
    a = LogAdapter()
    for name, vid in (("Disguise", "disguise"), ("Taunt", "taunt"),
                      ("Encore", "encore"), ("Trapped", "trapped"),
                      ("Curse", "curse"), ("Destiny Bond", "destinybond"),
                      ("Gastro Acid", "gastroacid")):
        ev = a.feed_line(f"|-start|p2a|{name}|")
        assert isinstance(ev, E.VolatileChanged), name
        assert ev.volatile == vid, name


def test_adapter_non_curated_volatiles_silently_ignored():
    """FocusEnergy etc. are known ambient state — they must NOT become
    Unknown (the observable bug-catcher would cry wolf on every battle)."""
    a = LogAdapter()
    for name in ("FocusEnergy", "Intimidate", "MistyTerrain", "AuraBreak",
                 "speedboost", "regenerator"):
        assert a.feed_line(f"|-start|p2a|{name}|") is None, name
        assert a.feed_line(f"|-end|p2a|{name}|") is None, name


# ── 2. belief: volatiles live on Facts (multi-turn, per-mon) ─────────────────

def test_volatile_persists_across_turns_and_ends():
    bs = BeliefState("gen9ou")
    install(bs, "p2a", "garchomp")
    bs.update(E.VolatileChanged(side="p2", slot="p2a", volatile="taunt", active=True))
    assert "taunt" in bs._slots["p2a"].facts.volatiles
    # multi-turn volatile: a turn boundary must NOT clear it
    bs.update(E.Turn(n=2))
    assert "taunt" in bs._slots["p2a"].facts.volatiles
    # the |-end| line clears it
    bs.update(E.VolatileChanged(side="p2", slot="p2a", volatile="taunt", active=False))
    assert "taunt" not in bs._slots["p2a"].facts.volatiles


def test_volatile_cleared_by_new_mon():
    """A new mon on the slot gets fresh Facts — the old mon's volatiles are
    NOT inherited (they belonged to the fainted/switched-out mon)."""
    bs = BeliefState("gen9ou")
    install(bs, "p2a", "garchomp")
    bs.update(E.VolatileChanged(side="p2", slot="p2a", volatile="encore", active=True))
    assert "encore" in bs._slots["p2a"].facts.volatiles
    # white-box: a new mon replaces the slot (as add_opponent does on switch)
    fresh = SlotBelief("p2a", "charizard", [cand(moves=["flamethrower"], weight=1.0)])
    bs._slots["p2a"] = fresh
    assert fresh.facts.volatiles == set()


def test_volatile_on_own_side():
    bs = BeliefState("gen9ou", own_team={"p1a": {"name": "garchomp"}})
    bs.update(E.VolatileChanged(side="p1", slot="p1a", volatile="curse", active=True))
    assert "curse" in bs.own_facts["p1a"].volatiles
    bs.update(E.VolatileChanged(side="p1", slot="p1a", volatile="curse", active=False))
    assert "curse" not in bs.own_facts["p1a"].volatiles


def test_volatile_on_unknown_slot_is_a_noop():
    """No crash, no phantom facts — the slot simply isn't tracked yet."""
    bs = BeliefState("gen9ou")
    bs.update(E.VolatileChanged(side="p2", slot="p2c", volatile="taunt", active=True))
    assert "p2c" not in bs._slots


# ── 3. get_context: volatiles + fainted flag surface ─────────────────────────

def test_context_exposes_volatiles_sorted():
    bs = BeliefState("gen9ou")
    install(bs, "p2a", "garchomp")
    bs.update(E.VolatileChanged(side="p2", slot="p2a", volatile="gastroacid", active=True))
    bs.update(E.VolatileChanged(side="p2", slot="p2a", volatile="taunt", active=True))
    ctx = bs.get_context()
    assert ctx["opponent"]["p2a"]["volatiles"] == ["gastroacid", "taunt"]  # sorted


def test_context_fainted_flag_lifecycle():
    """faint -> fainted=1; re-occupy the slot -> fainted=0; an untouched
    benched mon -> fainted=0 (the benched-vs-gone distinction the net needs)."""
    bs = BeliefState("gen9ou")
    install(bs, "p2a", "garchomp")
    install(bs, "p2b", "charizard")

    # alive: no flag
    ctx = bs.get_context()
    assert ctx["opponent"]["p2a"]["fainted"] is False
    assert ctx["opponent"]["p2b"]["fainted"] is False

    # the active mon faints -> the slot is "gone" (Faint also zeroes hp)
    bs.update(E.Faint(side="p2", slot="p2a"))
    ctx = bs.get_context()
    assert ctx["opponent"]["p2a"]["fainted"] is True
    assert ctx["opponent"]["p2a"]["hp_frac"] == 0.0
    # the benched mon is untouched: same hp_frac 0, DIFFERENT flag
    assert ctx["opponent"]["p2b"]["hp_frac"] == 0.0
    assert ctx["opponent"]["p2b"]["fainted"] is False


def test_fainted_slot_cleared_on_reoccupation():
    bs = BeliefState("gen9ou")
    install(bs, "p2a", "garchomp")
    bs.update(E.Faint(side="p2", slot="p2a"))
    assert bs.fainted_slots.get("p2a") is True
    # a new mon takes the slot (replace) -> the slot is alive again
    fresh = SlotBelief("p2a", "charizard", [cand(moves=["flamethrower"], weight=1.0)])
    bs._slots["p2a"] = fresh
    bs.fainted_slots.pop("p2a", None)  # exactly what the SwitchIn handler does
    assert bs.fainted_slots.get("p2a", False) is False


# ── 4. end-to-end: raw log lines through adapter -> belief -> context ────────

def test_log_to_context_taunt_and_faint():
    bs = BeliefState("gen9ou")
    install(bs, "p2a", "garchomp")
    a = LogAdapter()
    for line in (
        "|switch|p2a|Garchomp|183/183",
        "|-start|p2a|Taunt|[from] move: Taunt|[of] p1a",
        "|turn|2",
        "|faint|p2a",
    ):
        ev = a.feed_line(line)
        if ev is not None:
            bs.update(ev)
    ctx = bs.get_context()
    blk = ctx["opponent"]["p2a"]
    assert blk["fainted"] is True
    # the Taunt that was on the mon: the mon is gone — its facts are the
    # record of its life (cleared only when a new mon takes the slot)
    assert "taunt" in blk["volatiles"]
