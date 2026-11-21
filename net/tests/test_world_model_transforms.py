"""M2b action-transform primitives (decision-independent) — isolated tests.

These pure, POV-aware context transforms (``apply_hp_delta`` / ``apply_faint`` /
``apply_switch``) are the STRUCTURAL core of M2b's ``step`` that every viable
option shares: a switch / faint / HP-delta is the same regardless of OC-6
(bookkeeping fidelity) or OC-9 (posterior semantics). So they are tested here in
isolation (no net, no calc, no RNG) and encode no design decision — the OC-gated
parts (the calc spread + the posterior refinement) are wired on top later.

Run from the project root:

    CUDA_VISIBLE_DEVICES=-1 python -m pytest net/tests/test_world_model_transforms.py -v
"""
from __future__ import annotations

import copy

from net.world_model import apply_faint, apply_hp_delta, apply_switch


def base_ctx():
    """A minimal canonical (p1-POV) belief context (own uses ``name``, opponent
    uses ``species`` — the belief's real schema)."""
    return {
        "turn": 1,
        "fainted": {"p1": 0, "p2": 0},
        "active": {"p1": "p1a", "p2": "p2a"},
        "field": {"weather": "", "terrain": "", "trickroom": False, "wonderroom": False,
                  "tailwind_side": "", "reflect": {}, "lightscreen": {}, "auroraveil": {},
                  "spikes": {}, "toxicspikes": {}, "stealthrock": {}, "stickyweb": {},
                  "safeguard": {}},
        "own": {
            "p1a": {"name": "garchomp", "hp_frac": 1.0, "maxhp": 183,
                    "fainted": False, "moves": ["earthquake"]},
            "p1b": {"name": "blastoise", "hp_frac": 1.0, "maxhp": 200,
                    "fainted": False, "moves": ["hydro"]},
        },
        "opponent": {
            "p2a": {"species": "gengar", "hp_frac": 0.8, "maxhp": 150,
                    "fainted": False, "moves": ["shadowball"]},
            "p2b": {"species": "charizard", "hp_frac": 1.0, "maxhp": 180,
                    "fainted": False, "moves": ["flamethrower"]},
        },
    }


# ── apply_hp_delta ───────────────────────────────────────────────────────────

def test_hp_delta_reduces_fraction_by_damage_over_maxhp():
    ctx = base_ctx()
    out = apply_hp_delta(ctx, "p2", "p2a", damage=30, maxhp=150)
    # 0.8 - 30/150 = 0.8 - 0.2 = 0.6
    assert out["opponent"]["p2a"]["hp_frac"] == 0.6


def test_hp_delta_clamps_at_zero():
    ctx = base_ctx()
    out = apply_hp_delta(ctx, "p2", "p2a", damage=500, maxhp=150)
    assert out["opponent"]["p2a"]["hp_frac"] == 0.0


def test_hp_delta_tolerates_unknown_maxhp():
    """An unknown maxhp (None) must leave the fraction unchanged, never crash."""
    ctx = base_ctx()
    ctx["opponent"]["p2a"]["maxhp"] = None
    before = ctx["opponent"]["p2a"]["hp_frac"]
    out = apply_hp_delta(ctx, "p2", "p2a", damage=75, maxhp=None)
    assert out["opponent"]["p2a"]["hp_frac"] == before


# ── apply_faint ──────────────────────────────────────────────────────────────

def test_faint_marks_gone_and_counts():
    ctx = base_ctx()
    out = apply_faint(ctx, "p1", "p1a")
    assert out["own"]["p1a"]["fainted"] is True
    assert out["own"]["p1a"]["hp_frac"] == 0.0
    assert out["fainted"]["p1"] == 1


# ── apply_switch ─────────────────────────────────────────────────────────────

def test_switch_updates_active_pointer_p1():
    ctx = base_ctx()
    out = apply_switch(ctx, "p1", "blastoise")
    assert out["active"]["p1"] == "p1b"   # blastoise (p1b) is now active


def test_switch_is_pov_aware_p2():
    """decider=p2 switches on ITS OWN side (the context's opponent side)."""
    ctx = base_ctx()
    out = apply_switch(ctx, "p2", "charizard")
    assert out["active"]["p2"] == "p2b"   # charizard (p2b) is now active
    assert out["active"]["p1"] == "p1a"   # p1's active is untouched


# ── purity: the input context is never mutated ───────────────────────────────

def test_transforms_do_not_mutate_input():
    """Each transform returns a NEW context; the original is byte-identical after
    all three (the MCTS tree must never corrupt a shared parent ctx)."""
    ctx = base_ctx()
    snap = copy.deepcopy(ctx)
    apply_hp_delta(ctx, "p2", "p2a", damage=30, maxhp=150)
    apply_faint(ctx, "p1", "p1a")
    apply_switch(ctx, "p1", "blastoise")
    assert ctx == snap
