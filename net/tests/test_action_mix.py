"""S3.4 action-mix classifier tests (net/action_mix.py).

Hermetic: the classification logic is tested against FAKE move rows (the
shape of the DB rows: category / side_condition / boosts), so no DB, no
torch, no Node. One integration test uses the REAL DB to pin the field
names.
"""
from __future__ import annotations

import pytest

from net.action_mix import (action_mix, build_moves_lookup,
                            classify_action, classify_move)


def _move(**kw):
    base = {"category": None, "side_condition": None, "boosts": None}
    base.update(kw)
    return base


# ── classify_move ─────────────────────────────────────────────────────────────

def test_hazard_by_side_condition():
    for cond in ("spikes", "stealthrock", "toxicspikes", "stickyweb"):
        assert classify_move(_move(category="Status", side_condition=cond)) == "hazard"


def test_screens_are_not_hazards():
    # reflect/lightscreen/tailwind are side_conditions in the DB, but NOT
    # field hazards — they fall through to status.
    for cond in ("reflect", "lightscreen", "auroraveil", "tailwind", "safeguard"):
        assert classify_move(_move(category="Status", side_condition=cond)) == "status"


def test_setup_requires_total_plus_two():
    assert classify_move(_move(category="Status", boosts='{"atk":2}')) == "setup"       # Swords Dance
    assert classify_move(_move(category="Status", boosts='{"spa":1,"spd":1}')) == "setup"  # Calm Mind
    assert classify_move(_move(category="Status", boosts='{"atk":1,"def":1}')) == "setup"  # Bulk Up
    # Aromatis Mist: only +1 total -> status, NOT setup
    assert classify_move(_move(category="Status", boosts='{"spd":1}')) == "status"
    # negative boosts never count
    assert classify_move(_move(category="Status", boosts='{"spa":-2}')) == "status"


def test_attack_by_category():
    assert classify_move(_move(category="Physical", power=100)) == "attack"
    assert classify_move(_move(category="Special", power=90)) == "attack"


def test_status_moves():
    assert classify_move(_move(category="Status")) == "status"


def test_unknown_category_is_other():
    assert classify_move(_move()) == "other"


def test_bad_boosts_json_is_not_setup():
    assert classify_move(_move(category="Status", boosts="not-json")) == "status"


# ── classify_action ───────────────────────────────────────────────────────────

MOVES = {
    "earthquake": _move(category="Physical", power=100),
    "toxic": _move(category="Status"),
    "stealthrock": _move(category="Status", side_condition="stealthrock"),
    "swordsdance": _move(category="Status", boosts='{"atk":2}'),
    "reflect": _move(category="Status", side_condition="reflect"),
}


def test_action_switch_and_move():
    assert classify_action({"kind": "switch", "species": "garchomp"}, MOVES) == "switch"
    assert classify_action({"kind": "move", "move": "earthquake"}, MOVES) == "attack"
    assert classify_action({"kind": "move", "move": "stealthrock"}, MOVES) == "hazard"
    assert classify_action({"kind": "move", "move": "swordsdance"}, MOVES) == "setup"


def test_action_tera_unknown_and_none_are_other():
    assert classify_action({"kind": "tera", "move": "earthquake"}, MOVES) == "other"
    assert classify_action(None, MOVES) == "other"
    assert classify_action({"kind": "move", "move": "doesnotexist"}, MOVES) == "other"


# ── action_mix ────────────────────────────────────────────────────────────────

def test_action_mix_shares_sum_to_one():
    actions = [
        {"kind": "move", "move": "earthquake"},
        {"kind": "move", "move": "earthquake"},
        {"kind": "switch", "species": "a"},
        {"kind": "move", "move": "stealthrock"},
        {"kind": "move", "move": "swordsdance"},
        {"kind": "move", "move": "toxic"},
    ]
    mix = action_mix(actions, MOVES)
    assert set(mix) == {"attack", "switch", "hazard", "setup", "status", "other"}
    assert abs(sum(mix.values()) - 1.0) < 1e-12
    assert mix["attack"] == pytest.approx(2 / 6)
    assert mix["switch"] == pytest.approx(1 / 6)
    assert mix["hazard"] == pytest.approx(1 / 6)
    assert mix["setup"] == pytest.approx(1 / 6)
    assert mix["status"] == pytest.approx(1 / 6)
    assert mix["other"] == 0.0


def test_action_mix_empty_is_all_zero():
    mix = action_mix([], MOVES)
    assert all(v == 0.0 for v in mix.values())


# ── the real DB lookup (integration seam) ────────────────────────────────────

def test_build_moves_lookup_shape():
    moves = build_moves_lookup()
    assert len(moves) > 500
    # the three fields the classifier reads exist on every row
    for m in moves.values():
        assert set(m) == {"category", "side_condition", "boosts"}
    # spot-checks against the real data
    assert classify_move(moves["stealthrock"]) == "hazard"
    assert classify_move(moves["swordsdance"]) == "setup"
    assert classify_move(moves["toxic"]) == "status"
    assert classify_move(moves["reflect"]) == "status"
