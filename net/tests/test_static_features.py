"""Step 3.1/3.2 (review 2026-11-22) — static mon/move feature tests.

These pin the STATIC features (types / base stats / move power, type, category,
priority, accuracy, effect flags) that the net now consumes as ADDITIONAL float
blocks next to the learned id embeddings. The gates:

  * **deterministic** — same ctx + actions + gen -> bitwise-identical arrays;
  * **degrades, never raises** — an unknown species/move is a ZERO row (the net
    reads "unknown", exactly like the existing id channels), even though
    ``legality.lookup`` *raises* on unknown names;
  * **known values** — hand-checked against the legality DB (Garchomp dual
    Dragon/Ground 0.5/0.5; Earthquake power 0.5; Stealth Rock is_hazard=1);
  * **bench-slot permutation** — swapping two bench own slots permutes ONLY the
    matching own_types / own_stats rows (the S4.2 set-encoder is then
    permutation-invariant over the static rows too);
  * **encoder placement** — ``encode_context`` places exactly these five fields
    and zero-fills them when ``static_feats`` is absent (the pure/DB-free
    contract: the encoder never touches the DB itself).

All tests are OFFLINE and torch-free (numpy only).
"""
from __future__ import annotations

import copy

import numpy as np
import pytest

from net import static_features as sf
from net.encoder import encode_context
from net.registry import (
    K_MOVE_STATIC,
    LAYOUT,
    MAX_ACTIONS,
    MAX_SLOTS,
    N_STATS,
    N_TYPES,
    TYPES,
)
from net.tests.test_s41 import make_ctx, make_vocabs
from net.vocab import Vocabs  # noqa: F401  (kept for clarity of intent)

_TYPE_IDX = {t: i for i, t in enumerate(TYPES)}


# ── determinism ──────────────────────────────────────────────────────────
def test_build_deterministic_two_runs_identical():
    ctx = make_ctx()
    actions = [
        {"kind": "move", "move": "earthquake"},
        {"kind": "move", "move": "stealthrock"},
        {"kind": "switch", "species": "volcarona"},
    ]
    a = sf.build_static_features(ctx, actions, 9)
    b = sf.build_static_features(copy.deepcopy(ctx), actions, 9)
    for name in sf.STATIC_FEATURE_NAMES:
        assert np.array_equal(a[name], b[name]), f"non-deterministic: {name}"


# ── degrades, never raises (unknown species/move -> zero row) ────────────
def test_unknown_move_is_zero_row():
    v = sf.move_static_vector("zzznotamove", 9)
    assert v.shape == (K_MOVE_STATIC,)
    assert not v.any()


def test_unknown_species_is_zero_rows():
    assert not sf.types_onehot("zzznotamon", 9).any()
    assert not sf.stats_vector("zzznotamon", 9).any()


def test_empty_move_name_is_zero_row():
    assert not sf.move_static_vector("", 9).any()
    assert not sf.move_static_vector(None, 9).any()


# ── known values (hand-checked against the legality DB) ──────────────────
def test_earthquake_known_values():
    v = sf.move_static_vector("earthquake", 9)
    assert v[0] == pytest.approx(100 / 200.0)            # power 100 -> 0.5
    assert v[1] == pytest.approx(_TYPE_IDX["Ground"] / N_TYPES)
    assert v[2] == pytest.approx(0.0)                    # physical
    assert v[3] == pytest.approx((0 + 6) / 12.0)         # priority 0 -> 0.5
    assert v[4] == pytest.approx(1.0)                    # accuracy 100
    # no hazard / screen / boost / side effects
    assert v[5] == 0.0 and v[6] == 0.0 and v[7] == 0.0
    assert v[8] == 0.0 and v[9] == 0.0 and v[10] == 0.0


def test_stealthrock_is_hazard_and_status():
    v = sf.move_static_vector("stealthrock", 9)
    assert v[0] == 0.0                                    # no power (status move)
    assert v[2] == pytest.approx(1.0)                     # status category
    assert v[5] == 1.0                                    # is_hazard
    assert v[9] == 1.0                                    # side_condition (stealthrock)
    assert v[6] == 0.0 and v[7] == 0.0 and v[10] == 0.0   # not a screen/boost/tw


def test_garchomp_dual_type_05_05():
    t = sf.types_onehot("garchomp", 9)
    assert t.shape == (N_TYPES,)
    assert t[_TYPE_IDX["Dragon"]] == pytest.approx(0.5)
    assert t[_TYPE_IDX["Ground"]] == pytest.approx(0.5)
    assert t.sum() == pytest.approx(1.0)                  # exactly the two types


def test_landorus_therian_dual_type():
    t = sf.types_onehot("landorustherian", 9)
    assert t[_TYPE_IDX["Flying"]] == pytest.approx(0.5)
    assert t[_TYPE_IDX["Ground"]] == pytest.approx(0.5)
    assert t.sum() == pytest.approx(1.0)


def test_stats_normalised_by_255():
    s = sf.stats_vector("garchomp", 9)
    assert s.shape == (N_STATS,)
    assert s[0] == pytest.approx(108 / 255.0)             # hp
    assert s[1] == pytest.approx(130 / 255.0)             # atk
    # all in [0, 1]
    assert (s >= 0).all() and (s <= 1).all()


# ── build_static_features: correct slots + actions are filled, rest zero ─
def test_build_fills_correct_slots_and_actions():
    ctx = make_ctx()
    actions = [
        {"kind": "move", "move": "earthquake"},
        {"kind": "move", "move": "stealthrock"},
        {"kind": "switch", "species": "volcarona"},   # no move -> action row 2 stays 0
        {"kind": "tera", "species": "volcarona"},      # no move -> action row 3 stays 0
    ]
    out = sf.build_static_features(ctx, actions, 9)

    # action rows: only kind=="move" rows carry a move vector
    assert np.array_equal(out["action_move_static"][0], sf.move_static_vector("earthquake", 9))
    assert np.array_equal(out["action_move_static"][1], sf.move_static_vector("stealthrock", 9))
    assert not out["action_move_static"][2].any()
    assert not out["action_move_static"][3].any()

    # own types/stats filled for the three present own slots (p1a,b,c)
    assert out["own_types"][0].any()
    assert out["own_stats"][0].any()
    assert not out["own_types"][3].any()                # slot d empty
    # opponent slot 0 = garchomp (known) -> filled ...
    assert out["opp_types"][0].any()
    assert out["opp_stats"][0].any()
    # opponent slot 1 = toxicroc (NOT in the DB) -> degrades to a zero row,
    # exactly like an unknown id (the graceful-degradation contract).
    assert not out["opp_types"][1].any()
    assert not out["opp_stats"][1].any()


def test_build_respects_max_actions_cap():
    ctx = make_ctx()
    actions = [{"kind": "move", "move": "earthquake"}] * (MAX_ACTIONS + 5)
    out = sf.build_static_features(ctx, actions, 9)
    assert out["action_move_static"].shape == (MAX_ACTIONS, K_MOVE_STATIC)
    assert out["action_move_static"][MAX_ACTIONS - 1].any()


# ── bench-slot permutation permutes ONLY the matching rows ───────────────
def test_bench_swap_permutes_matching_static_rows():
    ctx = make_ctx()
    out1 = sf.build_static_features(ctx, [], 9)

    ctx2 = copy.deepcopy(ctx)
    # swap the two BENCH own slots (p1b <-> p1c); active (p1a) stays put
    ctx2["own"]["p1b"], ctx2["own"]["p1c"] = ctx["own"]["p1c"], ctx["own"]["p1b"]
    out2 = sf.build_static_features(ctx2, [], 9)

    # active slot row is identical (unchanged)
    assert np.array_equal(out1["own_types"][0], out2["own_types"][0])
    # the two bench rows are swapped, element-for-element
    assert np.array_equal(out1["own_types"][1], out2["own_types"][2])
    assert np.array_equal(out1["own_types"][2], out2["own_types"][1])
    assert np.array_equal(out1["own_stats"][1], out2["own_stats"][2])
    # everything else (opponent) untouched
    for name in ("opp_types", "opp_stats", "action_move_static"):
        assert np.array_equal(out1[name], out2[name]), name


# ── encoder placement (the pure/DB-free contract) ────────────────────────
def test_encoder_zero_fills_static_fields_when_absent():
    ctx = make_ctx()
    out = encode_context(ctx, make_vocabs(ctx))           # no static_feats
    for name in sf.STATIC_FEATURE_NAMES:
        assert name in out
        assert out[name].shape == dict((f.name, f.shape) for f in LAYOUT)[name]
        assert not out[name].any()                         # all zero


def test_encoder_places_provided_static_arrays():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    feats = sf.build_static_features(ctx, [{"kind": "move", "move": "earthquake"}], 9)
    out = encode_context(ctx, v, [{"kind": "move", "move": "earthquake"}],
                         static_feats=feats)
    # the encoder placed the exact arrays it was handed (no mutation, no loss)
    for name in sf.STATIC_FEATURE_NAMES:
        assert np.array_equal(out[name], feats[name]), name


def test_encoder_output_still_matches_registry_with_static():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    feats = sf.build_static_features(ctx, [], 9)
    out = encode_context(ctx, v, [], static_feats=feats)
    assert set(out.keys()) == {f.name for f in LAYOUT}
    for fld in LAYOUT:
        assert out[fld.name].shape == fld.shape, f"{fld.name} shape"
        assert out[fld.name].dtype == np.dtype(fld.dtype), f"{fld.name} dtype"
