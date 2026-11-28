"""Hermetic tests for the PURE parts of tools/bc_collect.py (the (c) BC
pipeline): the progress signature, the D6 ping-pong keep-mask, the
action-index identity, the recorder's ctx-failure path, and the npz
round-trip. No server, no torch, no GPU, fast.

The live collection driver (``collect_bc``) and the shared battle loop are
validated by a small live collection run (``python -m tools.bc_collect
--battles 2``) — the same discipline as ``test_learnability.py``.

The ping-pong mask tests PIN the exact implemented threshold semantics
(see the ``ping_pong_keep_mask`` docstring) so a later "simplification"
that changes the filter's behavior fails here loudly.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from net.registry import LAYOUT
from tools.bc_collect import (BCTrajectoryRecorder, action_index_of,
                              ping_pong_keep_mask, progress_signature,
                              save_bc_dataset)


# ── helpers ────────────────────────────────────────────────────────────────────
def _mon(hp_frac=0.9, maxhp=110, statuses=(), boosts=None, volatiles=(),
         stockpile=0, tera=None, last_hit=0.0, hit_this_turn=False,
         used_protect=False, last_move_failed=False):
    return {
        "hp_frac": hp_frac, "maxhp": maxhp, "statuses": list(statuses),
        "boosts": dict(boosts or {}), "volatiles": list(volatiles),
        "stockpile": stockpile, "tera": tera, "last_hit": last_hit,
        "hit_this_turn": hit_this_turn, "used_protect": used_protect,
        "last_move_failed": last_move_failed,
    }


def _ctx(own=None, opp=None, field=None, fainted=None, turn=5,
         active="garchomp"):
    """A minimal fake belief context (JSON-safe primitives, like
    ``build_ctx``'s output — only the keys the signature reads matter)."""
    return {
        "turn": turn,
        "active": active,
        "own": own if own is not None else {"0": _mon()},
        "opponent": opp if opp is not None else {"0": _mon(hp_frac=0.8,
                                                            maxhp=95)},
        "field": field or {"weather": "none", "terrain": None, "screens": {},
                           "hazards": {}, "tailwind": {}},
        "fainted": fainted if fainted is not None else {"0": 0, "1": 0},
    }


# ── progress_signature ─────────────────────────────────────────────────────────
def test_signature_is_deterministic():
    assert progress_signature(_ctx()) == progress_signature(_ctx())


def test_signature_ignores_turn_active_and_per_turn_flags():
    """The no-progress definition: a switch / pass / failed move that changes
    ONLY turn, the active slot, or per-turn flags is a NO-PROGRESS decision."""
    base = progress_signature(_ctx())
    c = _ctx(turn=99, active="dragonite")
    c["own"]["0"].update(hit_this_turn=True, used_protect=True,
                         last_move_failed=True, last_hit=0.5)
    c["opponent"]["0"].update(hit_this_turn=True, used_protect=True)
    assert progress_signature(c) == base


@pytest.mark.parametrize("mutate", [
    lambda c: c["own"]["0"].update(hp_frac=0.45),
    lambda c: c["own"]["0"].update(maxhp=115),
    lambda c: c["own"]["0"].update(statuses=["brn"]),
    lambda c: c["own"]["0"].update(boosts={"atk": 2}),
    lambda c: c["own"]["0"].update(volatiles=["taunt"]),
    lambda c: c["own"]["0"].update(stockpile=2),
    lambda c: c["own"]["0"].update(tera="fire"),
    lambda c: c["opponent"]["0"].update(hp_frac=0.1),
    lambda c: c["field"].update(weather="raindance"),
    lambda c: c["field"].update(terrain="electric"),
    lambda c: c["field"].update(screens={"reflect_p1": 5}),
    lambda c: c["field"].update(hazards={"stealthrock_p1": 1}),
    lambda c: c["field"].update(tailwind={"p1": 3}),
    lambda c: c.update(fainted={"0": 1, "1": 0}),
])
def test_signature_detects_real_progress(mutate):
    base = progress_signature(_ctx())
    c = _ctx()
    mutate(c)
    assert progress_signature(c) != base


def test_signature_is_order_independent_over_slots():
    a = _ctx()
    b = _ctx()
    b["own"] = {k: a["own"][k] for k in reversed(list(a["own"]))}
    b["opponent"] = {k: a["opponent"][k]
                     for k in reversed(list(a["opponent"]))}
    assert progress_signature(a) == progress_signature(b)


# ── ping_pong_keep_mask (pins the implemented threshold semantics) ───────────
def test_mask_empty_and_single():
    assert ping_pong_keep_mask([]) == []
    assert ping_pong_keep_mask(["A"]) == [True]


def test_mask_all_unique_all_kept():
    assert ping_pong_keep_mask(["A", "B", "C", "A"]) == [True] * 4


def test_mask_lone_no_progress_kept():
    assert ping_pong_keep_mask(["A", "B", "B"]) == [True, True, True]


def test_mask_run_keeps_first_drops_rest():
    # min_run=3 (explicit — the default is 8 per the USER ruling 2026-11-24):
    # 2 consecutive no-progress: keep the first, drop the second
    assert ping_pong_keep_mask(["A", "B", "B", "B"], min_run=3) == \
        [True, True, True, False]
    # 3 in a row: keep the first, drop the second AND third
    assert ping_pong_keep_mask(["A", "B", "B", "B", "B"], min_run=3) == \
        [True, True, True, False, False]


def test_mask_progress_resets_run():
    assert ping_pong_keep_mask(["A", "B", "B", "C", "B", "B", "B"],
                               min_run=3) == \
        [True, True, True, True, True, True, False]


def test_mask_min_run_threshold():
    # min_run=2: the very first repeat is dropped
    assert ping_pong_keep_mask(["A", "B", "B"], min_run=2) == \
        [True, True, False]
    # min_run=4: dropping starts at the 3rd consecutive no-progress
    assert ping_pong_keep_mask(["A", "B", "B", "B", "B", "B"], min_run=4) == \
        [True, True, True, True, False, False]
    # min_run<2: degenerate guard — nothing is dropped
    assert ping_pong_keep_mask(["A", "B", "B"], min_run=1) == \
        [True, True, True]
    # default min_run=8 (USER ruling 2026-11-24): a ping-pong loop keeps at
    # most 7 consecutive no-progress decisions; the 8th is dropped
    assert ping_pong_keep_mask(["A"] + ["B"] * 7) == [True] * 8
    assert ping_pong_keep_mask(["A"] + ["B"] * 8) == [True] * 8 + [False]


# ── action_index_of ────────────────────────────────────────────────────────────
def test_action_index_move_and_switch():
    acts = [{"kind": "move", "move": "earthquake"},
            {"kind": "switch", "species": "garchomp"}]
    assert action_index_of(acts, {"kind": "move", "move": "earthquake"}) == 0
    assert action_index_of(acts, {"kind": "switch", "species": "garchomp"}) == 1


def test_action_index_tera_is_a_distinct_action():
    acts = [{"kind": "move", "move": "draco_meteor"},
            {"kind": "tera", "move": "draco_meteor"}]
    assert action_index_of(acts, {"kind": "tera", "move": "draco_meteor"}) == 1
    assert action_index_of(acts, {"kind": "move", "move": "draco_meteor"}) == 0


def test_action_index_absent_or_none_is_minus_one():
    acts = [{"kind": "move", "move": "earthquake"}]
    assert action_index_of(acts, {"kind": "move", "move": "surf"}) == -1
    assert action_index_of(acts, None) == -1


def test_action_index_matches_by_identity_not_object():
    # the teacher's chosen action is a FRESH dict — identity must be
    # structural (kind + move/species), not object equality
    acts = [{"kind": "switch", "species": "garchomp"}]
    assert action_index_of(acts, dict(kind="switch", species="garchomp")) == 0


# ── recorder: the ctx-failure path must not fabricate a ping-pong run ─────────
def test_recorder_ctx_failure_keeps_all_decisions():
    rec = BCTrajectoryRecorder(fmt="gen9ou")
    acts = [{"kind": "move", "move": "earthquake"}]
    for _ in range(5):
        # battle=None -> build_ctx raises -> a UNIQUE CTX_UNKNOWN_<i> sig
        rec({}, acts, acts[0], battle=None)
    res = rec.finish()
    assert res["n_decisions"] == 5
    assert res["n_kept"] == 5        # unknown progress is NOT a no-progress repeat
    assert res["n_dropped"] == 0
    assert [r["action_index"] for r in res["kept"]] == [0] * 5
    # the failure sigs are all distinct (no fabricated run)
    sigs = [r["sig"] for r in res["kept"]]
    assert len(set(sigs)) == 5


# ── save_bc_dataset: npz round-trip ────────────────────────────────────────────
def _fake_obs(seed: int) -> dict:
    obs = {}
    for fld in LAYOUT:
        arr = np.zeros(fld.shape, dtype=fld.dtype)
        arr[...] = seed * 0.001 if fld.dtype == "float32" else seed
        obs[fld.name] = arr
    return obs


def _rec(action_index: int, seed: int) -> dict:
    return {"obs": _fake_obs(seed), "action_index": action_index,
            "action": {"kind": "move", "move": "x"}, "sig": f"s{seed}",
            "n_legal": 8}


def test_save_bc_dataset_roundtrip(tmp_path):
    per_battle = [
        {"battle_id": 0,
         "result": {"n_decisions": 4, "n_kept": 3, "n_dropped": 1,
                    "kept": [_rec(0, 1), _rec(1, 2), _rec(2, 3)]}},
        {"battle_id": 1,
         "result": {"n_decisions": 2, "n_kept": 2, "n_dropped": 0,
                    "kept": [_rec(3, 4), _rec(4, 5)]}},
    ]
    out = str(tmp_path / "bc")
    meta = {"fmt": "gen9ou", "battles_requested": 2, "seed": 7}
    summary = save_bc_dataset(out, per_battle, meta)

    assert summary == {"n_kept": 5, "n_decisions": 6, "n_dropped": 1}

    data = np.load(os.path.join(out, "data.npz"))
    assert set(data.files) == {f.name for f in LAYOUT} | \
        {"action_index", "battle_id"}
    for fld in LAYOUT:
        assert data[fld.name].shape == (5, *fld.shape), fld.name
        fill = 0.001 if fld.dtype == "float32" else 1
        expected = np.full(fld.shape, fill, dtype=fld.dtype)
        assert np.array_equal(data[fld.name][0], expected), fld.name
    assert data["action_index"].dtype == np.int64
    assert list(data["action_index"]) == [0, 1, 2, 3, 4]
    assert data["battle_id"].dtype == np.int32
    assert list(data["battle_id"]) == [0, 0, 0, 1, 1]

    with open(os.path.join(out, "meta.json"), encoding="utf8") as fh:
        meta_out = json.load(fh)
    assert meta_out["n_kept"] == 5
    assert meta_out["n_decisions"] == 6
    assert meta_out["n_dropped"] == 1
    assert meta_out["fmt"] == "gen9ou"
    assert meta_out["seed"] == 7


def test_save_bc_dataset_empty_raises(tmp_path):
    per_battle = [{"battle_id": 0,
                   "result": {"n_decisions": 3, "n_kept": 0, "n_dropped": 3,
                              "kept": []}}]
    with pytest.raises(ValueError, match="no decisions survived"):
        save_bc_dataset(str(tmp_path / "bc"), per_battle, {"fmt": "gen9ou"})
