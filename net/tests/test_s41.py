"""S4.1 — observation encoder gate tests (PLAN.md §3c S4.1).

The S4.1 gate (user spec):
  * **fixed shapes** (unit) — the encoder emits EXACTLY the registry's
    fields/shapes/dtypes, nothing more, nothing less;
  * **no leakage** (rule 4) — opponent UNREVEALED info reaches the net ONLY
    through the posterior / unknown channels; no raw opponent ability/item id;
  * **deterministic** — same context -> bitwise-identical tensor (2x);
  * **bench swap = slot permutation only** — swapping two bench slots permutes
    exactly the own-slot rows + own id channels, leaving everything else fixed
    (the S4.2 set-encoder is then permutation-invariant);
  * **dim-registry checksum** — the layout pin is stable and self-consistent;
  * baseline green (run separately: ``legality/damage/belief`` 331 suite).

All tests are OFFLINE and DB-free: the encoder is pure and the vocabs are built
from the ids present in the test context (the production path builds the SAME
Vocabs from the legality DB in S4.2 — same shapes either way).
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bot" / "data" / "replays" / "gen9ou.jsonl"
HAS_FIXTURE = FIXTURE.exists()
needs_fixture = pytest.mark.skipif(
    not HAS_FIXTURE, reason="gen9ou.jsonl human-ladder fixture missing")

from net import (
    DTYPES,
    LAYOUT,
    SHAPES,
    Vocabs,
    Vocab,
    checksum,
    encode_context,
    encode_sequence,
    total_elements,
)
from net.registry import (
    GLOBAL_INDEX,
    MAX_ACTIONS,
    MAX_OWN_MOVES,
    MAX_POST,
    MAX_SLOTS,
    MAX_TOP_MOVES,
    SLOT_INDEX,
)

# ── the layout pin ────────────────────────────────────────────────────
# Pinned the FIRST time the suite ran; if ANY field is added/removed/renamed/
# reshaped/reordered the checksum changes and this FAILS (the drift-catcher).
# Re-pin deliberately (and reconcile the encoder + S4.2 net) by updating this.
PINNED_CHECKSUM = "4765c031263b7be3c5572a27116ffef6dbf8f109a8405c739ed0ffb387f121a8"


def _mon(
    hp=0.5,
    statuses=(),
    boosts=None,
    moves=(),
    tera="",
    last_hit=0,
    hit_this_turn=False,
    last_move_failed=False,
    stockpile=0,
    used_protect=False,
    name="",
    species="",
    ability="",
    item="",
    posterior=None,
    top_moves=(),
    confidence=0.0,
) -> dict:
    blk = {
        "hp_frac": hp,
        "statuses": list(statuses),
        "boosts": dict(boosts or {}),
        "moves": list(moves),
        "tera": tera,
        "last_hit": last_hit,
        "hit_this_turn": hit_this_turn,
        "last_move_failed": last_move_failed,
        "stockpile": stockpile,
        "used_protect": used_protect,
    }
    if name:
        blk["name"] = name
    if species:
        blk["species"] = species
    if ability:
        blk["ability"] = ability
    if item:
        blk["item"] = item
    if posterior is not None:
        blk["posterior"] = posterior
    if top_moves:
        blk["top_moves"] = list(top_moves)
    blk["confidence"] = confidence
    return blk


def make_ctx() -> dict:
    """A realistic gen9-ou turn-start context (matches get_context())."""
    return {
        "turn": 3,
        "fainted": {"p1": 0, "p2": 1},
        "active": {"p1": "p1a", "p2": "p2a"},
        "field": {
            "weather": "raindance",
            "terrain": "",
            "trickroom": False,
            "wonderroom": False,
            "tailwind_side": "",
            "reflect": {"p1": 4},
            "lightscreen": {},
            "auroraveil": {},
            "spikes": {"p1": 2},
            "toxicspikes": {},
            "stealthrock": {"p2": True},
            "stickyweb": {},
            "safeguard": {},
        },
        "own": {
            "p1a": _mon(hp=0.75, statuses=("brn",), boosts={"atk": 2, "spe": -1},
                        moves=("earthquake", "stealthrock", "icebeam", "roar"),
                        tera="ground", last_hit=40, hit_this_turn=True,
                        name="landorustherian", ability="sandforce", item="leftovers"),
            "p1b": _mon(hp=1.0, moves=("flamethrower", "willowisp", "sludgewave", "roost"),
                        name="volcarona", ability="flamebody", item="lifeorb"),
            "p1c": _mon(hp=0.9, moves=("icebeam", "shadowball", "sludgewave", "roost"),
                        name="dragonite", ability="multiscale", item="focusband"),
        },
        "opponent": {
            "p2a": _mon(hp=0.5, moves=("outrage",),
                        species="garchomp",
                        posterior=[
                            {"ability": "roughskin", "moves": ["outrage", "earthquake"], "weight": 0.6},
                            {"ability": "sandveil", "moves": ["outrage", "dragonclaw"], "weight": 0.4},
                        ],
                        top_moves=("outrage", "earthquake", "dragonclaw", "roost"),
                        confidence=0.7),
            "p2b": _mon(hp=1.0, species="toxicroc",
                        posterior=[{"ability": "poisontouch", "moves": ["stealthrock"], "weight": 1.0}],
                        top_moves=("stealthrock", "stoneedge"), confidence=0.9),
        },
    }


def make_vocabs(ctx: dict) -> Vocabs:
    """Build the shared Vocabs from the ids present in the context (DB-free)."""
    species, abilities, items, moves = set(), set(), set(), set()
    for blk in ctx.get("own", {}).values():
        for k in ("name", "species"):
            if blk.get(k):
                species.add(blk[k])
        if blk.get("ability"):
            abilities.add(blk["ability"])
        if blk.get("item"):
            items.add(blk["item"])
        moves.update(blk.get("moves", []))
    for blk in ctx.get("opponent", {}).values():
        if blk.get("species"):
            species.add(blk["species"])
        moves.update(blk.get("moves", []))
        for c in blk.get("posterior", []):
            if c.get("ability"):
                abilities.add(c["ability"])
            moves.update(c.get("moves", []))
        moves.update(blk.get("top_moves", []))
    weather = {ctx["field"]["weather"]} if ctx["field"].get("weather") else set()
    terrain = {ctx["field"]["terrain"]} if ctx["field"].get("terrain") else set()
    return Vocabs(
        species=Vocab(species), ability=Vocab(abilities), item=Vocab(items),
        move=Vocab(moves), weather=Vocab(weather), terrain=Vocab(terrain),
    )


# ── fixed shapes (the width is DERIVED from the registry) ─────────────
def test_output_matches_registry_exactly():
    ctx = make_ctx()
    out = encode_context(ctx, make_vocabs(ctx))
    # keys: exactly the registry's fields, no more no less
    assert set(out.keys()) == {f.name for f in LAYOUT}
    # shapes + dtypes: each output matches the registry entry
    for fld in LAYOUT:
        assert out[fld.name].shape == fld.shape, f"{fld.name} shape {out[fld.name].shape} != {fld.shape}"
        assert out[fld.name].dtype == np.dtype(fld.dtype), f"{fld.name} dtype"


def test_no_ad_hoc_fields():
    ctx = make_ctx()
    out = encode_context(ctx, make_vocabs(ctx))
    # the D61-cut / not-yet-built channels must NOT sneak in
    for bad in ("opp_ability", "opp_item", "moveset_prior", "OBS_DIM"):
        assert bad not in out


# ── determinism (same context -> bitwise-identical tensor, 2x) ────────
def test_deterministic_two_runs_identical():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    a = encode_context(ctx, v)
    b = encode_context(ctx, make_vocabs(ctx))  # fresh vocabs, same ids
    for name in a:
        assert np.array_equal(a[name], b[name]), f"non-deterministic field: {name}"


def test_deterministic_same_vocab_object():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    a = encode_context(ctx, v)
    b = encode_context(ctx, v)
    for name in a:
        assert np.shares_memory(a[name], b[name]) is False  # fresh arrays each call
        assert np.array_equal(a[name], b[name]), name


# ── bench swap = slot permutation only (permutation invariance) ───────
def test_bench_swap_is_pure_row_permutation():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    enc1 = encode_context(ctx, v)

    ctx2 = copy.deepcopy(ctx)
    # swap the two BENCH own slots (p1b <-> p1c); the active slot (p1a) stays
    ctx2["own"]["p1b"], ctx2["own"]["p1c"] = ctx["own"]["p1c"], ctx["own"]["p1b"]
    enc2 = encode_context(ctx2, v)

    # global + opponent side: completely unchanged
    for name in ("global", "global_weather", "global_terrain",
                 "opp_slot", "opp_species", "opp_moves",
                 "opp_ability_post_ids", "opp_ability_post_w", "opp_top_moves",
                 "action_kind", "action_move", "action_species", "action_mask"):
        assert np.array_equal(enc1[name], enc2[name]), f"{name} changed by a bench swap"

    # own side: active row (index 0) unchanged; bench rows 1<->2 permuted
    for name in ("own_species", "own_ability", "own_item"):
        assert np.array_equal(enc1[name][0], enc2[name][0]), f"{name} active row moved"
        assert enc1[name][1] == enc2[name][2] and enc1[name][2] == enc2[name][1], name
        assert np.array_equal(enc1[name][3:], enc2[name][3:])
    for name in ("own_slot", "own_moves"):
        assert np.array_equal(enc1[name][0], enc2[name][0]), f"{name} active row moved"
        assert np.array_equal(enc1[name][1], enc2[name][2]) and \
               np.array_equal(enc1[name][2], enc2[name][1]), name
        assert np.array_equal(enc1[name][3:], enc2[name][3:])  # empty rows stay empty


# ── no leakage (rule 4) ───────────────────────────────────────────────
def test_opponent_unrevealed_only_via_posterior():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    out = encode_context(ctx, v)
    # there is NO raw opponent ability / item channel at all (structural)
    assert "opp_ability" not in out and "opp_item" not in out
    # the opponent's abilities reach the net ONLY as posterior ids
    post = set(int(x) for x in out["opp_ability_post_ids"] if x != 0)
    assert post, "expected the active opponent's posterior abilities to be present"
    for idx in post:
        assert v.ability.id_of(idx) in ("roughskin", "sandveil")
    # the opponent's raw revealed species IS allowed (it is revealed, not hidden)
    assert out["opp_species"][0] == v.species.index("garchomp")


def test_opponent_block_ability_key_never_leaked():
    """Even if an opponent block carried an 'ability' key (get_context does not
    for opponents), the encoder must NOT emit it as a raw channel — it can only
    surface through the posterior."""
    ctx = make_ctx()
    ctx["opponent"]["p2a"]["ability"] = "roughskin"  # hypothetical hidden field
    v = make_vocabs(ctx)
    out = encode_context(ctx, v)
    assert "opp_ability" not in out
    # 'roughskin' appears only as a posterior id, never as a standalone channel
    assert v.ability.id_of(out["opp_ability_post_ids"][0]) in ("roughskin", "sandveil")


# ── dim-registry checksum (the drift-catcher pin) ─────────────────────
def test_checksum_stable_and_wellformed():
    c1, c2 = checksum(), checksum()
    assert c1 == c2
    assert len(c1) == 64 and all(ch in "0123456789abcdef" for ch in c1)


def test_checksum_pin():
    assert checksum() == PINNED_CHECKSUM, (
        "observation layout changed — reconcile the encoder + S4.2 net, then "
        f"re-pin. current={checksum()}")


def test_total_elements_self_consistent():
    manual = 0
    for fld in LAYOUT:
        prod = 1
        for s in fld.shape:
            prod *= s
        manual += prod
    assert total_elements() == manual


# ── semantic spot-checks (the values are sane, not just shaped) ───────
def test_boost_sign_preserved():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    out = encode_context(ctx, v)
    atk_col = SLOT_INDEX["boost_atk"]
    spe_col = SLOT_INDEX["boost_spe"]
    # p1a has atk +2 -> +1/3, spe -1 -> -1/6  (signs MUST survive)
    assert out["own_slot"][0, atk_col] == pytest.approx(2 / 6.0)
    assert out["own_slot"][0, spe_col] == pytest.approx(-1 / 6.0)


def test_global_field_flags():
    ctx = make_ctx()
    out = encode_context(ctx, make_vocabs(ctx))
    g = out["global"]
    assert g[GLOBAL_INDEX["turn"]] == pytest.approx(3 / 20.0)
    assert g[GLOBAL_INDEX["reflect_p1"]] == 1.0
    assert g[GLOBAL_INDEX["spikes_p1"]] == pytest.approx(2 / 3.0)
    assert g[GLOBAL_INDEX["stealthrock_p2"]] == 1.0
    assert g[GLOBAL_INDEX["fainted_p2"]] == pytest.approx(1 / 6.0)
    assert out["global_weather"][0] != 0  # raindance is in the vocab


def test_status_one_hot_and_flags():
    ctx = make_ctx()
    out = encode_context(ctx, make_vocabs(ctx))
    row = out["own_slot"][0]
    assert row[SLOT_INDEX["st_brn"]] == 1.0
    assert row[SLOT_INDEX["st_par"]] == 0.0
    assert row[SLOT_INDEX["hit_this_turn"]] == 1.0
    assert row[SLOT_INDEX["tera_revealed"]] == 1.0
    assert row[SLOT_INDEX["hp_frac"]] == pytest.approx(0.75)
    assert row[SLOT_INDEX["active"]] == 1.0  # p1a is the active own slot
    assert out["own_slot"][1, SLOT_INDEX["active"]] == 0.0  # p1b is bench


# ── D60 action-list channel ───────────────────────────────────────────
def test_action_list_encoded_and_masked():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    actions = [
        {"kind": "move", "move": "earthquake"},
        {"kind": "move", "move": "icebeam"},
        {"kind": "switch", "species": "volcarona"},
    ]
    out = encode_context(ctx, v, actions=actions)
    assert out["action_mask"][:3].tolist() == [1.0, 1.0, 1.0]
    assert out["action_mask"][3:].tolist() == [0.0] * (MAX_ACTIONS - 3)
    from net.encoder import ACTION_KIND_IDX
    assert out["action_kind"][0] == ACTION_KIND_IDX["move"]
    assert out["action_kind"][2] == ACTION_KIND_IDX["switch"]
    assert out["action_move"][0] == v.move.index("earthquake")
    assert out["action_species"][2] == v.species.index("volcarona")


def test_empty_action_list_fully_masked():
    ctx = make_ctx()
    out = encode_context(ctx, make_vocabs(ctx))  # no actions
    assert out["action_mask"].sum() == 0.0
    assert out["action_kind"].sum() == 0


# ── sequence encoder (the GRU input shape) ────────────────────────────
def test_encode_sequence_shape_and_mask():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    seq = [ctx, ctx, ctx]  # 3 turns (content irrelevant to the shape contract)
    out = encode_sequence(seq, v)
    t = 3
    for fld in LAYOUT:
        assert out[fld.name].shape == (t,) + fld.shape, fld.name
    assert out["battle_mask"].shape == (t,)
    assert out["battle_mask"].tolist() == [1.0, 1.0, 1.0]


# ── vocab determinism + OOV safety ────────────────────────────────────
def test_vocab_deterministic_and_oov_safe():
    a = Vocab(["zeta", "alpha", "mid"])
    b = Vocab(["mid", "zeta", "alpha"])
    assert a.index("alpha") == b.index("alpha") != 0
    assert a.index("alpha") < a.index("mid") < a.index("zeta")  # sorted order
    assert a.index("never-seen") == 0  # OOV -> <pad>
    assert a.index("") == 0


def test_unknown_ids_never_crash_encoder():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    # drop one species from the vocab so it becomes OOV in the encoder
    v2 = Vocabs(
        species=Vocab(set(v.species._id2idx) - {"garchomp"}),
        ability=v.ability, item=v.item, move=v.move,
        weather=v.weather, terrain=v.terrain,
    )
    out = encode_context(ctx, v2)  # must not raise
    assert out["opp_species"][0] == 0  # garchomp now OOV -> <pad>


# ── REAL-DATA gate (the false-truth check) ────────────────────────────
# The synthetic make_ctx() above could drift from the real get_context(); this
# replays REAL human-ladder battles through the belief stack and encodes the
# actual contexts — so the encoder is pinned to the data it will actually see.
@needs_fixture
def test_encoder_on_real_corpus_contexts():
    from corpus.pull import load_local
    from corpus.trajectories import replay_many

    traces = [t for t in replay_many(load_local(str(FIXTURE), 20))
              if t.skipped is None]
    assert traces, "no real battles replayed"
    contexts = [ctx for t in traces for ctx in t.context_seq]
    assert contexts

    # build ONE shared Vocabs from every id present in the REAL contexts
    # (the S4.2 production path builds the same from the legality DB)
    species, abilities, items, moves, weather, terrain = (set() for _ in range(6))
    for ctx in contexts:
        f = ctx.get("field", {})
        if f.get("weather"):
            weather.add(f["weather"])
        if f.get("terrain"):
            terrain.add(f["terrain"])
        for blk in ctx.get("own", {}).values():
            for k in ("name", "species"):
                if blk.get(k):
                    species.add(blk[k])
            if blk.get("ability"):
                abilities.add(blk["ability"])
            if blk.get("item"):
                items.add(blk["item"])
            moves.update(blk.get("moves", []))
        for blk in ctx.get("opponent", {}).values():
            if blk.get("species"):
                species.add(blk["species"])
            moves.update(blk.get("moves", []))
            for c in blk.get("posterior", []):
                if c.get("ability"):
                    abilities.add(c["ability"])
                moves.update(c.get("moves", []))
            moves.update(blk.get("top_moves", []))
    voc = Vocabs(species=Vocab(species), ability=Vocab(abilities), item=Vocab(items),
                 move=Vocab(moves), weather=Vocab(weather), terrain=Vocab(terrain))

    # every real context: exact registry shape, finite floats, deterministic
    step = max(1, len(contexts) // 30)
    for ctx in contexts[::step]:
        out = encode_context(ctx, voc)
        out2 = encode_context(ctx, voc)
        for fld in LAYOUT:
            assert out[fld.name].shape == fld.shape, (fld.name, ctx)
            if np.issubdtype(out[fld.name].dtype, np.floating):
                assert np.isfinite(out[fld.name]).all(), (fld.name, "non-finite on real ctx")
            assert np.array_equal(out[fld.name], out2[fld.name]), (fld.name, "nondeterministic")

    # the sequence (GRU input) contract on one real battle
    seq = encode_sequence(traces[0].context_seq, voc)
    t = len(traces[0].context_seq)
    for fld in LAYOUT:
        assert seq[fld.name].shape == (t,) + fld.shape, fld.name
    assert seq["battle_mask"].tolist() == [1.0] * t
