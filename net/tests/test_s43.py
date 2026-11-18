"""S4.3 — BC warm-start gate tests (PLAN.md §3c S4.3).

Fast + offline (no replay, no network): they pin the **"always-switch" verdict
logic** — the critical anti-collapse detector the user's #1 concern is built on
("the net memorized default = switch instead of state -> action", hit in the
old system) — and the **evaluator** on synthetic samples.

The full N-battle run is the script (``python -m net.bc``); the heavy
real-corpus overfit is already covered by the S4.2 overfit-32 gate. What these
tests lock in:
  * ``verdict`` FAILS a net that only gets switches right (move-accuracy
    collapses below chance);
  * ``verdict`` FAILS the mirror collapse (always-move, switch-rate ~0);
  * ``verdict`` FAILS when the action mix is wildly off even if both
    accuracies beat chance (the predicted-switch-rate gate);
  * ``verdict`` PASSES a healthy net (both accuracies beat chance AND the
    predicted switch-rate is near the human rate);
  * ``evaluate`` counts move/switch accuracy correctly and reports well-defined
    chance baselines; after overfitting it reports near-perfect accuracy;
  * ``make_split`` is BY BATTLE — a battle's decisions never straddle
    train / held-out (no leakage).
"""
from __future__ import annotations

import copy

import torch

from net import (
    PokeNet,
    batch_to_torch,
    encode_context,
    overfit_train,
    samples_to_batch,
)
from net.batch import (Sample, _alive_bench, _shuffle_actions, build_samples,
                       decision_actions)
import net.bc as bc
from net.tests.test_s41 import make_ctx, make_vocabs
from corpus.trajectories import Decision


# ── the "always-switch" verdict (pure logic — deterministic) ────────────
def _base():
    return {
        "n": 100, "n_move": 60, "n_switch": 40,
        "move_acc": 0.7, "move_chance": 0.4, "move_beats_chance": True,
        "move_const_acc": 0.3, "move_beats_const": True,
        "switch_acc": 0.6, "switch_chance": 0.35, "switch_beats_chance": True,
        "switch_const_acc": 0.2, "switch_beats_const": True,
        "top1_acc": 0.65, "overall_chance": 0.2, "top1_beats_chance": True,
        "top1_const_acc": 0.3, "top1_beats_const": True,
        "pred_switch_rate": 0.42, "human_switch_rate": 0.40, "switch_rate_gap": 0.02,
        # D81: a healthy net does first-use and repeat comparably (small gap),
        # so the new shortcut gate does NOT fire here.
        "n_first_use": 40, "n_repeat": 20,
        "first_use_move_acc": 0.72, "repeat_move_acc": 0.68, "shortcut_gap": 0.04,
    }


def test_verdict_flags_collapse_to_switch():
    # a net that always predicts a switch: move-accuracy ~0, predicted-sw ~1.0
    m = _base()
    m.update({"move_acc": 0.0, "move_beats_chance": False,
              "pred_switch_rate": 0.99, "switch_rate_gap": 0.59,
              "switch_acc": 0.9, "top1_acc": 0.3})
    ok, fails = bc.verdict(m)
    assert not ok
    assert any("move-accuracy" in f for f in fails)
    assert any("switch-rate" in f for f in fails)


def test_verdict_flags_collapse_to_move():
    # the mirror: always uses a move, never switches (switch-accuracy ~0,
    # predicted-switch-rate ~0 while the human rate is ~0.4)
    m = _base()
    m.update({"switch_acc": 0.0, "switch_beats_chance": False,
              "pred_switch_rate": 0.01, "switch_rate_gap": -0.39,
              "move_acc": 0.8, "top1_acc": 0.5})
    ok, fails = bc.verdict(m)
    assert not ok
    assert any("switch-accuracy" in f for f in fails)
    assert any("switch-rate" in f for f in fails)


def test_verdict_fails_on_wild_action_mix_even_if_accs_beat_chance():
    # both accuracies beat chance, but the net predicts switch on 95% while
    # humans switch on 35% -> it is following the action prior, not the state
    m = _base()
    m.update({"pred_switch_rate": 0.95, "switch_rate_gap": 0.60})
    ok, fails = bc.verdict(m)
    assert not ok
    assert any("switch-rate" in f for f in fails)


def test_verdict_passes_healthy_net():
    m = _base()  # both accuracies beat chance + switch-rate near human
    ok, fails = bc.verdict(m)
    assert ok, fails
    assert fails == []


def test_verdict_flags_fixed_action_net():
    # THE regression that the first N=50 run exposed: a net that always
    # predicts ONE action index can clear the chance gate without learning
    # anything (short action lists put the human target at index 0 ~65% of
    # the time, so always-index-0 scores 0.65 vs chance 0.22). If the net's
    # accuracy merely EQUALS its constant-index baseline it must FAIL.
    m = _base()
    m.update({"move_acc": 0.65, "move_const_acc": 0.65, "move_beats_const": False,
              "switch_acc": 0.50, "switch_const_acc": 0.50, "switch_beats_const": False,
              "top1_acc": 0.60, "top1_const_acc": 0.60, "top1_beats_const": False})
    ok, fails = bc.verdict(m)
    assert not ok, fails
    assert any("constant-index" in f for f in fails)


def test_verdict_passes_net_that_beats_constant_baseline():
    # the healthy case must keep passing: accuracy strictly above BOTH the
    # chance and the constant-index baselines
    m = _base()
    ok, fails = bc.verdict(m)
    assert ok, fails


def test_evaluate_reports_constant_index_baseline():
    # every target at index 0 -> the constant-index baseline is 1.0 for both
    # kinds, even though an untrained net's accuracy is far below it
    samples, vocabs = _make_synthetic_samples(12)
    for s in samples:
        s.target = 0
    net = PokeNet(vocabs)
    m = bc.evaluate(net, samples)
    assert m["move_const_acc"] == 1.0
    assert m["switch_const_acc"] == 1.0
    assert m["top1_const_acc"] == 1.0
    assert not m["move_beats_const"]
    assert not m["switch_beats_const"]
    assert not m["top1_beats_const"]


def test_verdict_requires_both_action_types_present():
    # a corpus with no move-decisions cannot certify move-learning
    m = _base()
    m.update({"n_move": 0, "move_beats_chance": False})
    ok, fails = bc.verdict(m)
    assert not ok
    assert any("move-decisions" in f for f in fails)


def test_verdict_flags_first_use_shortcut():
    # THE leak the friend flagged (D81): the net learns "new move = the one to
    # pick" instead of state -> action. Overall move-accuracy can still beat
    # chance (the earlier gates pass), but first-use is ~0.95 while repeat is
    # ~0.40 -> the gap is the tell. This must FAIL.
    m = _base()
    m.update({"n_first_use": 40, "n_repeat": 20,
              "first_use_move_acc": 0.95, "repeat_move_acc": 0.40,
              "shortcut_gap": 0.55})
    ok, fails = bc.verdict(m)
    assert not ok, fails
    assert any("shortcut" in f for f in fails)


def test_verdict_skips_shortcut_gate_when_split_too_small():
    # below MIN_SPLIT_SAMPLES on either side the split is too small to be
    # meaningful -> the gate must NOT fire (even with a large gap).
    m = _base()
    m.update({"n_first_use": 3, "n_repeat": 20,              # first-use < 8
              "first_use_move_acc": 1.0, "repeat_move_acc": 0.40,
              "shortcut_gap": 0.60})
    ok, fails = bc.verdict(m)
    assert ok, fails
    assert not any("shortcut" in f for f in fails)


def test_evaluate_reports_first_use_repeat_split():
    # evaluate() must split move-decision accuracy on Sample.first_use and
    # report the two accuracies + the gap.
    samples, vocabs = _make_synthetic_samples(8)
    move_samples = [s for s in samples if s.kind == "move"]
    # alternate first-use / repeat deterministically
    for i, s in enumerate(move_samples):
        s.first_use = (i % 2 == 0)
    n_first = sum(1 for s in move_samples if s.first_use)
    n_repeat = sum(1 for s in move_samples if not s.first_use)
    net = PokeNet(vocabs)  # untrained; we only check the COUNTING, not accuracy
    m = bc.evaluate(net, move_samples)
    assert m["n_first_use"] == n_first
    assert m["n_repeat"] == n_repeat
    assert n_first > 0 and n_repeat > 0
    # both accuracies are defined (in [0,1]) and the gap is their difference
    assert m["first_use_move_acc"] is not None
    assert m["repeat_move_acc"] is not None
    assert abs(m["shortcut_gap"] - (m["first_use_move_acc"] - m["repeat_move_acc"])) < 1e-9


# ── the evaluator on synthetic samples (fast, no replay) ────────────────
def _make_synthetic_samples(n: int):
    base = make_ctx()
    vocabs = make_vocabs(base)
    samples = []
    for i in range(n):
        ctx = copy.deepcopy(base)
        ctx["turn"] = (i % 12) + 1          # vary the input so samples are distinct
        if i % 2 == 0:                      # a MOVE decision: 3 moves + 2 switches
            actions = [{"kind": "move", "move": "earthquake"},
                       {"kind": "move", "move": "icebeam"},
                       {"kind": "move", "move": "roar"},
                       {"kind": "switch", "species": "volcarona"},
                       {"kind": "switch", "species": "dragonite"}]
            target = i % 3
            kind = "move"
        else:                               # a SWITCH decision: 3 switches
            actions = [{"kind": "switch", "species": "volcarona"},
                       {"kind": "switch", "species": "dragonite"},
                       {"kind": "switch", "species": "toxicroc"}]
            target = i % 3
            kind = "switch"
        seq = [encode_context(ctx, vocabs, actions)]
        samples.append(Sample(battle_id=f"syn{i}", turn=ctx["turn"], kind=kind,
                              target=target, actions=actions, seq=seq))
    return samples, vocabs


def test_evaluate_counts_and_overfits():
    samples, vocabs = _make_synthetic_samples(12)
    # Force every target to position 0: the task becomes trivially learnable (the
    # net only needs to bias toward the first legal action), which cleanly
    # verifies that ``evaluate`` counts accuracy + chance correctly on a real
    # forward pass. (Learning a REAL state->action mapping on real data is the
    # S4.2 overfit-32 gate's job, already green.)
    for s in samples:
        s.target = 0
    net = PokeNet(vocabs)
    sb = samples_to_batch(samples)
    b = batch_to_torch(sb["batch"])
    targets = torch.as_tensor(sb["targets"], dtype=torch.int64)
    ts = torch.as_tensor(sb["ts"], dtype=torch.int64)
    final = overfit_train(net, [b], [targets], [ts], epochs=80, lr=3e-3)
    assert final < 0.2, final
    m = bc.evaluate(net, samples)
    assert m["n"] == 12
    assert m["n_move"] == 6 and m["n_switch"] == 6
    # it memorized the state -> action mapping (the "never learns" guard holds)
    assert m["top1_acc"] > 0.9, m
    assert m["move_acc"] > 0.9
    assert m["switch_acc"] > 0.9
    # chance baselines are well-defined and in [0, 1]
    assert 0.0 < m["move_chance"] <= 1.0
    assert 0.0 < m["switch_chance"] <= 1.0
    assert 0.0 < m["overall_chance"] <= 1.0
    # and the memorized net beats chance on both action types
    assert m["move_beats_chance"]
    assert m["switch_beats_chance"]


def test_evaluate_reports_switch_rate_from_actions():
    # all MOVE decisions: the human switch-rate is 0, so any net switch-pick
    # shows up in pred_switch_rate (the collapse signal the verdict reads).
    samples, vocabs = _make_synthetic_samples(6)
    move_only = [s for s in samples if s.kind == "move"]
    net = PokeNet(vocabs)  # untrained -> near-uniform over the legal list
    m = bc.evaluate(net, move_only)
    assert m["human_switch_rate"] == 0.0
    assert 0.0 <= m["pred_switch_rate"] <= 1.0
    assert m["n_switch"] == 0


# ── the split is BY BATTLE (no train/eval leakage) ──────────────────────
class _FakeTrace:
    def __init__(self, bid: str):
        self.battle_id = bid
        self.formatid = "gen9ou"   # build_samples (D73 decoy padding) reads it
        self.skipped = None
        self.context_seq = [make_ctx()]
        self.decisions = [Decision(turn=1, pov="p1", kind="move",
                                   action="icebeam", species="landorustherian",
                                   legal=True)]


def test_split_is_by_battle_no_leakage(monkeypatch):
    fakes = [_FakeTrace(f"b{i}") for i in range(8)]
    monkeypatch.setattr(bc, "replay_many", lambda battles: fakes)
    vocabs = make_vocabs(make_ctx())
    split = bc.make_split([{} for _ in range(8)], vocabs, heldout_frac=0.5, seed=0)
    train_ids = {s.battle_id for s in split.train_samples}
    held_ids = {s.battle_id for s in split.heldout_samples}
    assert train_ids and held_ids
    assert train_ids.isdisjoint(held_ids)              # no battle straddles the split
    assert split.n_train_battles + split.n_heldout_battles == 8
    assert split.n_train_battles == len(train_ids)
    assert split.n_heldout_battles == len(held_ids)


# ── P0 regressions: the BC action list must NOT degenerate to one action ────

def _ctx_own():
    """A turn-start context shaped exactly like ``get_context()``: one active
    own mon (2 revealed moves) + two LIVE bench mons + one FAINTED bench mon.
    Benched own mons carry ``hp_frac=0.0`` (the own side only tracks the
    active mon's facts) — liveness is the ``fainted`` flag."""
    return {
        "turn": 3,
        "fainted": {"p1": 1, "p2": 0},
        "active": {"p1": "p1a", "p2": "p2a"},
        "field": {},
        "own": {
            "p1a": {"name": "Garchomp", "active": True, "fainted": False,
                    "hp_frac": 0.8, "moves": ["earthquake", "icebeam"]},
            "p1b": {"name": "Volcarona", "active": False, "fainted": False,
                    "hp_frac": 0.0, "moves": []},
            "p1c": {"name": "Dragonite", "active": False, "fainted": False,
                    "hp_frac": 0.0, "moves": []},
            "p1d": {"name": "Toxicroak", "active": False, "fainted": True,
                    "hp_frac": 0.0, "moves": []},
        },
        "opponent": {},
    }


def test_alive_bench_is_not_active_and_not_fainted():
    """P0(a): the bench = own mons that are NOT active and NOT fainted, regardless
    of hp_frac. The old code required hp_frac>0 and returned [] for every bench,
    degenerating every decision to a single legal action."""
    ctx = _ctx_own()
    assert _alive_bench(ctx) == ["Dragonite", "Volcarona"]  # active + fainted out


def test_move_decision_action_list_is_not_degenerate():
    """P0(a)+P0(b): a move-decision offers the acting mon's revealed moves + the
    human's move AND the alive bench — a real choice, not one legal action."""
    ctx = _ctx_own()
    dec = Decision(turn=3, pov="p1", kind="move", action="icebeam",
                   species="garchomp", legal=True)
    acts, target = decision_actions(ctx, dec)
    moves = [a["move"] for a in acts if a["kind"] == "move"]
    switches = [a["species"] for a in acts if a["kind"] == "switch"]
    assert "move" in [a["kind"] for a in acts] and "switch" in [a["kind"] for a in acts]
    assert {"earthquake", "icebeam"} <= set(moves)      # revealed + the human's move
    assert set(switches) == {"Dragonite", "Volcarona"}  # alive bench
    assert acts[target]["kind"] == "move" and acts[target]["move"] == "icebeam"


def test_switch_decision_action_list_is_the_alive_bench():
    """P0(a): a forced switch-decision offers the alive bench (non-empty after
    the bench fix) with the switched-in mon as a legal position."""
    ctx = _ctx_own()
    dec = Decision(turn=3, pov="p1", kind="switch", action="Volcarona",
                   species="volcarona", legal=True)
    acts, target = decision_actions(ctx, dec)
    assert all(a["kind"] == "switch" for a in acts)
    assert {a["species"] for a in acts} == {"Dragonite", "Volcarona"}
    assert acts[target]["species"] == "Volcarona"


def test_build_samples_drops_degenerate_decisions():
    """D70(B) drop rule, restated under D73 decoy semantics.

    * MOVE decisions are no longer degenerate the old way: D73 pads them to a
      full 4-move set with species-appropriate decoys, so the old "NO revealed
      moves" case is now KEPT (4 move options, target position varies by seed).
    * SWITCH decisions with <2 alive-bench options still carry no selection
      signal (target at a constant index — the fake-PASS failure mode) and
      must be DROPPED; >=2 options must be kept. That is the case the rule
      still guards."""
    base = _ctx_own()

    class _T:
        def __init__(self, ctx, decs):
            self.battle_id = "deg"
            self.formatid = "gen9ou"   # build_samples (D73 decoy padding) reads it
            self.skipped = None
            self.context_seq = [ctx]
            self.decisions = decs

    # (a) move decision, NO revealed moves -> D73 decoys pad to 4 -> KEPT
    ctx1 = copy.deepcopy(base)
    ctx1["own"]["p1a"]["moves"] = []
    t1 = _T(ctx1, [Decision(turn=1, pov="p1", kind="move", action="outrage",
                            species="garchomp", legal=True)])
    # (b) switch decision, NO alive bench (p1b/p1c killed, p1d already fainted)
    #     -> exactly ONE option -> degenerate -> DROPPED
    ctx2 = copy.deepcopy(base)
    ctx2["own"]["p1b"]["fainted"] = True
    ctx2["own"]["p1c"]["fainted"] = True
    t2 = _T(ctx2, [Decision(turn=1, pov="p1", kind="switch", action="Dragonite",
                            species="dragonite", legal=True)])
    # (c) switch decision, two alive bench mons -> two options -> KEPT
    t3 = _T(copy.deepcopy(base),
            [Decision(turn=1, pov="p1", kind="switch", action="Dragonite",
                      species="dragonite", legal=True)])

    vocabs = make_vocabs(make_ctx())
    assert len(build_samples(t1, vocabs)) == 1   # (a) kept (D73 decoys)
    assert build_samples(t2, vocabs) == []       # (b) degenerate -> dropped
    assert len(build_samples(t3, vocabs)) == 1   # (c) kept


def test_order_shuffle_is_deterministic_and_target_follows_human():
    """D70: the per-decision order shuffle must be reproducible (same seed ->
    same permutation) and the target must ALWAYS land on the human's action,
    wherever the permutation puts it."""
    acts = [{"kind": "move", "move": "earthquake"},
            {"kind": "move", "move": "icebeam"},
            {"kind": "switch", "species": "volcarona"}]
    s1, t1 = _shuffle_actions(acts, 1, 12345)
    s2, t2 = _shuffle_actions(acts, 1, 12345)
    assert s1 == s2 and t1 == t2                        # deterministic per seed
    assert s1[t1] == acts[1]                            # target follows the human's action
    orders = set()
    for seed in range(6):
        s, t = _shuffle_actions(acts, 1, seed)
        assert s[t] == acts[1]
        orders.add(tuple(a["move"] if a["kind"] == "move" else a["species"] for a in s))
    assert len(orders) >= 2                             # seeds actually vary the order


def test_build_samples_target_is_human_action_after_shuffle():
    """D70: end-to-end through build_samples, the shuffled target index must
    still point at the human's move (not a stale position)."""
    base = _ctx_own()
    ctx = copy.deepcopy(base)
    ctx["own"]["p1a"]["moves"] = ["earthquake"]

    class _T:
        battle_id = "shuf"
        formatid = "gen9ou"   # build_samples (D73 decoy padding) reads it
        skipped = None
        context_seq = [ctx]
        decisions = [Decision(turn=1, pov="p1", kind="move", action="icebeam",
                              species="garchomp", legal=True)]

    samples = build_samples(_T(), make_vocabs(make_ctx()))
    assert len(samples) == 1
    s = samples[0]
    assert s.actions[s.target]["move"] == "icebeam"    # target = the human's move


def test_own_side_move_used_populates_moves_used():
    """P0(b) root cause: the own side's MoveUsed must populate
    own_facts[slot].moves_used (the opponent side does this in
    SlotBelief.update). It used to be skipped, so get_context's own 'moves' block
    was always empty and _revealed_moves returned []."""
    from belief import events as E
    from belief.state import BeliefState
    bs = BeliefState("gen9ou", own_team={"p1a": {"name": "Garchomp"}})
    bs.update(E.MoveUsed(side="p1", slot="p1a", move="earthquake"))
    bs.update(E.MoveUsed(side="p1", slot="p1a", move="icebeam"))
    assert bs.own_facts["p1a"].moves_used == {"earthquake", "icebeam"}
    assert bs.get_context()["own"]["p1a"]["moves"] == ["earthquake", "icebeam"]
