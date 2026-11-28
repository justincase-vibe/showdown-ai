"""C4 gate eval (tools/gate_eval.py): hermetic pins (no server, no torch).

The live run (``run_eval_battles`` via the shared battle loop) needs a
server pool — that is the CPU/GPU smoke, run manually. These pin the PURE
parts: the Wilson lower-CI math, the decision-rule verdict, and the
per-opponent pooling. The reference Wilson values were verified by hand
against the closed-form formula (z=1.959963984540054).

NOTE (2026-11-28): the pins below were RE-COMPUTED after the z^2/(4n^2) fix
in tools/gate_eval.py — the original pins had been derived with the wrong
term z^2/(4n), which inflated every half-width ~2.2x (plan §1.2, resolved).

Run:  python -m pytest tools/tests/test_gate_eval.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.gate_eval import (  # noqa: E402
    GATE_CONTINUE, GATE_PASS, PRIMARY_OPP, apply_validity, build_validity,
    gate_verdict, min_mode_bound, pool_rows, wilson_lower, wilson_upper,
    _entry_stats, _parse_subjects,
)


# ── wilson_lower ────────────────────────────────────────────────────────────

def test_wilson_no_data_is_nan():
    assert math.isnan(wilson_lower(0, 0))
    assert math.isnan(wilson_lower(5, -1))


def test_wilson_zero_wins_is_zero():
    assert wilson_lower(0, 10) == 0.0


def test_wilson_extreme_small_n():
    # verified closed-form values (z = 1.959963984540054), recomputed 2026-11-28
    # after the z^2/(4n^2) fix (the old pins used the wrong term z^2/(4n))
    assert abs(wilson_lower(1, 10) - 0.017876) < 1e-4
    assert abs(wilson_lower(5, 10) - 0.236593) < 1e-4
    assert abs(wilson_lower(9, 10) - 0.595850) < 1e-4
    assert abs(wilson_lower(10, 10) - 0.722467) < 1e-4


def test_wilson_gate_scale_values():
    # the values the gate verdict actually lives on (n ~ 400); recomputed
    # 2026-11-28 after the z^2/(4n^2) fix
    assert abs(wilson_lower(200, 400) - 0.451235) < 1e-4
    assert abs(wilson_lower(280, 400) - 0.653363) < 1e-4
    # monotonic in wins
    assert wilson_lower(150, 407) < wilson_lower(200, 407) < wilson_lower(280, 407)


def test_wilson_upper_symmetric():
    assert abs(wilson_upper(200, 400) - (1.0 - 0.451235)) < 1e-4
    # 10/10: the clamp may not fire (float sum lands 1 ulp below 1.0)
    assert abs(wilson_upper(10, 10) - 1.0) < 1e-9
    # 0/10: the upper bound is POSITIVE (true p could be > 0) — the classic
    # reason Wilson beats the naive interval
    assert 0.0 < wilson_upper(0, 10) < 1.0
    assert math.isnan(wilson_upper(5, 0))
    # interval brackets the point estimate
    assert wilson_lower(70, 100) < 0.70 < wilson_upper(70, 100)


# ── gate_verdict (the M1 decision rule) ─────────────────────────────────────

def test_verdict_thresholds():
    assert GATE_PASS == 0.40
    assert GATE_CONTINUE == 0.25
    assert gate_verdict(0.40) == "PASS"
    assert gate_verdict(0.55) == "PASS"
    assert gate_verdict(0.3999) == "CONTINUE"
    assert gate_verdict(0.25) == "CONTINUE"
    assert gate_verdict(0.2499) == "STOP"
    assert gate_verdict(0.0) == "STOP"


def test_verdict_no_data():
    assert gate_verdict(float("nan")) == "NO_DATA"
    assert gate_verdict(None) == "NO_DATA"


# ── per-mode stats + pooled verdict (C1-C5 spec) ────────────────────────────

def test_primary_opp_is_plain_strategy():
    assert PRIMARY_OPP == "strategy"


def test_min_mode_bound_takes_worst():
    assert abs(min_mode_bound({"greedy": 0.50, "sampled": 0.42}) - 0.42) < 1e-12
    assert abs(min_mode_bound({"greedy": 0.30}) - 0.30) < 1e-12


def test_min_mode_bound_empty_or_nan():
    assert math.isnan(min_mode_bound({}))
    assert math.isnan(min_mode_bound({"greedy": float("nan")}))


def test_min_mode_bound_ignores_nan_entries():
    # one mode errored (nan) -> the verdict falls back to the real bound
    assert abs(min_mode_bound({"greedy": 0.41, "sampled": float("nan")})
               - 0.41) < 1e-12


def _row(w, l, d=0):
    return {"eval_w": w, "eval_l": l, "eval_d": d, "eval_crashed": 0,
            "eval_wr": w / (w + l + d) if (w + l + d) else float("nan")}


def test_entry_stats_per_mode_and_pooled():
    pairs = {
        "strategy": {
            "greedy": _row(180, 20),
            "sampled": _row(120, 80),
        },
        "strategy_posterior": {"greedy": _row(90, 110)},
    }
    st = _entry_stats(pairs)
    assert set(st["per_mode_lower_ci"]["strategy"]) == {"greedy", "sampled"}
    assert st["per_mode_lower_ci"]["strategy"]["greedy"] \
        > st["per_mode_lower_ci"]["strategy"]["sampled"]
    p = st["pooled"]["strategy"]
    assert p["wins"] == 300 and p["n"] == 400
    assert 0.0 <= p["lower_ci_95"] < 0.75 < p["upper_ci_95"] <= 1.0
    # verdict bound = WORST pooled bound of strategy + strategy_posterior
    assert st["verdict_bound"] == st["pooled"]["strategy_posterior"]["lower_ci_95"]
    assert st["base_verdict"] == gate_verdict(st["verdict_bound"])
    assert st["mode_diff_warnings"]  # 0.45 vs 0.60 differ by 15 pts > 10


def test_entry_stats_mode_diff_warning_threshold():
    pairs = {"strategy": {"greedy": _row(55, 45), "sampled": _row(60, 40)}}
    assert _entry_stats(pairs)["mode_diff_warnings"] == []


def test_entry_stats_skips_error_rows():
    pairs = {"strategy": {"greedy": _row(50, 50), "sampled": {"error": "boom"}}}
    st = _entry_stats(pairs)
    assert "strategy" in st["pooled"]
    assert set(st["per_mode_lower_ci"]["strategy"]) == {"greedy"}


# ── validity rules (C1/C2 based) ─────────────────────────────────────────────

def _entries_with_c1(c1_wr_n, c2_wr_n=None):
    def _pooled(w, n):
        return {"wins": w, "losses": n - w, "draws": 0, "crashed": 0,
                "n": n, "win_rate": w / n,
                "lower_ci_95": wilson_lower(w, n),
                "upper_ci_95": wilson_upper(w, n)}
    entries = {}
    if c1_wr_n is not None:
        entries["control:random_init"] = {
            "stats": {"pooled": {"strategy": _pooled(*c1_wr_n)}}}
    if c2_wr_n is not None:
        entries["control:random"] = {
            "stats": {"pooled": {"strategy": _pooled(*c2_wr_n)}}}
    return entries


def test_validity_c2_too_strong_invalidates():
    # C2 random bot beats strategy 30% > 25% -> YARDSTICK_INVALID
    v = build_validity(_entries_with_c1((80, 400), (120, 400)))
    assert v["yardstick_invalid"] is True
    assert any("C2" in r for r in v["reasons"])


def test_validity_c1_upper_bound_invalidates():
    # C1 random-init at 50% (n=400): upper bound ~0.58 >= 0.40 -> invalid
    v = build_validity(_entries_with_c1((200, 400)))
    assert v["yardstick_invalid"] is True
    assert any("UPPER" in r for r in v["reasons"])


def test_validity_clean_when_floor_is_weak():
    # floor at 20% (n=400): upper bound ~0.27 < 0.40, C2 20% <= 25% -> valid
    v = build_validity(_entries_with_c1((80, 400), (80, 400)))
    assert v["yardstick_invalid"] is False
    assert v["c1"] is not None and v["c2"] is not None


def test_apply_validity_caps_pass_when_intervals_overlap():
    # C1 floor at 20% (n=400, upper 0.242); net at 32% (n=100, lower 0.227):
    # margin 12 pts >= 10 BUT the 95% intervals OVERLAP -> cap at CONTINUE.
    # 2026-11-28: re-pinned on the CORRECT Wilson widths — the old case
    # (130/400 vs 80/400, margin 12.5 pts) only overlapped because of the
    # inflated CIs; with correct widths it is a clean non-overlap (PASS).
    v = build_validity(_entries_with_c1((80, 400), (80, 400)))
    net = {"win_rate": 32 / 100, "lower_ci_95": wilson_lower(32, 100),
           "upper_ci_95": wilson_upper(32, 100)}
    # the cap must come from the OVERLAP branch (margin is fine)
    assert 32 / 100 - 80 / 400 >= 0.10
    out = apply_validity("PASS", net, v)
    assert out.startswith("CONTINUE (capped")


def test_apply_validity_margin_ok_but_overlapping_is_capped_explicitly():
    # guards the two conditions of the cap rule separately (correct CIs):
    # (a) margin ok + overlap -> cap;  (b) margin ok + no overlap -> PASS.
    # net margin is 12 pts (float-safe above the 10-pt bar).
    v = build_validity(_entries_with_c1((80, 400), (80, 400)))
    c1_upper = v["c1"]["upper_ci_95"]
    net_overlapping = {"win_rate": 32 / 100,
                       "lower_ci_95": wilson_lower(32, 100),
                       "upper_ci_95": wilson_upper(32, 100)}
    net_clear = {"win_rate": 0.50,
                 "lower_ci_95": wilson_lower(200, 400),
                 "upper_ci_95": wilson_upper(200, 400)}
    # sanity: the fixtures actually exercise the two branches
    assert 32 / 100 - 80 / 400 >= 0.10 \
        and c1_upper >= net_overlapping["lower_ci_95"]
    assert 0.50 - 0.20 >= 0.10 and c1_upper < net_clear["lower_ci_95"]
    assert apply_validity("PASS", net_overlapping, v).startswith(
        "CONTINUE (capped")
    assert apply_validity("PASS", net_clear, v) == "PASS"


def test_apply_validity_pass_survives_with_clear_margin():
    # C1 floor at 20% (n=400, upper ~0.27); net at 50% (n=400, lower ~0.39):
    # margin 30 pts and NON-overlapping intervals -> PASS stands
    v = build_validity(_entries_with_c1((80, 400), (80, 400)))
    net = {"win_rate": 0.5, "lower_ci_95": wilson_lower(200, 400),
           "upper_ci_95": wilson_upper(200, 400)}
    assert apply_validity("PASS", net, v) == "PASS"


def test_apply_validity_invalid_yardstick_overrides():
    v = build_validity(_entries_with_c1((200, 400)))
    net = {"win_rate": 0.9, "lower_ci_95": 0.8, "upper_ci_95": 0.95}
    assert apply_validity("PASS", net, v).startswith("YARDSTICK_INVALID")


def test_c1_falls_back_to_c2_row():
    # no random_init entry; the C2 row (identical net here) fills C1
    v = build_validity(_entries_with_c1(None, (80, 400)))
    assert v["c1"] is not None and v["c2"] is not None


def test_parse_subjects_plain_and_override():
    out = _parse_subjects("random_init,heuristic,strategy:strategy,strategy_posterior:strategy")
    assert out == [("random_init", None), ("heuristic", None),
                   ("strategy", ["strategy"]),
                   ("strategy_posterior", ["strategy"])]
    assert _parse_subjects(None) == []


# ── pool_rows (reporting-only pool of play modes for one opponent) ──────────

def test_pool_rows_sums_and_never_counts_draws_as_wins():
    rows = [
        {"eval_w": 100, "eval_l": 100, "eval_d": 5, "eval_crashed": 1},
        {"eval_w": 50, "eval_l": 150, "eval_d": 2, "eval_crashed": 0},
    ]
    p = pool_rows(rows)
    assert p["wins"] == 150 and p["losses"] == 250 and p["draws"] == 7
    assert p["n"] == 407 and p["crashed"] == 1
    assert abs(p["win_rate"] - 150 / 407) < 1e-12
    assert 0.0 <= p["lower_ci_95"] <= 1.0


def test_pool_rows_empty():
    p = pool_rows([])
    assert p["n"] == 0
    assert math.isnan(p["win_rate"])
    assert math.isnan(p["lower_ci_95"])


# ── summarize_eval_records (the shared row contract, refactored out of the
#    trainer) ────────────────────────────────────────────────────────────────

def test_summarize_eval_records_contract():
    from gui.m1_gate_main import summarize_eval_records
    records = [
        {"a1_won": True, "a2_won": None, "crashed": False},   # win
        {"a1_won": True, "a2_won": None, "crashed": False},   # win
        {"a1_won": None, "a2_won": True, "crashed": False},   # loss
        {"a1_won": None, "a2_won": None, "crashed": False},   # draw (stall)
        {"a1_won": True, "a2_won": True, "crashed": False},   # double K.O. = draw
        {"a1_won": None, "a2_won": None, "crashed": True},    # crashed
    ]
    row = summarize_eval_records(records)
    assert row["eval_w"] == 2 and row["eval_l"] == 1
    assert row["eval_d"] == 2 and row["eval_crashed"] == 1
    assert abs(row["eval_wr"] - 2 / 5) < 1e-12


def test_summarize_eval_records_empty_is_nan():
    from gui.m1_gate_main import summarize_eval_records
    row = summarize_eval_records([])
    assert row["eval_w"] == 0 and row["eval_crashed"] == 0
    assert math.isnan(row["eval_wr"])
    assert row["max_steps_terminations"] == 0
    assert math.isnan(row["max_steps_share"])


def test_summarize_eval_records_max_steps_share():
    """§2.4: battles ending at the loop cap are counted separately (they are
    scored as draws, but reported as not-a-real-result). Crashed records are
    never counted; records without a steps key (older rows / fixtures) are
    not counted either."""
    from gui.m1_gate_main import summarize_eval_records
    records = [
        {"a1_won": None, "a2_won": None, "crashed": False, "steps": 200},
        {"a1_won": None, "a2_won": None, "crashed": False, "steps": 200},
        {"a1_won": True, "a2_won": None, "crashed": False, "steps": 47},
        {"a1_won": None, "a2_won": True, "crashed": False, "steps": 88},
        {"a1_won": None, "a2_won": None, "crashed": True, "steps": 200},
    ]
    row = summarize_eval_records(records)  # default max_steps=200
    assert row["eval_w"] == 1 and row["eval_l"] == 1 and row["eval_d"] == 2
    assert row["eval_crashed"] == 1
    assert row["max_steps_terminations"] == 2
    assert abs(row["max_steps_share"] - 0.5) < 1e-12
    # the pre-existing row contract is unchanged
    assert abs(row["eval_wr"] - 1 / 4) < 1e-12
    # step-less records: never a max_steps termination
    row2 = summarize_eval_records(
        [{"a1_won": True, "a2_won": None, "crashed": False}])
    assert row2["max_steps_terminations"] == 0
    assert row2["max_steps_share"] == 0.0


# ── the C3 wrapper must keep its exact row contract (6.6a) ──────────────────

def test_eval_round_m1_contract_shape():
    """_eval_round_m1 returns eval_* keys ONLY (plus eval_opp) — the additive
    contract the trainer's metrics row depends on. Pinned with fakes; no
    server, no torch (run_eval_battles is monkeypatched)."""
    import gui.m1_gate_main as m1

    fake_row = {"eval_w": 3, "eval_l": 1, "eval_d": 0, "eval_crashed": 0,
                "eval_wr": 0.75, "eval_battles": 4}
    calls = {}

    def fake_run_eval_battles(net, opp_ref, fmt, vocabs, device, n_battles,
                              max_steps, n_servers, per_server, port_base,
                              seed_base, deterministic=True, battle_hook=None,
                              stop_check=None, label="eval", my_ref=None,
                              my_side="p1"):
        calls.update(dict(net=net, opp_ref=opp_ref, n_battles=n_battles,
                          seed_base=seed_base, deterministic=deterministic,
                          my_side=my_side))
        return dict(fake_row, eval_side=my_side)

    orig = m1.run_eval_battles
    m1.run_eval_battles = fake_run_eval_battles
    try:
        class _FakePool(dict):
            def names(self):
                return list(self.keys())

        class _FakeTrainer:
            net = "NET"
            round = 3

        pool = _FakePool()
        pool["strategy"] = "STRATEGY_REF"
        row = m1._eval_round_m1(_FakeTrainer(), pool, "gen9ou", "vocabs",
                                "cpu", "strategy", 4, 200, 6, 2, 49980,
                                None, lambda: False)
        # exact contract: additive eval_* keys + eval_opp + eval_side
        # (S3.1), nothing else
        assert set(row.keys()) == {"eval_opp", "eval_w", "eval_l", "eval_d",
                                   "eval_crashed", "eval_wr", "eval_battles",
                                   "eval_side"}
        assert row["eval_opp"] == "strategy"
        # default placement unchanged: subject on p1
        assert calls["my_side"] == "p1" and row["eval_side"] == "p1"
        # the C3 wrapper stays GREEDY and uses the round-disjoint seed space
        assert calls["deterministic"] is True
        assert calls["seed_base"] == 1_000_000 + 3 * 1000
        assert calls["n_battles"] == 4
    finally:
        m1.run_eval_battles = orig


# ── S3.1: side-aware learner W/L helpers + eval my_side plumbing ────────────

def _rec(a1, a2, side=None):
    r = {"a1_won": a1, "a2_won": a2}
    if side is not None:
        r["side"] = side
    return r


def test_learner_won_loss_p1_default():
    """No 'side' key = the historical a1-is-learner mapping (regression)."""
    from gui.m1_gate_main import _learner_won, _learner_lost
    assert _learner_won(_rec(True, None))
    assert not _learner_lost(_rec(True, None))
    assert _learner_lost(_rec(None, True))
    assert not _learner_won(_rec(None, True))
    # draws (stall / double K.O.): neither a win nor a loss
    assert not _learner_won(_rec(None, None))
    assert not _learner_lost(_rec(None, None))
    assert not _learner_won(_rec(True, True))
    assert not _learner_lost(_rec(True, True))


def test_learner_won_loss_p1_explicit():
    from gui.m1_gate_main import _learner_won, _learner_lost
    assert _learner_won(_rec(True, None, "p1"))
    assert _learner_lost(_rec(None, True, "p1"))
    assert not _learner_won(_rec(None, None, "p1"))
    assert not _learner_won(_rec(True, True, "p1"))
    assert not _learner_lost(_rec(True, True, "p1"))


def test_learner_won_loss_p2_swaps_seats():
    """Learner sat on p2: the a2 seat is the learner's result."""
    from gui.m1_gate_main import _learner_won, _learner_lost
    assert _learner_won(_rec(None, True, "p2"))
    assert not _learner_lost(_rec(None, True, "p2"))
    assert _learner_lost(_rec(True, None, "p2"))
    assert not _learner_won(_rec(True, None, "p2"))
    # draws: seat-independent — still neither
    assert not _learner_won(_rec(None, None, "p2"))
    assert not _learner_lost(_rec(None, None, "p2"))
    assert not _learner_won(_rec(True, True, "p2"))
    assert not _learner_lost(_rec(True, True, "p2"))


def test_eval_round_m1_my_side_passthrough():
    """S3.1: _eval_round_m1 forwards my_side to the shared loop; the row
    contract gains eval_side (additive). Default stays p1."""
    import gui.m1_gate_main as m1

    calls = []

    def fake_run_eval_battles(net, opp_ref, fmt, vocabs, device, n_battles,
                              max_steps, n_servers, per_server, port_base,
                              seed_base, deterministic=True, battle_hook=None,
                              stop_check=None, label="eval", my_ref=None,
                              my_side="p1"):
        calls.append(my_side)
        return {"eval_w": 0, "eval_l": 0, "eval_d": 0, "eval_crashed": 0,
                "eval_wr": float("nan"), "eval_battles": 1,
                "eval_side": my_side}

    orig = m1.run_eval_battles
    m1.run_eval_battles = fake_run_eval_battles
    try:
        class _FakePool(dict):
            def names(self):
                return list(self.keys())

        class _FakeTrainer:
            net = "NET"
            round = 0

        pool = _FakePool()
        pool["strategy"] = "STRATEGY_REF"
        for side in ("p1", "p2"):
            row = m1._eval_round_m1(_FakeTrainer(), pool, "gen9ou", "vocabs",
                                    "cpu", "strategy", 1, 200, 6, 2, 49980,
                                    None, lambda: False, my_side=side)
            assert calls[-1] == side
            assert row["eval_side"] == side
        # default stays p1 (historical placement)
        row = m1._eval_round_m1(_FakeTrainer(), pool, "gen9ou", "vocabs",
                                "cpu", "strategy", 1, 200, 6, 2, 49980,
                                None, lambda: False)
        assert calls[-1] == "p1" and row["eval_side"] == "p1"
    finally:
        m1.run_eval_battles = orig
