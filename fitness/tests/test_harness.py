"""Option A / A2 — the harness core, tested HERMETICALLY (no server, no torch,
no GPU). The battle loop is exercised through the driver's test-injection
seams: a fake ``env_factory`` (a scripted winner) + a ``decide`` that returns
``None`` (a pass/wait, which skips ``action_to_pe`` so no real battle object is
needed). This pins the ONE thing A2 owns — the win/loss/draw aggregation and
per-reference separation — independent of the battle mechanics (those are
covered by the league/driver suites).

Run from the project root:

    python -m pytest fitness/tests -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fitness.harness import evaluate, EvalResult, PerRefResult
from fitness.ladder import Ladder, Reference


# ── a scripted fake env: the driver's _step_loop interface, nothing more ─────
class _FakeBattle:
    def __init__(self):
        self.finished = False
        self.won = False


class _FakeEnv:
    """winner: 1 = bot (p1), 2 = reference (p2), 0 = draw (neither won)."""
    def __init__(self, winner: int):
        self.possible_agents = ("a1", "a2")
        self.agent1_to_move = True
        self.agent2_to_move = False
        self.battle1 = _FakeBattle()
        self.battle2 = _FakeBattle()
        self.battle1.won = (winner == 1)
        self.battle2.won = (winner == 2)

    def reset(self):
        return (None, {})

    def step(self, actions):
        self.battle1.finished = True
        self.battle2.finished = True
        a1, a2 = self.possible_agents
        rew = {a1: 0.0, a2: 0.0}
        term = {a1: True, a2: True}
        return (None, rew, term, {}, {})

    def close(self):
        pass


def _seq_env(winners):
    """An env_factory that hands back a fresh scripted env per battle, in order."""
    it = iter(list(winners))
    def factory():
        return _FakeEnv(next(it))
    return factory


def _pass_decide(*a, **k):
    # a pass/wait: the driver skips action_to_pe, so no real battle is touched.
    return None


def _dummy_bot(seed=None):
    # never invoked (decide bypasses policy_action); just a legal agent object.
    return object()


# ── aggregation ──────────────────────────────────────────────────────────────
def test_evaluate_aggregates_win_loss_draw():
    ref = Reference(name="random", make_agent=lambda s: object(), weight=1.0)
    result = evaluate(bot=_dummy_bot, refs=[ref], n_battles=3, fmt="gen9ou",
                      decide=_pass_decide, env_factory=_seq_env([1, 2, 0]))
    pr = result.per_ref["random"]
    assert (pr.wins, pr.losses, pr.draws) == (1, 1, 1)
    assert pr.n == 3
    assert pr.win_rate == pytest.approx(1 / 3)
    assert result.n_battles == 3
    assert result.format == "gen9ou"


def test_evaluate_all_wins_and_all_losses():
    ref = Reference(name="random", make_agent=lambda s: object(), weight=1.0)
    w = evaluate(bot=_dummy_bot, refs=[ref], n_battles=4,
                 decide=_pass_decide, env_factory=_seq_env([1, 1, 1, 1]))
    assert w.per_ref["random"].win_rate == 1.0
    l = evaluate(bot=_dummy_bot, refs=[ref], n_battles=4,
                 decide=_pass_decide, env_factory=_seq_env([2, 2, 2, 2]))
    assert l.per_ref["random"].losses == 4
    assert l.per_ref["random"].win_rate == 0.0


def test_evaluate_separates_references():
    r1 = Reference(name="floor", make_agent=lambda s: object(), weight=1.0)
    r2 = Reference(name="strong", make_agent=lambda s: object(), weight=2.0)
    # 2 vs floor (bot wins both), 2 vs strong (bot loses both)
    result = evaluate(bot=_dummy_bot, refs=Ladder([r1, r2]), n_battles=2,
                      decide=_pass_decide, env_factory=_seq_env([1, 1, 2, 2]))
    assert result.per_ref["floor"].wins == 2 and result.per_ref["floor"].losses == 0
    assert result.per_ref["strong"].losses == 2 and result.per_ref["strong"].wins == 0


def test_evaluate_tracks_steps():
    ref = Reference(name="random", make_agent=lambda s: object(), weight=1.0)
    result = evaluate(bot=_dummy_bot, refs=[ref], n_battles=2,
                      decide=_pass_decide, env_factory=_seq_env([1, 2]))
    # each fake battle runs exactly one step (terminal on the first step())
    assert result.per_ref["random"].steps == 2


# ── input validation ─────────────────────────────────────────────────────────
def test_evaluate_rejects_nonpositive_n_battles():
    ref = Reference(name="r", make_agent=lambda s: object())
    with pytest.raises(ValueError, match="n_battles"):
        evaluate(bot=_dummy_bot, refs=[ref], n_battles=0, env_factory=_seq_env([1]))


def test_evaluate_rejects_empty_refs():
    with pytest.raises(ValueError, match="at least one reference"):
        evaluate(bot=_dummy_bot, refs=[], n_battles=1, env_factory=_seq_env([1]))
    with pytest.raises(ValueError):
        evaluate(bot=_dummy_bot, refs=None, n_battles=1, env_factory=_seq_env([1]))


def test_evaluate_rejects_no_server_and_no_fake_env():
    ref = Reference(name="r", make_agent=lambda s: object())
    with pytest.raises(ValueError, match="pool|server_cfg|env_factory"):
        evaluate(bot=_dummy_bot, refs=[ref], n_battles=1)  # no way to play


def test_evaluate_real_battle_requires_vocabs():
    # Real battles (real decide path) build a real observation, which REQUIRES
    # vocabs. A missing vocabs must fail LOUD at the API boundary — not silently
    # play a blind zero-obs bot (the A3 self-check bug). A fake `decide` (test
    # injection) bypasses that path, so vocabs is only required when decide is
    # absent. (Checked AFTER bot validation, so a bad bot still reports bot.)
    ref = Reference(name="r", make_agent=lambda s: object())
    with pytest.raises(ValueError, match="vocabs"):
        evaluate(bot=_dummy_bot, refs=[ref], n_battles=1, server_cfg=object())
    # ...and the SAME shape with a fake decide (test injection) is fine — it
    # never touches build_obs, so vocabs may be absent (the hermetic path).
    ok = evaluate(bot=_dummy_bot, refs=[ref], n_battles=1,
                  env_factory=_seq_env([1]), decide=_pass_decide)
    assert ok.per_ref[ref.name].n == 1


def test_evaluate_rejects_unknown_bot():
    ref = Reference(name="r", make_agent=lambda s: object())
    with pytest.raises(TypeError, match="bot must be"):
        evaluate(bot=12345, refs=[ref], n_battles=1, env_factory=_seq_env([1]))


# ── the result is JSON-safe (the GUI/log contract) ───────────────────────────
def test_evalresult_is_json_serializable():
    import json

    ref = Reference(name="random", make_agent=lambda s: object(), weight=1.0)
    result = evaluate(bot=_dummy_bot, refs=[ref], n_battles=3,
                      decide=_pass_decide, env_factory=_seq_env([1, 2, 0]))
    blob = json.dumps(result.to_dict())  # must not raise
    d = json.loads(blob)
    assert d["per_ref"]["random"]["wins"] == 1
    # A4: one ref (weight 1.0) with a 1/3 win-rate -> fitness = 1/3.
    assert d["fitness_score"] == pytest.approx(1.0 / 3.0)


# ── live self-check (deselected by default: needs a real server + torch) ────
# Convention (matches control/pool/replays *_live tests): NO registered mark —
# the test NAME contains "live", so the default ``-k 'not live'`` filter
# deselects it. Run it explicitly: ``pytest fitness/tests -k live`` (starts a
# real server + builds real nets; CPU, ~30-60s).
def test_evaluate_bot_beats_random_live():
    # Live smoke: a REAL net vs a real random reference on a real server.
    # Deselected by the default -k "not live" filter (CPU/GPU discipline).
    # Run it explicitly: pytest fitness/tests -k live (starts a server, ~30-60s).
    from pool import ResourcePool, PoolConfig

    pool = ResourcePool(PoolConfig(n_servers=1, per_server=1, port_base=8901))
    try:
        pool.start(wait_healthy=True)
        from fitness.ladder import random_reference
        from net.vocab_db import build_vocabs

        vocabs = build_vocabs()
        # a random net as the "bot" (a legal but untrained policy) vs the
        # random reference — both weak, so it just needs to RUN to completion.
        # `vocabs` is REQUIRED for the real obs path (see net.env.build_obs).
        result = evaluate(
            bot=(lambda seed: random_reference(vocabs, device="cpu").agent(seed)),
            refs=[random_reference(vocabs, device="cpu")],
            n_battles=2, fmt="gen9bss3v3", pool=pool, max_steps=40,
            vocabs=vocabs)
        assert result.per_ref["random"].n == 2
    finally:
        pool.shutdown()


# ── A4: the weighted aggregate (pure — no battles needed) ──────────────────
def test_compute_fitness_is_weighted_average_of_win_rates():
    from fitness.harness import compute_fitness
    # beat the floor (w=1, 100%), lose to the strong rung (w=2, 0%):
    #   (1.0*1.0 + 2.0*0.0) / (1.0 + 2.0) = 1/3  -> mid-range, as the plan wants.
    f = compute_fitness({"random": 1.0, "heuristic": 0.0},
                        {"random": 1.0, "heuristic": 2.0})
    assert f == pytest.approx(1.0 / 3.0)


def test_compute_fitness_bounds_all_wins_and_all_losses():
    from fitness.harness import compute_fitness
    w = {"random": 1.0, "heuristic": 2.0}
    assert compute_fitness({"random": 1.0, "heuristic": 1.0}, w) == pytest.approx(1.0)
    assert compute_fitness({"random": 0.0, "heuristic": 0.0}, w) == pytest.approx(0.0)


def test_compute_fitness_accepts_perrefresult_objects():
    from fitness.harness import compute_fitness, PerRefResult
    pr = {"random": PerRefResult(ref="random", wins=3, losses=1),         # 0.75
          "heuristic": PerRefResult(ref="heuristic", wins=1, losses=3)}  # 0.25
    f = compute_fitness(pr, {"random": 1.0, "heuristic": 2.0})
    assert f == pytest.approx((1.0 * 0.75 + 2.0 * 0.25) / 3.0)


def test_compute_fitness_defaults_missing_weights_and_handles_empty():
    from fitness.harness import compute_fitness
    # a ref not in `weights` defaults to weight 1.0
    assert compute_fitness({"mystery": 0.5}) == pytest.approx(0.5)
    assert compute_fitness({}) == 0.0
    assert compute_fitness(None) == 0.0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
