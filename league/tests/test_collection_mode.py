"""Review Step 1 (2026-11-22) — the league must COLLECT sampled decisions in
learning mode and stay argmax for eval/ELO.

Pins the fix for the policy-collapse bug: the league used to collect with a
single ``greedy=True`` default, so PPO trained on argmax-collected data (data
the policy would not have produced by sampling) -> collapse onto the strongest
move, no switches, no hazards. Now:

  * ``greedy_eval``  (default True)  — argmax for ELO/eval measurement;
  * ``greedy_train`` (default False) — learning-mode collection SAMPLES;
  * ``collection_greedy(cfg)``       — the ONE place that maps mode -> flag,
    used by BOTH battle factories (eval + learning).
"""
from __future__ import annotations


def test_config_defaults_eval_argmax_train_sample():
    from league.engine import LeagueConfig
    cfg = LeagueConfig()
    assert cfg.greedy_eval is True, "eval/ELO measurement must stay deterministic"
    assert cfg.greedy_train is False, "learning-mode collection must SAMPLE"


def test_collection_mode_follows_the_learning_flag():
    from league.engine import LeagueConfig, collection_greedy
    assert collection_greedy(LeagueConfig(learning=True)) is False, \
        "learning mode must collect sampled (stochastic) decisions"
    assert collection_greedy(LeagueConfig(learning=False)) is True, \
        "eval-only mode must stay argmax (deterministic ELO)"
    # explicit user overrides still win (a user may force argmax training or
    # sampled eval — the mapping never ignores the flags)
    assert collection_greedy(LeagueConfig(learning=True, greedy_train=True)) is True
    assert collection_greedy(LeagueConfig(learning=False, greedy_eval=False)) is False


def test_config_roundtrip_carries_both_flags():
    from league.engine import LeagueConfig
    cfg = LeagueConfig(greedy_eval=False, greedy_train=True, learning=True)
    same = LeagueConfig.from_dict(cfg.to_dict())
    assert (same.greedy_eval, same.greedy_train) == (False, True)


def test_legacy_greedy_key_is_ignored_not_fatal():
    """Old GUI payloads / saved configs carry the pre-split ``greedy`` key;
    ``from_dict`` must drop it (unknown-key tolerance) instead of crashing,
    and the NEW defaults must apply (eval argmax / train sample)."""
    from league.engine import LeagueConfig
    cfg = LeagueConfig.from_dict({"greedy": True, "learning": True})
    assert not hasattr(cfg, "greedy")
    assert cfg.greedy_eval is True and cfg.greedy_train is False
