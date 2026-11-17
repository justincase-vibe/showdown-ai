"""S4.0 gate runner — replay N real human battles and check the S4.0 gates.

Gates (PLAN.md §3c S4.0):
  * ``n_replayed`` battles through the adapter with **0 unknown-tag errors**;
  * ``legal_rate`` >= 0.95 (>=95% of decision points yield a legal action);
  * a corpus report (battle count, format mix, rating, **human switch-rate**);
  * the puller is deterministic + resumable (covered by the fixture test);
  * the 331/331 baseline stays green (a separate pytest run).

Default is OFFLINE: it loads the on-disk fixture ``bot/data/replays/{formatid}.jsonl``
(no network). ``use_network=True`` falls back to the HF puller when the fixture
is missing/short.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

from .pull import DEFAULT_OUT_DIR, load_local, pull_battles
from .report import corpus_report, format_report
from .trajectories import BattleTrace, replay_many


def run_gate(jsonl_path: Optional[str] = None, formatid: str = "gen9ou",
             n_battles: int = 200, min_rating: float = 1400.0,
             use_network: bool = False) -> Tuple[dict, List[BattleTrace]]:
    """Load N real human battles, replay them through the belief adapter, and
    return ``(report, traces)`` (and print the report)."""
    path = jsonl_path or os.path.join(DEFAULT_OUT_DIR, f"{formatid}.jsonl")
    if os.path.exists(path):
        battles = load_local(path, n_battles)
    elif use_network:
        battles = pull_battles(formatid, n_battles, min_rating, out_dir=DEFAULT_OUT_DIR)
    else:
        raise FileNotFoundError(
            f"no fixture at {path!r} — pass jsonl_path=... or use_network=True")

    traces = replay_many(battles)
    report = corpus_report(traces)
    print(format_report(report))
    return report, traces


def gate_passes(rep: dict, n_battles: int = 200,
                min_replayed_frac: float = 0.725) -> Tuple[bool, List[str]]:
    """Evaluate the S4.0 gate on a report dict. Returns (ok, [failures]).

    ``min_replayed_frac`` defaults to the USER-APPROVED D68 floor (72.5% —
    the 145/200 gate result): a non-trivial slice of human-ladder battles
    carries banned/Pokemon the fork skips (Gouging Fire, Volcarona, ...), so
    the replay-rate floor is deliberately below 100% — NOT a data-quality
    relaxation, just the approved acceptance bar for the corpus source.
    """
    fails = []
    if rep["unknown_count"] != 0:
        fails.append(f"unknown-tag errors = {rep['unknown_count']} (want 0) "
                     f"tags={rep['unknown_tags']}")
    if rep["n_decisions"] and rep["legal_rate"] < 0.95:
        fails.append(f"legal_rate = {rep['legal_rate']:.3f} (want >= 0.95)")
    if rep["n_replayed"] < n_battles * min_replayed_frac:
        fails.append(f"n_replayed = {rep['n_replayed']} (want >= {int(n_battles*min_replayed_frac)})")
    return (not fails), fails
