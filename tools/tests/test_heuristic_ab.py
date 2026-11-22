"""Step 6.6 — the heuristic A/B tool: hermetic pins (no server, no torch).

The LIVE run (``run_ab``) needs a server pool — that is the 6.6 validation
itself, run manually. These pin the PURE parts: the matrix (the pair names
must match ``run_ab``'s ``refs`` dict exactly — a typo there would KeyError
mid-run) and the gen helper.

Run:  python -m pytest tools/tests/test_heuristic_ab.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.heuristic_ab import _gen_from_format, matrix, parse_only  # noqa: E402


def test_matrix_pairs_in_order():
    # headline first (strategy vs heuristic), then the two floor checks, then
    # the 6.6a baseline (strategy_posterior vs heuristic).
    assert matrix() == [
        ("strategy", "heuristic"),
        ("strategy", "random"),
        ("heuristic", "random"),
        ("strategy_posterior", "heuristic"),
    ]


def test_matrix_names_match_the_refs_dict():
    # run_ab builds refs keyed by exactly these names (strategy_reference ->
    # "strategy", strategy_posterior_reference -> "strategy_posterior",
    # heuristic_reference -> "heuristic", random_reference -> "random"). A
    # drift on either side would KeyError mid-run; pin the set.
    assert {n for pair in matrix() for n in pair} == \
        {"strategy", "strategy_posterior", "heuristic", "random"}


def test_parse_only_none_and_empty():
    # no --only flag -> full matrix (pairs=None is the run_ab default).
    assert parse_only(None) is None
    assert parse_only("") is None


def test_parse_only_valid_pair():
    # the 6.6a baseline flag: single pair, whitespace-tolerant, order preserved.
    assert parse_only("strategy_posterior,heuristic") == \
        [("strategy_posterior", "heuristic")]
    assert parse_only(" strategy , random ") == [("strategy", "random")]


def test_parse_only_rejects_unknown_pair():
    # a typo must fail loudly BEFORE any battle is started (not mid-run).
    import pytest
    with pytest.raises(ValueError):
        parse_only("strategy_posterior, random_init")


def test_gen_from_format():
    assert _gen_from_format("gen9ou") == 9
    assert _gen_from_format("gen8bss3v3") == 8
    assert _gen_from_format("gen9championsou") == 0   # D32: champions engine
    assert _gen_from_format("champions") == 9          # unregistered -> 9
