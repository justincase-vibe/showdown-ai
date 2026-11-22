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

from tools.heuristic_ab import _gen_from_format, matrix  # noqa: E402


def test_matrix_pairs_in_order():
    # headline first (strategy vs heuristic), then the two floor checks.
    assert matrix() == [
        ("strategy", "heuristic"),
        ("strategy", "random"),
        ("heuristic", "random"),
    ]


def test_matrix_names_match_the_refs_dict():
    # run_ab builds refs keyed by exactly these names (strategy_reference ->
    # "strategy", heuristic_reference -> "heuristic", random_reference ->
    # "random"). A drift on either side would KeyError mid-run; pin the set.
    assert {n for pair in matrix() for n in pair} == \
        {"strategy", "heuristic", "random"}


def test_gen_from_format():
    assert _gen_from_format("gen9ou") == 9
    assert _gen_from_format("gen8bss3v3") == 8
    assert _gen_from_format("champions") == 9   # unrecognised -> primary
