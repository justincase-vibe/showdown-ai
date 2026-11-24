"""B1 — ALL-FORMAT regression test (Claude small-addition #1).

Per GATE_CONTROLS_C1_C5.md / the user's rulings:
  (a) ``engine_key`` / ``engine_key_or_warn`` resolves for EVERY target format —
      no silent engine-key fallbacks (a fallback must be visible in logs), and
  (b) calc-vs-sim damage agreement on a few reference numbers PER FORMAT
      (``b1_calc_vs_sim_all_formats.js``): a real headless fork battle per
      scenario, actual damage checked against the @smogon/calc band using that
      format's engine key + level.

Ground truth = @smogon/calc (user ruling: the Smogon damage calc is the source of
truth; the fork is the thing being checked). The must-pass set is engine 9 (gen9,
incl. BSS 3v3) and 0 (champions) — the formats the AI actually plays. Old-gen
(engine 1-8) divergences are reported (WARN) without failing the suite, because
the calc library's per-gen data tables have known gaps (e.g. gen1 Karate Chop
lacks ``category`` and carries the wrong type) that are not patchable locally —
they stay visible in the output instead of hiding in the numbers.

Run:  python -m pytest tools/tests/test_b1_all_formats.py -q -s
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from legality import formats  # noqa: E402


def _node() -> str:
    exe = shutil.which("node")
    if not exe:
        raise AssertionError("node not found on PATH (B1 needs the fork runtime)")
    return exe


# ── (a) engine-key resolution: every format, no silent fallbacks ─────────

def test_engine_key_resolves_for_every_target_format():
    """Every registry format must resolve an engine key WITHOUT the fallback path."""
    assert len(formats.TARGET_FORMATS) == 50, f"expected 50 target formats, got {len(formats.TARGET_FORMATS)}"
    for f in formats.TARGET_FORMATS:
        key = formats.engine_key(f["id"])  # must not raise
        assert isinstance(key, int) and key >= 0, (f["id"], key)
        assert formats.engine_key_or_warn(f["id"]) == key, f["id"]


def test_engine_key_fallback_is_loud_not_silent(caplog):
    """The fallback path (unknown format id) must be visible: a logged WARNING."""
    with caplog.at_level(logging.WARNING, logger="legality.formats"):
        key = formats.engine_key_or_warn("gen9notarealformat")
    assert key == 9  # regex genN fallback semantics unchanged
    assert any("engine key fallback" in r.message for r in caplog.records), (
        "engine_key_or_warn fell back WITHOUT a warning log — silent fallbacks are "
        "exactly the bug class D32 (champions mapped to gen 9) was about")


def test_engine_key_champions_is_not_gen9():
    """D32: champions is the library's engine-0 mechanics set, NOT gen 9."""
    assert formats.engine_key("gen9championsou") == 0
    assert formats.engine_key("gen9championsvgc2026regmc") == 0
    assert formats.engine_key("gen9bss3v3") == 9  # BSS is vanilla gen 9
    assert formats.engine_key("gen9ou") == 9


def test_engine_key_old_gens_map_to_their_gen():
    for g in range(1, 9):
        assert formats.engine_key(f"gen{g}ou") == g
        assert formats.engine_key(f"gen{g}ubers") == g
    assert formats.engine_key("gen1lc") == 1
    assert formats.engine_key("gen8vgc2022") == 8


# ── (b) calc-vs-sim per format (node; real headless fork battles) ────────

def test_b1_calc_vs_sim_all_formats():
    """The full 50-format sweep. Must-pass set (gen9 + champions) must be all-PASS."""
    payload = [
        {
            "id": f["id"],
            "engineKey": formats.engine_key(f["id"]),
            "level": f["default_level"],
            "gameType": f["game_type"],
        }
        for f in formats.TARGET_FORMATS
    ]
    js = ROOT / "tools" / "tests" / "b1_calc_vs_sim_all_formats.js"
    proc = subprocess.run(
        [_node(), str(js), json.dumps(payload)],
        capture_output=True, text=True, timeout=1800, cwd=str(ROOT))
    out = proc.stdout or ""
    err = proc.stderr or ""
    # print() so the per-format matrix is visible with `-s` and on assertion output
    print(out)
    assert proc.returncode == 0, (
        "B1 calc-vs-sim failures (must-pass set = gen9 + champions)\n" + out + "\n" + err)
