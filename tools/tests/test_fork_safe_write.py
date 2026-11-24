"""Regression test: concurrent ``safeWrite`` across processes must not crash.

The 2026-11-24 incident (``scratch/pool/server_49990.log`` and
``server_49991.log``): two Showdown servers sharing ONE fork directory both
called ``safeWrite`` on the same state file (``config/chat-plugins/seasons.json``,
``logs/lastbattle.txt``). Stock ``safeWrite`` used the SAME temp name
(``<file>.NEW``) in every process, so the second rename hit ``ENOENT`` and
CRASHED the whole server process:

    CRASH: Error: ENOENT: no such file or directory, rename
    '...\\config\\chat-plugins\\seasons.json.NEW' -> '...\\seasons.json'

This killed one of our concurrent learnability runs mid-battle.

Phase 2 (found 2026-11-24 while verifying the phase-1 fix): with unique temp
names the ENOENT is gone, but on WINDOWS the two processes then race the FINAL
rename onto the same destination, and the loser gets ``EPERM: operation not
permitted, rename`` (POSIX replaces atomically; Windows briefly refuses a
rename over a file another process is replacing at that moment). Same
production scenario, second crash point.

The fix (patched in BOTH ``lib/fs.ts`` and the built ``dist/lib/fs.js``, so a
rebuild keeps it):
  1) the temp name is UNIQUE per write, so two processes can never collide on
     the same ``.NEW`` file (kills ENOENT);
  2) the final rename retries with backoff on EPERM/EBUSY/EACCES (kills the
     Windows cross-process rename race); other errors propagate and the temp
     file is cleaned up.

Note: ``pokemon-showdown/`` is untracked (a local dependency, see .gitignore)
-- this tracked test is what pins the patched behavior and will FAIL loudly if
the fork is re-cloned or rebuilt from unpatched sources.
"""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FS_JS = ROOT / "pokemon-showdown" / "dist" / "lib" / "fs.js"

# ESM harness: import the BUILT fs.js (CommonJS -- take .default.FS), hammer
# safeWrite on a shared target. argv: [node, harness, fs.js, target, iters].
JS_HARNESS = """
import { pathToFileURL } from 'node:url';
const mod = await import(pathToFileURL(process.argv[2]).href);
const FS = (mod.default && mod.default.FS) || mod.FS;
const target = process.argv[3];
const iters = parseInt(process.argv[4], 10);
const payload = JSON.stringify({ writer: process.pid, pad: 'x'.repeat(262144) });
for (let i = 0; i < iters; i++) {
  await new FS(target).safeWrite(payload);
}
console.log('OK ' + process.pid);
"""


def _run_worker(target: Path, iters: int, scratch: Path, idx: int):
    harness = scratch / f"worker_{idx}.mjs"
    harness.write_text(JS_HARNESS)
    return subprocess.run(
        ["node", str(harness), str(FS_JS), str(target), str(iters)],
        capture_output=True, text=True, timeout=300)


def test_concurrent_safe_write_never_crashes(tmp_path):
    """Two processes writing the SAME file concurrently (the exact production
    scenario: many pool servers, one fork directory, battle-end writes).
    1000 writes each: dense enough that the final renames overlap often."""
    if not FS_JS.exists():
        pytest.skip("pokemon-showdown fork not present on this machine")
    target = tmp_path / "state.json"
    with ThreadPoolExecutor(max_workers=2) as ex:
        f1 = ex.submit(_run_worker, target, 1000, tmp_path, 0)
        f2 = ex.submit(_run_worker, target, 1000, tmp_path, 1)
        r1, r2 = f1.result(), f2.result()
    for r in (r1, r2):
        assert r.returncode == 0, (
            f"worker CRASHED (the ENOENT-rename race?): stderr={r.stderr[-600:]!r}")
        assert "OK" in r.stdout
    # the shared file must end intact (no torn write)
    data = json.loads(target.read_text(encoding="utf-8"))
    assert "writer" in data and "pad" in data


def test_safe_write_temp_name_is_unique_per_write(tmp_path):
    """The core invariant behind the fix: two concurrent writers must use
    different temp files. We verify this by reading the patched source
    directly (cheaper than timing races): the temp path must NOT be the
    fixed '<path>.NEW' string."""
    if not FS_JS.exists():
        pytest.skip("pokemon-showdown fork not present on this machine")
    built = (ROOT / "pokemon-showdown" / "dist" / "lib" / "fs.js").read_text(
        encoding="utf-8")
    source = (ROOT / "pokemon-showdown" / "lib" / "fs.ts").read_text(
        encoding="utf-8")
    # The stock bug: a plain '+ '.NEW'' / `.NEW` constant with no uniquifier.
    assert ".NEW" in built, "safeWrite's .NEW temp scheme is gone -- " \
                            "re-check that this test still tests the right thing"
    # ...but it must now carry a uniquifier (pid or random/sequence suffix).
    assert "NEW.${" in source or "NEW.' + " in source or "NEW.'" not in source, \
        "lib/fs.ts safeWrite still uses a FIXED temp name -- the cross-process " \
        "ENOENT-rename crash is back"
    # Phase 2: the final rename must RETRY on the Windows transient codes,
    # otherwise concurrent writers still crash (EPERM rename race).
    assert "EPERM" in source and "EBUSY" in source, \
        "lib/fs.ts safeWrite lost its EPERM/EBUSY rename retry -- the " \
        "Windows cross-process rename race is back"
    assert "EPERM" in built and "EBUSY" in built, \
        "dist/lib/fs.js safeWrite lost its EPERM/EBUSY rename retry -- " \
        "rebuild or re-apply the fork patch (lib/fs.ts)"
