"""Corpus trace cache — replay battles ONCE, save the ``BattleTrace``s, reload
for free on every subsequent train / split / gate run.

This is the enabler for scaling the BC warm-start (N=50 -> thousands) without
re-paying the ~tens-of-seconds/battle replay cost (the belief stack's
ability-posterior node spawns, ``belief/evidence.py``) on every run.

**Non-invasive by design:** it does NOT change the belief stack, the calc, the
encoder, or the net. It only memoizes ``replay_battle``'s OUTPUT
(``BattleTrace.to_dict()``) to a JSONL file. The cached traces are
bit-for-bit the same objects ``replay_many`` would produce, so everything
downstream (``build_samples`` / ``encode_context`` / the net) is unchanged.

**Cache invalidation (loud, not silent):** a fingerprint of the code that
PRODUCES the traces (belief + encoder + registry + calc) plus the legality DB
(size + mtime). If any of those change, ``load_cache`` refuses the stale cache
with a clear error. Rebuild it to refresh.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from multiprocessing import Pool
from typing import List

from corpus.trajectories import BattleTrace, replay_battle
from legality import db as _db

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root

#: the code whose behaviour the cached traces depend on. Any edit to one of
#: these (or a DB change) changes the fingerprint -> the cache is stale.
_FINGERPRINT_FILES = [
    "belief/state.py", "belief/evidence.py", "belief/log_events.py",
    "belief/events.py", "net/encoder.py", "net/registry.py",
    "damage/calc.py", "damage/node_calc.js",
]


def fingerprint() -> str:
    """A stable content key for the trace-producing code + the legality DB."""
    h = hashlib.sha256()
    for rel in _FINGERPRINT_FILES:
        p = os.path.join(ROOT, rel)
        with open(p, "rb") as fh:
            h.update(rel.encode("utf-8"))
            h.update(fh.read())
    try:
        st = os.stat(_db.DB_PATH)
        h.update(f"db:{os.path.basename(_db.DB_PATH)}:{st.st_size}:{int(st.st_mtime)}"
                 .encode("utf-8"))
    except OSError:
        pass
    return h.hexdigest()[:20]


def _meta_path(cache_path: str) -> str:
    return cache_path + ".meta.json"


def _stored_fp(cache_path: str) -> str | None:
    mp = _meta_path(cache_path)
    if os.path.exists(mp):
        with open(mp, "r", encoding="utf-8") as fh:
            return json.load(fh).get("fingerprint")
    return None


def load_cache(cache_path: str, check_fp: bool = True) -> List[BattleTrace]:
    """Read a trace cache (one ``BattleTrace`` per JSONL line).

    If ``check_fp`` and the stored fingerprint does not match the current
    code/DB, raise (a stale cache must never be used silently — it would train
    the net on belief-state snapshots that no longer match the live stack)."""
    if check_fp:
        stored = _stored_fp(cache_path)
        cur = fingerprint()
        if stored != cur:
            raise ValueError(
                f"trace cache {os.path.basename(cache_path)} is STALE "
                f"(stored fp {stored} != current {cur}). The belief/encoder/"
                f"calc code or the legality DB changed since it was built. "
                f"Rebuild it with corpus.cache.build_cache (or delete the file).")
    traces: List[BattleTrace] = []
    with open(cache_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                traces.append(BattleTrace.from_dict(json.loads(line)))
    return traces


def build_cache(battles: List[dict], cache_path: str, workers: int = 4,
                force: bool = False) -> List[BattleTrace]:
    """Replay ``battles`` through the belief stack (in parallel) and cache the
    ``BattleTrace``s to ``cache_path`` (JSONL, one per line).

    * **Resumable:** battles already present (by battle id) are skipped, so an
      interrupted build continues where it left off.
    * **Stale-safe:** if the cache exists but its fingerprint != the current
      code/DB, it is wiped and rebuilt (``force=True`` also rebuilds).
    * Returns the traces in ``battles`` order (a stable, de-duplicated set).
    """
    cur_fp = fingerprint()
    if os.path.exists(cache_path) and not force:
        stored = _stored_fp(cache_path)
        if stored is not None and stored != cur_fp:
            print(f"[cache] stale fingerprint ({stored} != {cur_fp}) -> rebuilding",
                  flush=True)
            os.remove(cache_path)
            force = True

    keep_ids = {str(b.get("id", "")) for b in battles}
    existing_ids: set = set()
    if os.path.exists(cache_path) and not force:
        existing_ids = {t.battle_id for t in load_cache(cache_path, check_fp=False)
                        if t.battle_id in keep_ids}

    todo = [b for b in battles if str(b.get("id", "")) not in existing_ids]
    if todo:
        t0 = time.time()
        if workers and workers > 1:
            done = 0
            with open(cache_path, "a", encoding="utf-8") as fh, \
                    Pool(workers) as pool:
                for tr in pool.imap_unordered(replay_battle, todo, chunksize=2):
                    if tr.battle_id in keep_ids:
                        fh.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
                    done += 1
                    if done % 25 == 0 or done == len(todo):
                        el = time.time() - t0
                        print(f"[cache] {done}/{len(todo)} "
                              f"({done/max(1e-9, el):.1f}/s, {el:.0f}s)", flush=True)
        else:
            with open(cache_path, "a", encoding="utf-8") as fh:
                for b in todo:
                    tr = replay_battle(b)
                    if tr.battle_id in keep_ids:
                        fh.write(json.dumps(tr.to_dict(), ensure_ascii=False) + "\n")
        print(f"[cache] replayed {len(todo)} battles "
              f"({(time.time()-t0)/max(1,len(todo)):.1f}s/battle, {workers} workers)",
              flush=True)

    # de-duplicate + write in a stable (battles) order; keep the meta fingerprint.
    by_id: dict = {}
    for t in load_cache(cache_path, check_fp=False):
        if t.battle_id in keep_ids:
            by_id[t.battle_id] = t
    ordered = [by_id[str(b.get("id", ""))] for b in battles
               if str(b.get("id", "")) in by_id]
    with open(cache_path, "w", encoding="utf-8") as fh:
        for t in ordered:
            fh.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    n_replayed = sum(1 for t in ordered if t.skipped is None)
    with open(_meta_path(cache_path), "w", encoding="utf-8") as fh:
        json.dump({
            "fingerprint": cur_fp,
            "n_battles": len(ordered), "n_replayed": n_replayed,
            "n_skipped": len(ordered) - n_replayed,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "workers": workers,
        }, fh, indent=2)
    return ordered
