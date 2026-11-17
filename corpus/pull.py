"""S4.0 pull layer — reuse ``bot.datamine.fetch.fetch_top`` (don't rewrite).

The network puller is deterministic and resumable:
  * deterministic — ``fetch_top`` returns battles sorted by rating desc, and we
    write the same records to the same path on every run;
  * resumable     — if the output ``{formatid}.jsonl`` already holds >=
    ``max_battles`` records (after the ``min_rating`` floor) we read it back and
    do NOT re-hit the network (a cache hit);
  * testable      — the fetcher is injectable, so the fixture test pins the
    determinism + cache behaviour with no network.

Only ``fetch_top`` is reused; the S4.0-specific bits (resumable cache + loader)
are new and small.
"""
from __future__ import annotations

import json
import os
from typing import Callable, List, Optional

from bot.datamine import fetch as _fetch

#: bot/data/replays — the on-disk fixtures (and where new pulls are cached).
#: Reuse the fetcher's own default so the corpus never drifts from where the
#: bot already keeps its replay fixtures.
DEFAULT_OUT_DIR = _fetch.DEFAULT_OUT

#: (formatid, max_battles, min_rating) -> List[dict]   (fetch_top's signature)
Fetcher = Callable[[str, int, Optional[float]], List[dict]]


def load_local(jsonl_path: str, max_battles: Optional[int] = None) -> List[dict]:
    """Read a battles ``.jsonl`` (one JSON object per line). Deterministic order
    = file order. ``max_battles`` truncates to the puller's top-N contract."""
    recs: List[dict] = []
    # errors="replace": the HF fixtures occasionally carry a stray non-UTF8 byte in
    # a player name / chat line — a hard decode error would abort the whole load,
    # so we replace it (the belief adapter ignores lobby/chat lines anyway).
    with open(jsonl_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    if max_battles is not None:
        recs = recs[:max_battles]
    return recs


def _write_jsonl(recs: List[dict], path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def pull_battles(formatid: str, max_battles: int = 200, min_rating: float = 1400.0,
                 out_dir: Optional[str] = None,
                 fetcher: Optional[Fetcher] = None) -> List[dict]:
    """Return the top ``max_battles`` rated battles for ``formatid`` (>= ``min_rating``).

    ``out_dir`` enables the resumable cache: if ``{out_dir}/{formatid}.jsonl``
    already holds >= ``max_battles`` records (after the ``min_rating`` floor) we
    read it back (no network). ``fetcher`` is injectable for offline/fixture
    tests (same signature as ``fetch_top``); defaults to the real network puller.
    """
    fetcher = fetcher or _fetch.fetch_top
    if out_dir is None:
        return fetcher(formatid, max_battles, min_rating)[:max_battles]

    path = os.path.join(out_dir, f"{formatid}.jsonl")
    if os.path.exists(path):
        cached = load_local(path)
        if min_rating is not None:
            cached = [r for r in cached if (r.get("rating") or 0) >= min_rating]
        if len(cached) >= max_battles:
            return cached[:max_battles]  # cache hit — no network

    recs = fetcher(formatid, max_battles, min_rating)
    _write_jsonl(recs, path)
    return recs[:max_battles]
