"""Fetch top-tier Pokemon Showdown battles from the HolidayOugi HF dataset.

The dataset (``HolidayOugi/pokemon-showdown-replays``) is one parquet file **per format**,
and the big formats are split into ``_part1..N`` files. Each row is a battle:

    id, format, players[], log, uploadtime, views, formatid, rating

``rating`` is the ELO (float, **null for unrated battles**). ``log`` is the standard
Showdown battle protocol (``|``-separated lines) -- the thing we parse into moves/switches.

We only need a **top-tier sample**, so we download just the *first* part-file of a format
(which holds far more battles than we'll keep) and keep the highest-ELO battles.

    # inspect one format (download + stats + one sample log, writes nothing)
    python bot/datamine/fetch.py --format gen8ru --inspect

    # fetch top-N battles for one format into bot/data/replays/{formatid}.jsonl
    python bot/datamine/fetch.py --format gen8ru --max 2000

    # fetch a batch
    python bot/datamine/fetch.py --format gen6ru,gen7ru,gen8ru,gen9ru --max 2000
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from typing import List, Optional
import urllib.parse

import pyarrow.parquet as pq
import requests

HF_ID = "HolidayOugi/pokemon-showdown-replays"
BASE_URL = f"https://huggingface.co/datasets/{HF_ID}/resolve/main"
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "replays")

# formatid -> dataset "display" base name (without .parquet). Matches the on-disk file names.
FORMAT_FILES = {
    "gen6ru": "[Gen 6] RU", "gen6uu": "[Gen 6] UU", "gen6ou": "[Gen 6] OU",
    "gen7ru": "[Gen 7] RU", "gen7uu": "[Gen 7] UU", "gen7ou": "[Gen 7] OU",
    "gen8ru": "[Gen 8] RU", "gen8uu": "[Gen 8] UU", "gen8ou": "[Gen 8] OU",
    "gen9ru": "[Gen 9] RU", "gen9uu": "[Gen 9] UU", "gen9ou": "[Gen 9] OU",
    "gen7randombattle": "[Gen 7] RANDOMBATTLE",
    "gen8randombattle": "[Gen 8] RANDOMBATTLE",
    "gen9randombattle": "[Gen 9] RANDOMBATTLE",
}


def _resolve_file_url(base: str) -> str:
    """Return the download URL for the first part-file of ``base`` (single file or _part1)."""
    candidates = [f"{base}.parquet", f"{base}_part1.parquet"]
    for name in candidates:
        url = BASE_URL + "/" + urllib.parse.quote(name)
        r = requests.head(url, allow_redirects=True, timeout=30)
        if r.status_code == 200:
            return url
    raise FileNotFoundError(f"no parquet file for format base {base!r}; tried {candidates}")


def fetch_top(formatid: str, max_battles: int, min_rating: Optional[float] = None) -> List[dict]:
    """Download the first part-file of ``formatid`` and return the top battles by ELO.

    Keeps battles with a non-null rating (optionally >= ``min_rating``), sorted descending,
    truncated to ``max_battles``. Each record: {id, formatid, format, rating, players, log}.
    """
    base = FORMAT_FILES[formatid]
    url = _resolve_file_url(base)
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False)
    tmp.close()
    try:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp.name, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)

        # Predicate pushdown: "rating > 0" selects rated battles (ELO is always > 0; unrated
        # are null) and prunes row groups where every rating is null.
        table = pq.read_table(tmp.name, filters=[("rating", ">", 0)])
        cols = {c: table.column(c).to_pylist() for c in
                ["id", "format", "formatid", "rating", "players", "log"]}
        n = len(cols["id"])
        recs = [
            {
                "id": cols["id"][i],
                "formatid": cols["formatid"][i],
                "format": cols["format"][i],
                "rating": float(cols["rating"][i]),
                "players": cols["players"][i],
                "log": cols["log"][i],
            }
            for i in range(n)
        ]
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    if min_rating is not None:
        recs = [r for r in recs if r["rating"] >= min_rating]
    recs.sort(key=lambda r: r["rating"], reverse=True)
    recs = recs[:max_battles]
    return recs


def _print_stats(formatid: str, recs: List[dict]) -> None:
    if not recs:
        print(f"[{formatid}] no rated battles found")
        return
    ratings = [r["rating"] for r in recs]
    ratings.sort()
    import statistics as st
    print(f"[{formatid}] kept {len(recs)} battles")
    print(f"    ELO  min={ratings[0]:.0f}  med={st.median(ratings):.0f}  max={ratings[-1]:.0f}")
    turns = [len([ln for ln in r['log'].splitlines() if ln.startswith('|turn|')]) for r in recs[:200]]
    print(f"    avg turns (first 200): {st.mean(turns):.1f}")
    print(f"    sample id: {recs[0]['id']}  rating={recs[0]['rating']:.0f}  players={recs[0]['players']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--format", required=True, help="formatid or comma list (e.g. gen8ru,gen7ou)")
    p.add_argument("--max", type=int, default=2000, help="max battles to keep per format")
    p.add_argument("--min-rating", type=float, default=None, help="optional ELO floor")
    p.add_argument("--out", default=DEFAULT_OUT, help="output dir for .jsonl files")
    p.add_argument("--inspect", action="store_true", help="download + print stats + one sample log")
    args = p.parse_args()

    fmts = [f.strip() for f in args.format.split(",") if f.strip()]
    for f in fmts:
        if f not in FORMAT_FILES:
            raise SystemExit(f"unknown formatid {f!r}. Known: {sorted(FORMAT_FILES)}")

    if args.inspect:
        f = fmts[0]
        recs = fetch_top(f, max(5, args.max), args.min_rating)
        _print_stats(f, recs)
        if recs:
            log = recs[0]["log"]
            print(f"\n    ---- sample battle log ({len(log)} chars) ----")
            for ln in log.splitlines():
                print("     |", ln)
        return

    os.makedirs(args.out, exist_ok=True)
    for f in fmts:
        recs = fetch_top(f, args.max, args.min_rating)
        out = os.path.join(args.out, f"{f}.jsonl")
        with open(out, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        _print_stats(f, recs)
        print(f"    wrote {out}")


if __name__ == "__main__":
    main()
