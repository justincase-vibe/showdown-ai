"""Ingest ``bot/data/movesets.json`` (Smogon sets) into the strategy tables.

The source file is the standard Smogon set format, downloaded verbatim by
``bot/build_movesets.py`` from ``https://pkmn.github.io/smogon/data/sets/{fmt}.json``::

    { format: { Species: { "<Set Name>": {
          moves: [str, ...], item: str, ability?: str, nature: str,
          evs: {stat: int, ...}, ivs?: {...}, teratypes?: str|list
    }}})

We emit the standard ingest payload (``ingest_protocol.md`` §4.5) and hand it
to ``api.ingest`` — **one transaction per format**, so a bad set can only roll
back its own format's batch. Protocol rules honored:

* ``strategy.id`` = stable slug ``smogon-{fmt}-{form_id}-{n}`` (n = set index
  within the species in the file, stable across re-runs);
* ``source_url`` required (set to the actual data source URL);
* ``evs``/``ivs`` = canonical raw strings (``"252 Atk / 4 Def / 252 Spe"``);
* ``strategy_move.ord`` = 1..n in page order;
* ``scraped_at`` = mtime of the source file (ISO-8601 UTC).

Name→id resolution uses Showdown ``toID``. Legacy ``Hidden Power <type>``
variants all carry the server id ``hiddenpower`` (see extract_gen9.js), so
they map to the base id. A set whose move/item/ability cannot be resolved is
skipped with a warning (never silently half-written).

Usage (from the project root):
    python -m legality.ingest_movesets
    python -m legality.ingest_movesets --formats gen9ou,gen9uu,gen9ru
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

from . import api, db, formats

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "..", "bot", "data", "movesets.json")

#: canonical EV/IV display order + names (protocol §4.5 examples)
STAT_ORDER = ["hp", "atk", "def", "spa", "spd", "spe"]
STAT_NAME = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}


def to_id(name: str) -> str:
    """Showdown toID: lowercase alphanumerics only."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def evs_to_str(evs: Optional[Dict[str, int]]) -> Optional[str]:
    """{'spa': 252, 'def': 4, 'spe': 252} -> '252 SpA / 4 Def / 252 Spe'."""
    if not evs:
        return None
    parts = [f"{evs[s]} {STAT_NAME[s]}" for s in STAT_ORDER if evs.get(s)]
    return " / ".join(parts) if parts else None


def _id_table(conn, table: str) -> Set[str]:
    return {r["id"] for r in conn.execute(f"SELECT id FROM {table}")}


class Resolver:
    """Resolve display names to DB ids; None = unresolvable (set skipped)."""

    def __init__(self, conn: Any) -> None:
        self.moves = _id_table(conn, "move")
        self.items = _id_table(conn, "item")
        self.abilities = _id_table(conn, "ability")
        self.forms = _id_table(conn, "form")
        self.legal: Dict[str, Set[str]] = {}
        for r in conn.execute("SELECT format_id, form_id FROM format_form"):
            self.legal.setdefault(r["format_id"], set()).add(r["form_id"])

    def form_legal_in(self, fmt: str, form_id: str) -> bool:
        return form_id in self.legal.get(fmt, set())

    def move(self, name: str) -> Optional[str]:
        mid = to_id(name)
        if mid in self.moves:
            return mid
        # Hidden Power variants: the server assigns all of them id 'hiddenpower'
        if mid.startswith("hiddenpower") and "hiddenpower" in self.moves:
            return "hiddenpower"
        return None

    def item(self, name: str) -> Optional[str]:
        mid = to_id(name)
        return mid if mid in self.items else None

    def ability(self, name: str) -> Optional[str]:
        aid = to_id(name)
        return aid if aid in self.abilities else None

    def form(self, name: str) -> Optional[str]:
        fid = to_id(name)
        return fid if fid in self.forms else None


def build_payload(fmt: str, species_map: Dict[str, Dict], source_url: str,
                  scraped_at: str, res: Resolver) -> Dict[str, Any]:
    """One format -> standard ingest payload (strategy + strategy_move rows)."""
    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for sp_name, sets in species_map.items():
        form_id = res.form(sp_name)
        if form_id is None:
            skipped.append(f"{sp_name}: unknown form")
            continue
        # A set is only playable where the species is legal (validator verdict,
        # format_form). The Smogon set files are tier-independent and can list
        # species this server bans in the tier (e.g. Deoxys in gen9ou) — those
        # sets are unusable for team gen / belief, so skip them.
        if not res.form_legal_in(fmt, form_id):
            skipped.append(f"{sp_name}: {len(sets)} set(s) skipped — form not legal in {fmt}")
            continue
        for n, (title, s) in enumerate(sets.items(), start=1):
            move_ids = [res.move(m) for m in s.get("moves") or []]
            if any(m is None for m in move_ids):
                bad = [m for m in s.get("moves") or [] if res.move(m) is None]
                skipped.append(f"{sp_name} / {title}: unresolvable move(s) {bad}")
                continue
            item_id = res.item(s["item"]) if s.get("item") else None
            if s.get("item") and item_id is None:
                skipped.append(f"{sp_name} / {title}: unknown item '{s['item']}'")
                continue
            ability_id = res.ability(s["ability"]) if s.get("ability") else None
            if s.get("ability") and ability_id is None:
                skipped.append(f"{sp_name} / {title}: unknown ability '{s['ability']}'")
                continue
            tera = s.get("teratypes")
            tera_type = to_id(tera[0]) if isinstance(tera, list) and tera else (
                to_id(tera) if isinstance(tera, str) else None)
            sid = f"smogon-{fmt}-{form_id}-{n}"
            rows.append({"table": "strategy", "data": {
                "id": sid, "format_id": fmt, "form_id": form_id,
                "title": title, "source": "smogon", "source_url": source_url,
                "scraped_at": scraped_at,
                "item_id": item_id, "ability_id": ability_id,
                "nature": s.get("nature"),
                "evs": evs_to_str(s.get("evs")), "ivs": evs_to_str(s.get("ivs")),
                "tera_type": tera_type,
            }})
            for ord_, mid in enumerate(move_ids, start=1):
                rows.append({"table": "strategy_move",
                             "data": {"set_id": sid, "ord": ord_, "move_id": mid}})
    return {"source": "smogon",
            "note": f"Smogon sets — {fmt} ({len(species_map)} species)",
            "rows": rows, "_skipped": skipped}


def ingest_movesets(data_path: str, fmts: Optional[List[str]] = None) -> int:
    data_path = os.path.normpath(data_path)
    with open(data_path, encoding="utf8") as fh:
        ms = json.load(fh)
    scraped_at = datetime.fromtimestamp(os.path.getmtime(data_path), tz=timezone.utc) \
        .isoformat(timespec="seconds")

    fmts = fmts or [f for f in ms if f in set(formats.format_ids())]
    skipped_formats = [f for f in ms if f not in set(formats.format_ids())]
    if skipped_formats:
        print(f"note: skipping format(s) not in the registry: {skipped_formats}")

    conn = db.connect()
    try:
        res = Resolver(conn)
    finally:
        conn.close()

    total_sets = 0
    failures = 0
    for fmt in fmts:
        payload = build_payload(
            fmt, ms[fmt],
            source_url=f"https://pkmn.github.io/smogon/data/sets/{fmt}.json",
            scraped_at=scraped_at, res=res)
        skipped = payload.pop("_skipped")
        for msg in skipped:
            print(f"  skip {fmt}: {msg}")
        n_sets = sum(1 for r in payload["rows"] if r["table"] == "strategy")
        try:
            counts = api.ingest(payload, source="smogon")
        except Exception as e:
            failures += 1
            print(f"  FAIL {fmt}: {type(e).__name__}: {e}")
            continue
        total_sets += n_sets
        print(f"  {fmt}: {n_sets} sets -> "
              f"{counts.get('strategy', 0)} strategy + "
              f"{counts.get('strategy_move', 0)} strategy_move rows")
    print(f"done: {total_sets} sets ingested across {len(fmts) - failures} format(s)")
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=DEFAULT_DATA,
                    help="path to movesets.json (default: bot/data/movesets.json)")
    ap.add_argument("--formats", default=None,
                    help="comma-separated format ids (default: all in the registry)")
    args = ap.parse_args(argv)
    fmts = [f.strip() for f in args.formats.split(",")] if args.formats else None
    return ingest_movesets(args.data, fmts)


if __name__ == "__main__":
    sys.exit(main())
