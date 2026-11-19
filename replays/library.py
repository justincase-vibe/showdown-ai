"""The self-play REPLAY LIBRARY — save + index + fetch (Phase 2 core).

Storage (all under one root, default ``<project>/runs/replays``, gitignored):

    {root}/{formatid}.jsonl   one RECORD per line — the BC fixture shape
                              (see ``replays/record.py``); append-only.
    {root}/index.json         fast index: one metadata entry per battle,
                              WITH BYTE OFFSETS so ``fetch_last``/``fetch_top``
                              read exactly N records — O(N) in N, never in
                              file size (10k battles x ~100KB stays fast).

Design (long-term):
  * **Thread-safe by one lock** — the P4/P5 trainers save from N battle threads
    concurrently; append + index update are atomic under it.
  * **Dedup by battle id** — a battle tag is unique; re-saving the same battle
    is a no-op (returns the existing path), so a crashed+retried run can never
    double-count a battle in the library or the index.
  * **Fetch is pure read** — never mutates; safe to call from the GUI while a
    training run is still appending.
  * **No pruning policy yet** — open decision #4 (save-all vs top-ELO-only +
    disk budget) is the user's call; the index makes a future prune cheap
    (delete rows + rewrite, or keep an ``active`` flag — decided then).
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = str(PROJECT_ROOT / "runs" / "replays")

#: index entry keys that are metadata (not read from the record)
_META_KEYS = ("ts", "formatid", "rating", "elo", "winner_side", "bots")


class ReplayLibrary:
    def __init__(self, root: Optional[str] = None):
        self.root = Path(root or DEFAULT_ROOT)
        self.index_path = self.root / "index.json"
        self._lock = threading.Lock()
        self._index: List[Dict[str, Any]] = self._load_index()

    # ── paths ──────────────────────────────────────────────────────────────
    def path(self, formatid: str) -> Path:
        return self.root / f"{formatid}.jsonl"

    # ── save ───────────────────────────────────────────────────────────────
    def save(self, record: dict) -> Dict[str, Any]:
        """Append ONE record (dedup by id). Returns ``{"path", "offset",
        "size", "deduped"}`` — the offset/size of the record's line."""
        fid = str(record.get("formatid") or "unknown")
        rid = str(record.get("id") or "")
        if not rid:
            raise ValueError("ReplayLibrary.save: record has no 'id'")
        line = (json.dumps(record, ensure_ascii=False) + "\n").encode("utf-8")
        path = self.path(fid)
        with self._lock:
            if any(e["id"] == rid for e in self._index):
                return {"path": str(path), "offset": -1, "size": len(line),
                        "deduped": True}
            self.root.mkdir(parents=True, exist_ok=True)
            with open(path, "ab") as fh:
                fh.seek(0, os.SEEK_END)
                offset = fh.tell()
                fh.write(line)
            self._index.append(self._index_entry(rid, offset, len(line), fid, record))
            self._write_index()
        return {"path": str(path), "offset": offset, "size": len(line),
                "deduped": False}

    # ── fetch (pure reads) ─────────────────────────────────────────────────
    def fetch_last(self, n: int, formatid: Optional[str] = None) -> List[dict]:
        """The LAST ``n`` records (insertion order, most recent last), optionally
        one format. Returns FULL records (with ``log``)."""
        n = max(0, int(n))
        if n == 0:
            return []
        entries = self._select(formatid=formatid)
        return [self._read_record(e) for e in entries[-n:]]

    def fetch_top(self, n: int, key: str = "elo",
                  formatid: Optional[str] = None) -> List[dict]:
        """The TOP ``n`` records by ``key`` desc (``"elo"`` now, ``"rating"``
        for the P4 era). Records missing the key sink to the bottom (``-inf``)
        instead of crashing — a partial library (some battles with ELO, some
        without) still returns the best it has. Stable: ties keep insertion
        order (most recent first)."""
        if key not in ("elo", "rating", "ts"):
            raise ValueError(f"fetch_top: unknown key {key!r} (use 'elo', 'rating' or 'ts')")
        n = max(0, int(n))
        entries = self._select(formatid=formatid)
        # Split by presence of the key so a str key (ts) is never compared with a
        # float (or with the None placeholder) — missing-key entries sink.
        have = [(i, e) for i, e in enumerate(entries) if e.get(key) is not None]
        missing = [e for e in entries if e.get(key) is None]

        def rank(ie):
            v = ie[1][key]
            if isinstance(v, str):                 # ISO-8601 sorts = chronological
                return (v, -ie[0])
            return (-float(v), -ie[0])             # ties: MOST RECENT first (ie[0] desc)

        ranked = [ie[1] for ie in sorted(have, key=rank)[:n]]
        ranked += missing[: max(0, n - len(ranked))]
        return [self._read_record(e) for e in ranked]

    def count(self, formatid: Optional[str] = None) -> int:
        return len(self._select(formatid=formatid))

    def formats(self) -> List[str]:
        seen: List[str] = []
        for e in self._index:
            if e["formatid"] not in seen:
                seen.append(e["formatid"])
        return seen

    def summary(self) -> Dict[str, Any]:
        """A small dict for the GUI: counts, top-ELO preview, disk size."""
        per_fmt: Dict[str, int] = {}
        for e in self._index:
            per_fmt[e["formatid"]] = per_fmt.get(e["formatid"], 0) + 1
        top = self.fetch_top(5, key="elo")
        return {
            "root": str(self.root),
            "total": len(self._index),
            "per_format": per_fmt,
            "formats": self.formats(),
            "top5_by_elo": [{"id": r["id"], "elo": r.get("elo"),
                             "rating": r.get("rating"), "ts": r.get("ts")}
                            for r in top],
        }

    # ── index maintenance ──────────────────────────────────────────────────
    def rebuild_index(self) -> int:
        """Rebuild ``index.json`` from the JSONL files (the index is a cache —
        the JSONL is the truth). Use after manual edits / a corrupt index.
        Returns the new entry count."""
        entries = self._scan_files()
        with self._lock:
            self._index = entries
            self._write_index()
        return len(entries)

    @staticmethod
    def _index_entry(rid: str, offset: int, size: int, fid: str, rec: dict) -> Dict[str, Any]:
        entry: Dict[str, Any] = {"id": rid, "offset": offset, "size": size,
                                "formatid": fid, "ts": rec.get("ts")}
        for k in ("rating", "elo", "winner_side", "bots"):
            entry[k] = rec.get(k)
        return entry

    def _scan_files(self) -> List[Dict[str, Any]]:
        """The one place the JSONL files are scanned into index entries
        (rebuild + cold start both use it — no duplicated scan logic)."""
        entries: List[Dict[str, Any]] = []
        for fid in self.formats_on_disk():
            path = self.path(fid)
            with open(path, "rb") as fh:
                data = fh.read()
            offset = 0
            for line in data.split(b"\n"):
                if not line.strip():
                    offset += 1
                    continue
                rec = json.loads(line.decode("utf-8", errors="replace"))
                entries.append(self._index_entry(
                    str(rec.get("id", f"{fid}-{offset}")), offset, len(line) + 1, fid, rec))
                offset += len(line) + 1
        return entries

    def formats_on_disk(self) -> List[str]:
        if not self.root.exists():
            return []
        out = []
        for p in sorted(self.root.glob("*.jsonl")):
            out.append(p.stem)
        return out

    # ── internals ──────────────────────────────────────────────────────────
    def _select(self, formatid: Optional[str] = None) -> List[Dict[str, Any]]:
        if formatid is None:
            return list(self._index)
        return [e for e in self._index if e["formatid"] == formatid]

    def _read_record(self, entry: Dict[str, Any]) -> dict:
        path = self.path(entry["formatid"])
        with open(path, "rb") as fh:
            fh.seek(entry["offset"])
            raw = fh.read(entry["size"])
        return json.loads(raw.decode("utf-8", errors="replace").rstrip("\n"))

    def _load_index(self) -> List[Dict[str, Any]]:
        if not self.index_path.exists():
            # no index yet — if there ARE jsonl files (a pre-existing library),
            # rebuild from them; otherwise start empty.
            if self.formats_on_disk():
                entries = self._scan_files()
                self.index_path.parent.mkdir(parents=True, exist_ok=True)
                self._write_index_locked(entries)
                return entries
            return []
        try:
            with open(self.index_path, "r", encoding="utf-8") as fh:
                idx = json.load(fh)
            return idx if isinstance(idx, list) else []
        except Exception:
            # corrupt index: rebuild from the (authoritative) JSONL
            self.rebuild_index()
            return list(self._index)

    def _write_index(self) -> None:
        assert self._lock.locked(), "_write_index must be called under the lock"
        self._write_index_locked(self._index)

    def _write_index_locked(self, entries: List[Dict[str, Any]]) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.index_path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entries, fh, ensure_ascii=False)
        os.replace(tmp, self.index_path)   # atomic on Windows+POSIX
