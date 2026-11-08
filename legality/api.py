"""Public API for the legality DB (see PLAN.md).

Reads work immediately (empty until data is ingested). All data enters through
``ingest`` / ``ingest_file`` (D16 protocol — see ``ingest_protocol.md``).
Win-rate record/read/wipe is fully functional (mechanism, not data).
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from . import db, formats

# ── per-table primary keys (for upsert ON CONFLICT) ─────────────────
PRIMARY_KEYS: Dict[str, List[str]] = {
    "pokemon": ["id"],
    "form": ["id"],
    "move": ["id"],
    "move_gen": ["move_id", "gen"],
    "ability": ["id"],
    "item": ["id"],
    "form_type": ["form_id", "gen", "type"],
    "form_stats": ["form_id", "gen"],
    "form_tier": ["form_id", "gen"],
    "form_ability": ["form_id", "gen", "ability_id"],
    "form_learnset": ["form_id", "move_id", "gen"],
    "form_learnset_constraints": ["form_id", "move_id", "gen"],
    "type_chart": ["attacking", "defending", "gen"],
    "format": ["id"],
    "format_form": ["format_id", "form_id"],
    "format_form_ability": ["format_id", "form_id", "ability_id"],
    "format_form_move": ["format_id", "form_id", "move_id"],
    "format_item": ["format_id", "item_id"],
    "form_item": ["form_id", "item_id"],
    "strategy": ["id"],
    "strategy_move": ["set_id", "ord"],
    "win_form": ["format_id", "form_id"],
    "win_move": ["format_id", "form_id", "move_id"],
}

_INGESTABLE = frozenset(PRIMARY_KEYS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _require_format(fmt: str) -> None:
    try:
        formats.by_id(fmt)
    except KeyError as e:
        raise ValueError(
            f"format '{fmt}' is not in the registry — add it to legality/formats.py first"
        ) from e


# ═══════════════════════════════ status ════════════════════════════
def status() -> Dict[str, Any]:
    return db.status()


def formats_list(game_type: Optional[str] = None) -> List[str]:
    """Registered format ids (all, or filtered by 'singles'/'doubles')."""
    rows = [f["id"] for f in formats.TARGET_FORMATS
            if game_type is None or f["game_type"] == game_type]
    return rows


# ═══════════════════════════════ reads ═════════════════════════════
def legal_forms(fmt: str) -> List[str]:
    _require_format(fmt)
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT f.form_id FROM format_form f JOIN format x ON x.id=f.format_id "
            "WHERE f.format_id=? ORDER BY f.form_id", (fmt,)).fetchall()
        return [r["form_id"] for r in rows]
    finally:
        conn.close()


def legal_abilities(fmt: str, form_id: str) -> List[str]:
    _require_format(fmt)
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT ability_id FROM format_form_ability WHERE format_id=? AND form_id=? "
            "ORDER BY ability_id", (fmt, form_id)).fetchall()
        return [r["ability_id"] for r in rows]
    finally:
        conn.close()


def legal_moves(fmt: str, form_id: str) -> List[str]:
    """Legal moves for the form in this format (Fissure-rule applied at ingest)."""
    _require_format(fmt)
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT move_id FROM format_form_move WHERE format_id=? AND form_id=? "
            "ORDER BY move_id", (fmt, form_id)).fetchall()
        return [r["move_id"] for r in rows]
    finally:
        conn.close()


def tier(form_id: str, gen: int) -> Optional[str]:
    conn = db.connect()
    try:
        r = conn.execute(
            "SELECT tier FROM form_tier WHERE form_id=? AND gen=?", (form_id, gen)).fetchone()
        return r["tier"] if r else None
    finally:
        conn.close()


def stats(form_id: str, gen: int) -> Optional[Dict[str, int]]:
    conn = db.connect()
    try:
        r = conn.execute(
            "SELECT hp,atk,def,spa,spd,spe FROM form_stats WHERE form_id=? AND gen=?",
            (form_id, gen)).fetchone()
        if not r:
            return None
        return {k: r[k] for k in ("hp", "atk", "def", "spa", "spd", "spe")}
    finally:
        conn.close()


def types(form_id: str, gen: int) -> List[str]:
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT type FROM form_type WHERE form_id=? AND gen=? ORDER BY type",
            (form_id, gen)).fetchall()
        return [r["type"] for r in rows]
    finally:
        conn.close()


def moveset(form_id: str, gen: int) -> List[Dict[str, Any]]:
    """Full learnset for a form in a gen: [{move_id, sources}, ...] (Smogon moves page)."""
    conn = db.connect()
    try:
        rows = conn.execute(
            "SELECT move_id, sources FROM form_learnset WHERE form_id=? AND gen=? "
            "ORDER BY move_id", (form_id, gen)).fetchall()
        return [{"move_id": r["move_id"], "sources": r["sources"]} for r in rows]
    finally:
        conn.close()


def sets(fmt: str, form_id: str) -> List[Dict[str, Any]]:
    """Scraped strategies (Step 1) for a form in a format, moves in order.

    (table is `strategy` — `set` is a SQL keyword)"""
    _require_format(fmt)
    conn = db.connect()
    try:
        out = []
        for s in conn.execute(
                "SELECT * FROM strategy WHERE format_id=? AND form_id=? ORDER BY id",
                (fmt, form_id)).fetchall():
            d = dict(s)
            d["moves"] = [r["move_id"] for r in conn.execute(
                "SELECT move_id FROM strategy_move WHERE set_id=? ORDER BY ord", (s["id"],))]
            out.append(d)
        return out
    finally:
        conn.close()


# ═══════════════════════════════ ingest ════════════════════════════
def _table_columns(conn: sqlite3.Connection, table: str) -> List[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


def ingest(payload: Dict[str, Any], source: Optional[str] = None) -> Dict[str, int]:
    """Upsert a batch of rows in ONE transaction (D16).

    payload: {"source": "server|smogon|net", "note": "...",
              "rows": [{"table": "form", "data": {...}}, ...]}
    Returns per-table row counts. Raises on unknown table/column or FK violation.
    """
    if not isinstance(payload, dict) or "rows" not in payload:
        raise ValueError("payload must be a dict with a 'rows' list")
    source = source or payload.get("source") or "unknown"
    rows: List[Dict[str, Any]] = payload["rows"]

    # order children after parents (FK-safe)
    order = {t: i for i, t in enumerate(formats.TABLE_ORDER)}
    rows = sorted(rows, key=lambda r: order.get(r.get("table"), 999))

    conn = db.connect()
    counts: Dict[str, int] = {}
    try:
        cur = conn.cursor()
        for r in rows:
            table = r.get("table")
            data = r.get("data")
            if table not in _INGESTABLE:
                raise ValueError(f"ingest: unknown/protected table '{table}'")
            if not isinstance(data, dict) or not data:
                raise ValueError(f"ingest: row for '{table}' needs a non-empty 'data' dict")
            cols = _table_columns(conn, table)
            bad = [k for k in data if k not in cols]
            if bad:
                raise ValueError(f"ingest: unknown column(s) {bad} in table '{table}'")
            pk = PRIMARY_KEYS[table]
            missing = [k for k in pk if k not in data]
            if missing:
                raise ValueError(f"ingest: row for '{table}' missing pk column(s) {missing}")
            keys = list(data.keys())
            conflict = ",".join(pk)
            updates = ",".join(f"{k}=excluded.{k}" for k in keys if k not in pk)
            # Row carries only PK columns (e.g. membership tables) → nothing to
            # update; a bare `DO UPDATE SET` is invalid SQL ("incomplete input").
            clause = f"DO UPDATE SET {updates}" if updates else "DO NOTHING"
            sql = (f"INSERT INTO {table} ({','.join(keys)}) VALUES ({','.join('?' * len(keys))}) "
                   f"ON CONFLICT({conflict}) {clause}")
            cur.execute(sql, [data[k] for k in keys])
            counts[table] = counts.get(table, 0) + 1
        if rows:
            for t, n in counts.items():
                conn.execute(
                    "INSERT INTO ingest_log(ts, source, table_name, n_rows, note) "
                    "VALUES(?,?,?,?,?)",
                    (_now(), source, t, n, payload.get("note")))
        conn.commit()
        return counts
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ingest_file(path: str, source: Optional[str] = None) -> Dict[str, int]:
    with open(path, encoding="utf8") as fh:
        return ingest(json.load(fh), source=source)


# ═══════════════════════════════ win rates ═════════════════════════
def record_battle(fmt: str, winner: Sequence[str], loser: Sequence[str]) -> None:
    """Bump battles for all 12 forms, wins for the 6 winners (form ids)."""
    _require_format(fmt)
    if len(winner) != 6 or len(loser) != 6:
        raise ValueError("expected 6 form ids each for winner and loser")
    ts = _now()
    conn = db.connect()
    try:
        for side in (winner, loser):
            for form_id in side:
                conn.execute(
                    "INSERT INTO win_form(format_id, form_id, battles, wins, last_updated) "
                    "VALUES(?,?,1,0,?) "
                    "ON CONFLICT(format_id, form_id) DO UPDATE SET "
                    "battles=battles+1, last_updated=excluded.last_updated",
                    (fmt, form_id, ts))
        for form_id in winner:
            conn.execute(
                "UPDATE win_form SET wins=wins+1 WHERE format_id=? AND form_id=?",
                (fmt, form_id))
        conn.commit()
    finally:
        conn.close()


def win_rate(fmt: str, form_id: str) -> Optional[Dict[str, Any]]:
    conn = db.connect()
    try:
        r = conn.execute(
            "SELECT battles, wins, last_updated FROM win_form "
            "WHERE format_id=? AND form_id=?", (fmt, form_id)).fetchone()
        if not r or not r["battles"]:
            return None
        return {"battles": r["battles"], "wins": r["wins"],
                "rate": r["wins"] / r["battles"], "last_updated": r["last_updated"]}
    finally:
        conn.close()


def wipe_win_rates(fmt: Optional[str] = None) -> int:
    """DELETE win-rate rows (all formats, or one). Never touches T1 tables.

    Returns number of rows removed across win_form + win_move."""
    conn = db.connect()
    try:
        if fmt is None:
            n = conn.execute("DELETE FROM win_form").rowcount
            n += conn.execute("DELETE FROM win_move").rowcount
        else:
            _require_format(fmt)
            n = conn.execute("DELETE FROM win_form WHERE format_id=?", (fmt,)).rowcount
            n += conn.execute("DELETE FROM win_move WHERE format_id=?", (fmt,)).rowcount
        conn.execute("INSERT INTO ingest_log(ts, source, table_name, n_rows, note) "
                     "VALUES(?,?,?,?,?)", (_now(), "reset", "win_rates", n,
                                          f"wipe_win_rates({fmt})"))
        conn.commit()
        return n
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def wipe_table(table: str, where: Optional[Dict[str, Any]] = None) -> int:
    """Audit-logged DELETE on a data table (e.g. after a bad scrape batch).

    ``where`` = equality conditions (safe: parameterized, no raw SQL)."""
    if table in db.PROTECTED_TABLES:
        raise ValueError(f"table '{table}' is protected (provenance) — cannot be wiped")
    if table not in db.DATA_TABLES:
        raise ValueError(f"unknown data table '{table}'")
    params: List[Any] = []
    if where:
        for k, v in where.items():
            if k not in _table_columns(db.connect(), table):
                raise ValueError(f"unknown column '{k}' in '{table}'")
            params.append(v)
        clause = " WHERE " + " AND ".join(f"{k}=?" for k in where)
        params = list(where.values())
    else:
        clause = ""
    conn = db.connect()
    try:
        n = conn.execute(f"DELETE FROM {table}{clause}", params).rowcount
        conn.execute("INSERT INTO ingest_log(ts, source, table_name, n_rows, note) "
                     "VALUES(?,?,?,?,?)", (_now(), "reset", table, n,
                                          f"wipe_table({table}, {where})"))
        conn.commit()
        return n
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ═══════════════════════ team gen — stub (future step) ══════════════
def random_team(fmt: str, n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Legal random team (6 unique base species by default; n/format-size for VGC).

    NOT IMPLEMENTED YET — future work, requires populated T1 tables."""
    raise NotImplementedError("random_team is a future feature (needs populated legality data)")


if __name__ == "__main__":
    print(json.dumps(status(), indent=2))
