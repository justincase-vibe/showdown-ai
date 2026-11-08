"""Database access + bootstrap for the legality DB.

The DB file is generated (git-ignored). Bootstrap is idempotent: it creates the
schema (from ``schema.sql``), upserts the 50-format registry, and stamps ``meta``.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

from . import formats

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(HERE, "schema.sql")
DB_PATH = os.environ.get("LEGALITY_DB", os.path.join(HERE, "legality.db"))

SCHEMA_VERSION = 3
#: tables that must never be wiped (provenance + schema bookkeeping)
PROTECTED_TABLES = frozenset({"meta", "ingest_log"})
#: tables with data (for status + wipe whitelist)
DATA_TABLES: List[str] = [
    "pokemon", "form", "move", "move_gen", "ability", "item",
    "form_type", "form_stats", "form_tier", "form_ability",
    "form_learnset", "form_learnset_constraints",
    "type_chart",
    "format",
    "format_form", "format_form_ability", "format_form_move",
    "format_item", "form_item",
    "strategy", "strategy_move",
    "win_form", "win_move",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def bootstrap(db_path: str = DB_PATH, force: bool = False) -> Dict[str, Any]:
    """Create schema + registry + meta. Idempotent (upserts the registry)."""
    if force and os.path.exists(db_path):
        os.remove(db_path)
    with open(SCHEMA_PATH, encoding="utf8") as fh:
        ddl = fh.read()

    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(ddl)  # PRAGMA foreign_keys ignored in executescript; fine here
        conn.execute("PRAGMA foreign_keys = ON")

        # upsert format registry
        cols = ["id", "name", "gen", "tier", "game_type", "min_team", "max_team",
                "default_level", "section", "mod", "exists_in_server"]
        for f in formats.TARGET_FORMATS:
            conn.execute(
                f"INSERT INTO format ({','.join(cols)}) VALUES ({','.join('?' * len(cols))}) "
                f"ON CONFLICT(id) DO UPDATE SET "
                + ",".join(f"{c}=excluded.{c}" for c in cols if c != "id"),
                [f[c] for c in cols],
            )

        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        conn.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("bootstrapped_at", _now()),
        )
        conn.commit()
        return status(db_path)
    finally:
        conn.close()


def status(db_path: str = DB_PATH) -> Dict[str, Any]:
    out: Dict[str, Any] = {"db_path": db_path, "tables": {}, "meta": {}, "formats": 0}
    if not os.path.exists(db_path):
        return out
    conn = connect(db_path)
    try:
        for t in DATA_TABLES:
            out["tables"][t] = conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"]
        for row in conn.execute("SELECT key, value FROM meta ORDER BY key"):
            out["meta"][row["key"]] = row["value"]
        out["formats"] = conn.execute("SELECT COUNT(*) AS n FROM format").fetchone()["n"]
        out["integrity"] = conn.execute("PRAGMA integrity_check").fetchone()[0]
    finally:
        conn.close()
    return out
