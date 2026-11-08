"""CLI for the legality DB (bootstrap + status).

Usage (run from the repo root, ``showdown AI-imitation/``):

    python -m legality.build bootstrap [--force]   # create schema + 50-format registry
    python -m legality.build status                # table counts + meta + integrity

The DB file is ``legality/legality.db`` unless ``LEGALITY_DB`` points elsewhere.
Bootstrap is idempotent: it re-runs the DDL (all ``IF NOT EXISTS``) and upserts
the format registry, so it is safe to run after ``schema.sql`` changes.
"""
from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional

from . import db


def _print_status(info: Dict[str, Any]) -> None:
    print(f"db:        {info['db_path']}")
    print(f"integrity: {info.get('integrity')}")
    print(f"formats:   {info.get('formats')}")
    for k, v in info.get("meta", {}).items():
        print(f"meta.{k}:  {v}")
    if "tables" in info:
        print("tables:")
        for t, n in info["tables"].items():
            print(f"  {t:32} {n}")


def cmd_bootstrap(args: argparse.Namespace) -> int:
    info = db.bootstrap(force=args.force)
    _print_status(info)
    ok = info.get("integrity") == "ok"
    print("bootstrap OK (forced rebuild)" if args.force and ok else "bootstrap OK" if ok
          else "BOOTSTRAP FAILED — see status above")
    return 0 if ok else 1


def cmd_status(_args: argparse.Namespace) -> int:
    info = db.status()
    _print_status(info)
    ok = info.get("integrity") == "ok"
    return 0 if ok else 1


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(prog="legality.build", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bootstrap", help="create schema + upsert the 50-format registry")
    b.add_argument("--force", action="store_true",
                   help="delete the DB file first (full rebuild)")
    b.set_defaults(fn=cmd_bootstrap)

    s = sub.add_parser("status", help="table counts + meta + integrity check")
    s.set_defaults(fn=cmd_status)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
