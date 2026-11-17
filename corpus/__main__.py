"""``python -m corpus`` — run the S4.0 gate on real human battles.

Examples (from the project root):

    python -m corpus --format gen9ou --n 200            # offline fixture
    python -m corpus --jsonl bot/data/replays/gen9uu.jsonl --n 200
    python -m corpus --format gen9ou --n 200 --network  # HF puller if no fixture
"""
from __future__ import annotations

import argparse

from .gate import gate_passes, run_gate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--format", default="gen9ou", help="formatid (default gen9ou)")
    p.add_argument("--n", type=int, default=200, help="number of battles (default 200)")
    p.add_argument("--jsonl", default=None, help="explicit battles .jsonl path")
    p.add_argument("--min-rating", type=float, default=1400.0, help="ELO floor (network pull)")
    p.add_argument("--network", action="store_true", help="use the HF puller if no fixture")
    a = p.parse_args()

    rep, _ = run_gate(jsonl_path=a.jsonl, formatid=a.format, n_battles=a.n,
                      min_rating=a.min_rating, use_network=a.network)
    ok, fails = gate_passes(rep, n_battles=a.n)
    if ok:
        print("\nGATE PASS (S4.0)")
        return 0
    print("\nGATE FAIL (S4.0):")
    for f in fails:
        print("  -", f)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
