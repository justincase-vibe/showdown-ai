"""Replay health-check: scan recent self-play replays and FLAG bad behaviour.

Why this exists
---------------
The bot's silent failure modes look identical to "a reasonable choice" when you read a
single replay by eye:

  * a **turn-1 concede** (forfeit with the team still intact) looks like a normal game;
  * **"always switch to the same mon"** (a mask / forced-switch bug) looks like a strategy;
  * **duplicate replays** in the "last N" list hide the fact that we only have 2-3 games.

This tool parses the embedded Showdown log of each replay HTML and reports the signals
that would catch those, so you can audit a run in seconds without spectating by hand.

Pure stdlib, CPU-only, no server / no torch -- safe to run any time.

Usage
-----
    python bot/replay_diag.py --last 50                 # 50 most-recent replays
    python bot/replay_diag.py --last 200 --json         # machine-readable summary
    python bot/replay_diag.py --file "bot/replays/ShowdownEnv 8fyf8 - battle-gen8uu-1672.html"
    python bot/replay_diag.py --last 50 --min-moves 4   # raise the "did it even fight" bar
"""

from __future__ import annotations

import argparse
import collections
import datetime as _dt
import hashlib
import json
import os
import re
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Log extraction + parsing
# --------------------------------------------------------------------------- #
_LOG_BLOCK_RE = re.compile(
    r'<script type="text/plain" class="battle-log-data">(.*?)</script>',
    re.S,
)
_SLOT_FROM_CHAR = {c: i for i, c in enumerate("abcdef")}


def extract_log(html_path: str) -> str:
    """Pull the raw pipe-delimited battle log out of a poke-env replay HTML file."""
    with open(html_path, "r", encoding="utf-8", errors="replace") as fh:
        html = fh.read()
    m = _LOG_BLOCK_RE.search(html)
    if not m:
        # Some replays embed the log inline in a <div class="battle-log">; fall back to
        # grabbing every line that starts with '|' or '>'.
        lines = [ln.strip() for ln in html.splitlines()]
        body = [ln for ln in lines if ln.startswith("|") or ln.startswith(">")]
        return "\n".join(body)
    return m.group(1)


def _side_from_token(tok: str) -> Optional[str]:
    """`p1a: Hatterene` -> 'p1'; `p2` -> 'p2'; otherwise None."""
    tok = tok.strip()
    if tok[:2] in ("p1", "p2") and (len(tok) == 2 or not tok[2].isalpha() or tok[2] in "abcdef"):
        return tok[:2]
    return None


def _slot_from_token(tok: str) -> Optional[int]:
    """`p1a: Hatterene` -> 0. Only the letter right after p1/p2 counts as a slot."""
    tok = tok.strip()
    if len(tok) >= 3 and tok[:2] in ("p1", "p2") and tok[2] in _SLOT_FROM_CHAR:
        return _SLOT_FROM_CHAR[tok[2]]
    return None


def concentration(counts: Dict[int, int]) -> float:
    """Max fraction of a single bench slot (1.0 = every switch hit the same mon)."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


def parse_battle(log: str) -> Dict:
    """Parse one battle log into the health-check fields we care about."""
    res = {
        "turns": 0,
        "forfeit": False,
        "forfeit_turn": None,
        "winner": None,
        "winner_side": None,
        "concede": False,          # forfeit + the conceding side barely played
        "moves": {"p1": 0, "p2": 0},
        "switches": {"p1": 0, "p2": 0},
        "switch_slots": {"p1": {}, "p2": {}},   # side -> {slot: count}
        "distinct_moves": {"p1": set(), "p2": set()},
        "faints": {"p1": 0, "p2": 0},
        "signature": None,         # hash of the action stream (duplicate detection)
    }
    title = None
    stream: List[str] = []
    current_turn = 0

    for raw in log.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(">"):
            continue  # replay id / first line
        if not line.startswith("|"):
            continue
        fields = line.split("|")
        # fields[0] is '' (leading '|'); the tag is fields[1].
        if len(fields) < 2:
            continue
        tag = fields[1].strip()
        if tag == "title":
            title = fields[2] if len(fields) > 2 else None
        elif tag == "turn" and len(fields) >= 3:
            try:
                current_turn = int(fields[2])
                res["turns"] = max(res["turns"], current_turn)
            except ValueError:
                pass
        elif tag == "switch" and len(fields) >= 3:
            side = _side_from_token(fields[2])
            slot = _slot_from_token(fields[2])
            if side:
                res["switches"][side] += 1
                if slot is not None:
                    res["switch_slots"][side][slot] = (
                        res["switch_slots"][side].get(slot, 0) + 1)
                stream.append(f"sw:{side}:{slot}")
        elif tag == "move" and len(fields) >= 4:
            side = _side_from_token(fields[2])
            move = fields[3].strip() if len(fields) > 3 else ""
            if side:
                res["moves"][side] += 1
                if move:
                    res["distinct_moves"][side].add(move.lower())
                stream.append(f"mv:{side}:{move}")
        elif tag == "faint" and len(fields) >= 3:
            side = _side_from_token(fields[2])
            if side:
                res["faints"][side] += 1
        elif tag == "forfeit":
            res["forfeit"] = True
            if res["forfeit_turn"] is None:
                res["forfeit_turn"] = current_turn
        elif tag == "win" and len(fields) >= 3:
            res["winner"] = fields[2].strip()
            res["winner_side"] = _side_from_token(fields[2])
        elif tag == "-message" and len(fields) >= 3:
            msg = fields[2].lower()
            if "forfeited" in msg or "forfeit" in msg:
                res["forfeit"] = True
                if res["forfeit_turn"] is None:
                    res["forfeit_turn"] = current_turn

    # A 'forfeit' can also be signalled only via the win line + a still-intact losing
    # team; keep the explicit flag as the source of truth, but infer the conceding side.
    if res["winner"] is not None and title:
        parts = [p.strip() for p in title.split("vs.")]
        if len(parts) == 2:
            p1name, p2name = parts[0], parts[1]
            if res["winner"] == p1name:
                res["winner_side"] = "p1"
            elif res["winner"] == p2name:
                res["winner_side"] = "p2"

    # Concede = forfeit where the losing side barely fought (<= min-moves moves).
    if res["forfeit"] and res["winner_side"] is not None:
        loser = "p2" if res["winner_side"] == "p1" else "p1"
        res["concede"] = res["moves"][loser] <= 1

    res["sig"] = hashlib.md5("|".join(stream).encode("utf-8")).hexdigest() if stream else None
    res["switch_conc"] = {
        s: round(concentration(res["switch_slots"][s]), 3)
        for s in ("p1", "p2")
    }
    # Drop the mutable sets from the returned structure (keep them separately).
    res["n_distinct_moves"] = {
        s: len(res["distinct_moves"][s]) for s in ("p1", "p2")
    }
    res["_distinct_moves"] = res["distinct_moves"]
    del res["switch_slots"]
    return res


# --------------------------------------------------------------------------- #
# Collection + report
# --------------------------------------------------------------------------- #
def latest_replays(folder: str, n: int) -> List[str]:
    paths = []
    for name in os.listdir(folder):
        if name.lower().endswith(".html"):
            paths.append(os.path.join(folder, name))
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return paths[:n]


def analyze(paths: List[str], min_moves: int = 4) -> Dict:
    per_replay = []
    dup_groups: Dict[str, List[str]] = collections.defaultdict(list)
    for p in paths:
        try:
            r = parse_battle(extract_log(p))
        except Exception as exc:  # noqa: BLE001 - report, don't crash the whole scan
            per_replay.append({"file": os.path.basename(p), "error": str(exc)[:120]})
            continue
        r["file"] = os.path.basename(p)
        per_replay.append(r)
        if r.get("sig"):
            dup_groups[r["sig"]].append(os.path.basename(p))

    # --- aggregate ---
    n = len(per_replay)
    n_forfeit = sum(1 for r in per_replay if r.get("forfeit"))
    n_concede = sum(1 for r in per_replay if r.get("concede"))
    n_degen = 0
    degen_replays: List[str] = []
    for r in per_replay:
        for s in ("p1", "p2"):
            if r.get("switches", {}).get(s, 0) >= min_moves and r.get("switch_conc", {}).get(s, 0) >= 0.8:
                n_degen += 1
                degen_replays.append(f"{r['file']} [{s}: {r['switches'][s]} switches, conc={r['switch_conc'][s]}]")
                break
    low_div = 0
    low_replays: List[str] = []
    for r in per_replay:
        for s in ("p1", "p2"):
            if r.get("moves", {}).get(s, 0) >= min_moves and r.get("n_distinct_moves", {}).get(s, 99) <= 1:
                low_div += 1
                low_replays.append(f"{r['file']} [{s}: {r['moves'][s]} moves, {r['n_distinct_moves'][s]} distinct]")
                break
    dup_files = [v for v in dup_groups.values() if len(v) > 1]
    dup_count = sum(len(v) - 1 for v in dup_files)  # number of *extra* copies

    return {
        "n_replays": n,
        "n_forfeit": n_forfeit,
        "n_concede": n_concede,
        "n_degenerate_switching": n_degen,
        "degenerate_switching": degen_replays,
        "n_low_move_diversity": low_div,
        "low_move_diversity": low_replays,
        "n_duplicate_copies": dup_count,
        "duplicate_groups": [v for v in dup_files],
        "per_replay": per_replay,
    }


def print_report(rep: Dict, show_all: bool = False) -> None:
    bar = "=" * 62
    print(bar)
    print(f"REPLAY HEALTH CHECK  --  {rep['n_replays']} replays scanned")
    print(bar)
    print(f"  forfeits / concedes : {rep['n_forfeit']}  /  {rep['n_concede']}  "
          f"{'  <-- BUG if concede > 0' if rep['n_concede'] else ''}")
    print(f"  degenerate switching: {rep['n_degenerate_switching']}")
    print(f"  low move diversity  : {rep['n_low_move_diversity']}")
    print(f"  duplicate copies    : {rep['n_duplicate_copies']}"
          f"  ({len(rep['duplicate_groups'])} group(s))")

    if rep["n_concede"] or rep["n_degenerate_switching"] or rep["n_low_move_diversity"]:
        print("\n  -- flagged replays --")
        for f in rep["degenerate_switching"]:
            print("   [SWITCH]  " + f)
        for f in rep["low_move_diversity"]:
            print("   [DIVERS]  " + f)

    # Concede / forfeit detail (the thing the user keeps hitting).
    forfeit_rows = [r for r in rep["per_replay"] if r.get("forfeit")]
    if forfeit_rows:
        print("\n  -- forfeit / concede detail --")
        for r in forfeit_rows:
            tag = "CONCEDE" if r.get("concede") else "forfeit"
            print(f"   [{tag:7}] {r['file']}  winner={r.get('winner_side')} "
                  f"forfeit_turn={r.get('forfeit_turn')} "
                  f"moves p1/p2={r['moves']['p1']}/{r['moves']['p2']}")

    if dup_groups_flat := rep["duplicate_groups"]:
        print("\n  -- duplicate replay groups (same action stream) --")
        for grp in dup_groups_flat:
            print(f"   {len(grp)}x: {', '.join(grp)}")

    if show_all:
        print("\n  -- per-replay table --")
        for r in rep["per_replay"]:
            if "error" in r:
                print(f"   ERR  {r['file']}: {r['error']}")
                continue
            print(
                f"   {r['file']}\n"
                f"       turns={r['turns']} winner={r.get('winner_side')} "
                f"mv p1/p2={r['moves']['p1']}/{r['moves']['p2']} "
                f"sw p1/p2={r['switches']['p1']}/{r['switches']['p2']} "
                f"conc p1/p2={r['switch_conc']['p1']}/{r['switch_conc']['p2']} "
                f"distinct p1/p2={r['n_distinct_moves']['p1']}/{r['n_distinct_moves']['p2']}"
            )
    print(bar)


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    default_folder = os.path.join(here, "replays")
    p = argparse.ArgumentParser(description="Scan Showdown replays and flag bad behaviour.")
    p.add_argument("--folder", default=default_folder, help="replays directory")
    p.add_argument("--last", type=int, default=50, help="how many most-recent replays to scan")
    p.add_argument("--file", default=None, help="scan a single replay file")
    p.add_argument("--min-moves", type=int, default=4,
                   help="a side needs >= this many moves before 'low diversity' applies")
    p.add_argument("--show-all", action="store_true", help="print the full per-replay table")
    p.add_argument("--json", action="store_true", help="emit a machine-readable JSON summary")
    args = p.parse_args()

    if args.file:
        paths = [args.file]
    else:
        paths = latest_replays(args.folder, args.last)

    if not paths:
        print(f"no replays found in {args.folder}")
        return

    rep = analyze(paths, min_moves=args.min_moves)

    # Make the JSON output serializable (strip sets / per-replay internals).
    if args.json:
        slim = {k: v for k, v in rep.items() if k != "per_replay"}
        slim["per_replay"] = [
            {k: v for k, v in r.items() if k not in ("_distinct_moves",)}
            for r in rep["per_replay"]
        ]
        print(json.dumps(slim, indent=2, default=str))
        return

    print_report(rep, show_all=args.show_all)


if __name__ == "__main__":
    main()
