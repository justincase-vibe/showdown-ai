"""Build coherent, server-validated team pools from Smogon's curated sample teams.

Data source
-----------
crob.at mirrors Smogon's official "Sample Teams" as clean JSON
(``https://crob.at/api/team/{slug}``, each team's ``teams[].paste`` is the exact
Showdown import string) and lists a format's teams at ``https://crob.at/teams/{fmt}``.
That list page tags every team card with ``data-team-browser-source`` -- ``sample``
(council/sourced, ``data-trust-band="current_sourced"``) vs ``community`` (user-
submitted) -- so we parse the *card attributes* to get a reliable, ordered,
provenance-labelled slug list. (The ``?tab=`` and ``?page=`` URL params are no-ops
on the current crob.at build; the whole list is served on one page.)

Each team is parsed into our mon-dict schema (species/item/ability/moves/nature/evs)
and the *whole* 6-mon team is validated against the local server's own TeamValidator
via ``teamgen.validate`` (the same oracle that accepts/rejects at battle start). A
team is kept only if the oracle accepts it, so every stored team is guaranteed
runnable. This yields **coherent teams** (the real 6 that go together), which is
what teaches team synergy -- distinct from a random bag of sets. The flat union of
every mon across a format's teams is also stored so a "random team" generator can
sample from it later.

Provenance is tracked per format (``meta.provenance``) and per team
(``meta.team_provenance``) so the user can see how many are Smogon-curated vs
community vs (later) synthesized.

    python bot/smogon_teams.py --formats gen9ou --dry-run
    python bot/smogon_teams.py --formats gen6uu,gen6ru,gen7uu,gen7ru,gen8ou,gen8uu,gen8ru,gen9ou,gen9uu,gen9ru
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
POOLS_PATH = os.path.join(HERE, "team_pools.json")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
CROB_TEAM = "https://crob.at/api/team/{slug}"
CROB_LIST = "https://crob.at/teams/{fmt}"
STAT = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}
IGNORE_PREFIX = ("gender", "tera type", "ivs", "level", "shiny", "dex", "height",
                 "weight", "shininess")

# --- card-list parsing (reliable provenance signal) ---------------------------
# Each team card is an <a ...> that carries both an href="/SLUG" and a
# data-team-browser-source="sample|community". Order in the page is the site's
# curated order (samples first). Attribute order varies, so we match a full <a>
# tag and pull the two fields out of it.
_A_TAG_RE = re.compile(r"<a\b[^>]*?>", re.S)
_HREF_RE = re.compile(r'href="(/[A-Za-z0-9_-]+)"')
_SOURCE_RE = re.compile(r'data-team-browser-source="(sample|community)"')
# Fallback (no card attrs): any crob.at/<slug> link, 6-8 word chars.
_SLUG_FALLBACK_RE = re.compile(r"crob\.at/([A-Za-z0-9]{6,8})")


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=25) as r:
        return r.read().decode("utf8", "replace")


def parse_paste(paste: str) -> List[Dict]:
    """Showdown/Smogon import paste -> list of mon-dicts (our schema)."""
    mons: List[Dict] = []
    for block in re.split(r"\n\s*\n", paste.strip()):
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        first = lines[0]
        if " @ " in first:
            species, item = (x.strip() for x in first.split(" @ ", 1))
        else:
            species, item = first.strip(), ""
        ability: Optional[str] = None
        nature: Optional[str] = None
        evs = [0, 0, 0, 0, 0, 0]
        moves: List[str] = []
        for ln in lines[1:]:
            low = ln.lower()
            if low.startswith("ability:"):
                ability = ln.split(":", 1)[1].strip()
            elif low.startswith(IGNORE_PREFIX):
                continue
            elif low.startswith("evs:"):
                for m in re.finditer(r"(\d+)\s+(hp|atk|def|spa|spd|spe)", ln, re.I):
                    evs[STAT[m.group(2).lower()]] = int(m.group(1))
            elif re.search(r"nature$", low):
                nature = re.sub(r"\s*nature$", "", low).strip()
            elif ln.startswith("- "):
                moves.append(ln[2:].strip())
        mons.append(dict(
            species=species, item=item, ability=(ability or ""),
            moves=moves[:4], nature=(nature or "Neutral").capitalize(), evs=evs,
        ))
    return mons


def _team_paste(data: Dict, fmt: str) -> Optional[str]:
    """Paste for ``fmt`` from a team JSON (multi-format teams pick their slot)."""
    for t in data.get("teams", []):
        if t.get("format") == fmt and t.get("paste"):
            return t["paste"]
    return None


def list_team_slugs(fmt: str) -> List[Tuple[str, str]]:
    """Ordered (source, slug) for a format -- samples first, then community.

    Parses the list page's team cards (``data-team-browser-source``). Falls back to
    raw slug links (provenance unknown -> 'community') if the card markup changes.
    """
    html = fetch(CROB_LIST.format(fmt=fmt))
    ordered: List[Tuple[str, str]] = []
    seen = set()
    for m in _A_TAG_RE.finditer(html):
        tag = m.group(0)
        if "data-team-browser-source" not in tag:
            continue
        sm = _SOURCE_RE.search(tag)
        hm = _HREF_RE.search(tag)
        if not sm or not hm:
            continue
        slug = hm.group(1).lstrip("/")
        if slug not in seen:
            seen.add(slug)
            ordered.append((sm.group(1), slug))
    if ordered:
        # Stable sort: samples (0) before community (1), preserving page order.
        ordered.sort(key=lambda x: 0 if x[0] == "sample" else 1)
        return ordered
    # Fallback: no card attrs found -> grab raw slugs, treat as community.
    for slug in _SLUG_FALLBACK_RE.findall(html):
        if slug not in seen:
            seen.add(slug)
            ordered.append(("community", slug))
    return ordered


def fetch_team_mons(fmt: str, slug: str) -> Optional[List[Dict]]:
    """Fetch a team; keep only if it actually belongs to ``fmt`` (strict).

    All 12 target formats are 6-mon, so we require exactly 6 parsed mons (a 5- or
    7-mon paste is not a standard team for these formats)."""
    data = json.loads(fetch(CROB_TEAM.format(slug=slug)))
    paste = _team_paste(data, fmt)
    if not paste:
        return None
    mons = parse_paste(paste)
    if len(mons) != 6:
        return None
    return mons


def build_format(fmt: str, target: int = 25, verbose: bool = True
                 ) -> Tuple[List[List[Dict]], List[str]]:
    """Collect up to ``target`` oracle-validated teams for ``fmt``.

    Samples first, then community (to top up). Returns (teams, provenance) where
    ``provenance[i]`` is the source label for ``teams[i]``.
    """
    import teamgen  # local: pulls in the server oracle

    ordered = list_team_slugs(fmt)
    teams: List[List[Dict]] = []
    provenance: List[str] = []
    seen_mons = set()
    for source, slug in ordered:
        if len(teams) >= target:
            break
        try:
            mons = fetch_team_mons(fmt, slug)
        except Exception:
            time.sleep(0.25)
            continue
        if not mons:
            continue
        # skip exact duplicates (same 6 species+items)
        key = tuple(sorted((m["species"], m["item"]) for m in mons))
        if key in seen_mons:
            continue
        ok, errs = teamgen.validate(fmt, mons)
        if ok:
            seen_mons.add(key)
            teams.append([dict(m) for m in mons])
            provenance.append(source)
            if verbose:
                print(f"  OK  [{source:9}] {', '.join(m['species'] for m in mons)}")
        else:
            if verbose:
                print(f"  --  [{source:9}] ILLEGAL: {errs[:2]}")
        time.sleep(0.2)
    return teams, provenance


def _flat_union(teams: List[List[Dict]]) -> List[Dict]:
    seen, out = set(), []
    for team in teams:
        for m in team:
            k = (m["species"], m["item"], m["ability"], tuple(m["moves"]))
            if k not in seen:
                seen.add(k)
                out.append(dict(m))
    return out


def save_pools(new: Dict[str, Dict], out_path: str) -> None:
    pools: Dict = {}
    if os.path.exists(out_path):
        try:
            pools = json.load(open(out_path, encoding="utf8"))
        except Exception:
            pools = {}
    for fmt, payload in new.items():
        pools[fmt] = payload  # {"teams": [...], "flat": [...], "meta": {...}}
    with open(out_path, "w", encoding="utf8") as fh:
        json.dump(pools, fh, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build validated team pools from Smogon sample teams")
    ap.add_argument("--formats", default=",".join([
        "gen6uu", "gen6ru", "gen7uu", "gen7ru", "gen8ou", "gen8uu",
        "gen8ru", "gen9ou", "gen9uu", "gen9ru"]))
    ap.add_argument("--target", type=int, default=25)
    ap.add_argument("--out", default=POOLS_PATH)
    ap.add_argument("--dry-run", action="store_true", help="report only, do not write")
    args = ap.parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    new: Dict[str, Dict] = {}
    for fmt in formats:
        print(f"[{fmt}] collecting up to {args.target} validated teams ...")
        teams, provenance = build_format(fmt, target=args.target)
        flat = _flat_union(teams)
        from collections import Counter
        prov = dict(Counter(provenance))
        print(f"  => {len(teams)} legal teams ({prov}), {len(flat)} unique mons")
        new[fmt] = {
            "teams": teams,
            "flat": flat,
            "meta": {
                "source": "smogon sample+community via crob.at",
                "n_teams": len(teams), "n_mons": len(flat),
                "provenance": prov,
                "team_provenance": provenance,
                "built": time.strftime("%Y-%m-%d"),
            },
        }

    if args.dry_run:
        print(f"\nDRY-RUN: not writing {args.out}")
        return
    save_pools(new, args.out)
    print(f"\nwrote {args.out}")
    for fmt, p in new.items():
        print(f"  {fmt}: {p['meta']['n_teams']} teams "
              f"({p['meta']['provenance']}) / {p['meta']['n_mons']} mons")


if __name__ == "__main__":
    main()
