"""Build the two reusable team-data files consumed by ``bot/teams.py``.

Both outputs are **plain data files** (no code), keyed by format, so they can be
reused in other projects. They are also copied into the Legality DB project
(``legality/data/``) for its future Smogon ingest (see ``legality/ingest_protocol.md``).

* ``data/movesets.json`` -- per-species **competitive sets** from
  ``https://pkmn.github.io/smogon/data/sets/{format}.json`` (the standard Smogon set
  format: ``{species: {set_name: {moves, item, ability?, nature, evs, ivs?, teratypes?}}}``).
  Kept verbatim (raw) so any project can read the standard format. The random
  teambuilder normalizes/decides alternatives at draw time.
* ``data/teams.json`` -- the **coherent 6-mon teams** the project already has (moved
  out of ``bot/team_pools.json``), i.e. ``{format: [ [mon x6], ... ]}``. Used by the
  ``strong`` builder (the "play immediately" real lineups that teach synergy).

``pkmn.github.io/smogon/data/sets/index.json`` lists every available format; all of the
project's tiers (OU/UU/RU, gens 6-9) are present there.

    python bot/build_movesets.py
    python bot/build_movesets.py --formats gen9ou,gen9uu
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
TEAMS_PATH = os.path.join(DATA_DIR, "teams.json")
MOVESETS_PATH = os.path.join(DATA_DIR, "movesets.json")
POOLS_PATH = os.path.join(HERE, "team_pools.json")
SET_URL = "https://pkmn.github.io/smogon/data/sets/{fmt}.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# The project's single-tier formats across gens 6-9 (all present in the sets index).
DEFAULT_FORMATS: List[str] = [
    "gen6ou", "gen7ou", "gen6uu", "gen6ru",
    "gen7uu", "gen7ru", "gen8ou", "gen8uu",
    "gen8ru", "gen9ou", "gen9uu", "gen9ru",
]


def fetch_json(url: str) -> object:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode("utf8", "replace"))


# ---------------------------------------------------------------------------
# Build-time legality filter
#
# The pkmn.github.io competitive-set data is high quality but NOT guaranteed
# format-legal: it can carry a banned move in an alternative (Umbreon/Yawn under the
# Gen9 Sleep Moves Clause, Comfey/Tera Blast in gen9 OU/UU), a banned form (Zarude-Dada
# tagged UUBL in gen9 UU), or a Hidden Power whose IVs the set never pins down. The
# draw-time code only handles *within-set* move collisions, so we filter the genuinely
# illegal bits HERE, once, using the same Showdown CLI oracle the server itself runs
# (``teamgen`` / ``pokemon-showdown validate-team``). We drop only the offending
# alternatives (banned moves / items / abilities) and keep everything legal -- so the
# random teambuilder can then never emit an illegal team, and the data stays reusable.
# ---------------------------------------------------------------------------

import re  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402

STAT_IDX = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}


def _norm(s: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _base(species: str) -> str:
    s = str(species).strip().lower()
    return s.split("-", 1)[0] if "-" in s else s


def _first_ev_spread(ev):
    if isinstance(ev, (list, tuple)):
        ev = ev[0] if ev else None
    out = [0, 0, 0, 0, 0, 0]
    if isinstance(ev, dict):
        for k, v in ev.items():
            i = STAT_IDX.get(str(k).lower())
            if i is not None:
                out[i] = int(v)
    return out


def _first_iv_spread(iv):
    if isinstance(iv, (list, tuple)):
        iv = iv[0] if iv else None
    if not isinstance(iv, dict) or not iv:
        return None
    out = [31, 31, 31, 31, 31, 31]
    for k, v in iv.items():
        i = STAT_IDX.get(str(k).lower())
        if i is not None:
            out[i] = int(v)
    return out


def _distinct(v):
    """A possibly-list Smogon field -> ordered unique list of values."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        vals = list(v)
    else:
        vals = [v]
    out, seen = [], set()
    for x in vals:
        if x in (None, ""):
            continue
        n = _norm(x)
        if n and n not in seen:
            seen.add(n)
            out.append(x)
    return out


def _set_candidate_moves(set_obj) -> List[str]:
    out: List[str] = []
    for slot in (set_obj.get("moves") or [])[:4]:
        cands = slot if isinstance(slot, (list, tuple)) else [slot]
        for c in cands:
            if c and c not in out:
                out.append(c)
    return out


def _ev_dict(evs_list):
    """6-int stat list -> partial Smogon stat dict (zeros omitted)."""
    keys = ["hp", "atk", "def", "spa", "spd", "spe"]
    d = {k: v for k, v in zip(keys, evs_list) if v}
    return d or {"hp": 1}


def _iv_dict(ivs_list):
    """6-int IV list -> partial Smogon IV dict (31s omitted), or None if all 31."""
    if ivs_list is None:
        return None
    keys = ["hp", "atk", "def", "spa", "spd", "spe"]
    return {k: v for k, v in zip(keys, ivs_list) if v != 31}


def filter_illegal(data: Dict, verbose: bool = True) -> Dict:
    """Validate every set against the format's Showdown CLI and emit legal, concrete sets.

    For each set we find a concrete lineup the server actually accepts: start from the
    set's first choices, validate the mon, and if it is rejected iteratively drop the
    offending move / item / ability (a banned move, a Hidden Power whose IVs the set never
    pins down, a second conflicting Hidden Power, an item the species can't carry, an
    Uber-tagged form, ...) and retry. A set is kept only when a legal lineup of >=1 move
    remains; otherwise it is dropped (e.g. a banned form or a shiny-only event move). The
    output is a *concrete* validated mon per set -- no unresolved alternatives -- which
    is the strongest guarantee that the random teambuilder can never emit an illegal team,
    and keeps the data clean for reuse in other projects.

    This recovers popular mons that fail only on one niche move (Moltres, Azelf, Omastar,
    Serperior, ...) by stripping that move, instead of discarding the whole set.

    Returns a NEW ``{format: {species: {set: concrete_set}}}`` dict.
    """
    import os as _os
    import sys as _sys
    _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
    from poke_env.teambuilder import Teambuilder
    import teamgen
    from teams import _to_tb

    def pack(mons):
        return Teambuilder.join_team([_to_tb(m) for m in mons])

    def run_validate(fmt, mons):
        p = subprocess.run(
            [teamgen.NODE, "pokemon-showdown", "validate-team", fmt],
            input=pack(mons).encode("utf8"), capture_output=True,
            cwd=_os.path.join(teamgen.ROOT, "pokemon-showdown"), timeout=120,
        )
        errs = [l.strip() for l in p.stderr.decode("utf8", "replace").splitlines() if l.strip()]
        return p.returncode == 0, errs

    def _evs_rep(sobj):
        """First non-zero EV spread for a set (a 0-EV mon is rejected by the server)."""
        ev = sobj.get("evs")
        spreads = list(ev) if isinstance(ev, (list, tuple)) else ([ev] if ev else [])
        for s in spreads:
            sp = _first_ev_spread(s)
            if any(sp):
                return sp
        return [100, 0, 0, 0, 0, 0]  # non-zero placeholder so a 0-EV set stays a valid mon

    def analyze(errs):
        """Classify one mon's validator errors.

        returns (drop: set[norm], unfixable: bool, hp_conflict: bool)
          drop        -- norm names (move/item/ability) to remove and retry.
          unfixable   -- species banned / shiny-only / 0 EV: no move-drop fixes it.
          hp_conflict -- a Hidden Power legality problem: remove ALL Hidden Powers.
        """
        drop = set()
        unfixable = False
        hp_conflict = False
        for e in errs:
            m = re.search(r"can't learn (.+?)\.", e, re.I)
            if m:
                drop.add(_norm(m.group(1))); continue
            m = re.search(r'"(.+?)" is an invalid move', e, re.I)
            if m:
                drop.add(_norm(m.group(1))); continue
            m = re.search(r"'s move (.+?) is banned", e, re.I)
            if m:
                drop.add(_norm(m.group(1))); continue
            m = re.search(r"'s move (.+?) can't be transferred", e, re.I)
            if m:
                drop.add(_norm(m.group(1))); continue
            m = re.search(r"has Hidden Power (.+?), but its IVs are for", e, re.I)
            if m:
                hp_conflict = True; drop.add(_norm(m.group(1))); continue
            if re.search(r"multiple copies of|incompatible with", e, re.I):
                hp_conflict = True; continue
            m = re.search(r"can't have (.+?)\.", e, re.I)
            if m:
                drop.add(_norm(m.group(1))); continue
            m = re.search(r"invalid (item|ability)\b", e, re.I)
            if m:
                drop.add(_norm(m.group(1))); continue
            if re.search(r"must be shiny|is tagged|is banned|is illegal|exactly 0 EV|"
                         r"not a valid|doesn't exist|is not allowed", e, re.I):
                unfixable = True; continue
        return drop, unfixable, hp_conflict

    def resolve_set(fmt, species, sobj):
        """Find a legal concrete set for ``species``/``sobj`` in ``fmt`` (or None).

        Returns ``dict(moves=[...], item=str|None, ability=str|None, nature=str, evs=[..],
        ivs=[..]|None, tera=str|None)`` -- the exact lineup that validated.
        """
        cands = _set_candidate_moves(sobj)
        if not cands:
            return None
        items = _distinct(sobj.get("item"))
        abils = _distinct(sobj.get("ability"))
        nature = (_distinct(sobj.get("nature")) or ["Neutral"])[0]
        evs = _evs_rep(sobj)
        ivs = _first_iv_spread(sobj.get("ivs"))
        tera = (_distinct(sobj.get("teratypes")) or [None])[0]
        moves = list(cands)

        def first_bad(lst, bad):
            for x in lst:
                if _norm(x) in bad:
                    return x
            return None

        max_iters = len(moves) + len(items) + len(abils) + 3
        for _ in range(max_iters):
            trial = moves[:4]
            if not trial:
                return None
            mon = dict(species=species, item=(items[0] if items else None),
                       ability=(abils[0] if abils else None), moves=trial,
                       nature=nature, evs=evs, ivs=ivs, tera_type=tera)
            ok, errs = run_validate(fmt, [mon])
            if ok:
                return dict(moves=trial, item=(items[0] if items else None),
                            ability=(abils[0] if abils else None), nature=nature,
                            evs=evs, ivs=ivs, tera=(tera))
            drop, unfixable, hp_conflict = analyze(errs)
            if unfixable:
                return None
            if hp_conflict:
                moves = [m for m in moves if not re.match(r"hidden power", str(m).lower())]
                continue
            if not drop:
                return None
            mv = first_bad(moves, drop)
            if mv is not None:
                out, done = [], False
                for m in moves:
                    if not done and _norm(m) == _norm(mv):
                        done = True
                        continue
                    out.append(m)
                moves = out
                continue
            it = first_bad(items, drop)
            if it is not None:
                items = [x for x in items if _norm(x) != _norm(it)]
                continue
            ab = first_bad(abils, drop)
            if ab is not None:
                abils = [x for x in abils if _norm(x) != _norm(ab)]
                continue
            return None  # no identifiable thing to drop -> can't make progress
        return None

    clean: Dict = {}
    for fmt, by_species in data.items():
        clean[fmt] = {}
        kept = dropped = 0
        for species, named in by_species.items():
            if not isinstance(named, dict):
                continue
            clean[fmt][species] = {}
            for name, sobj in named.items():
                if not isinstance(sobj, dict):
                    clean[fmt][species][name] = sobj
                    kept += 1
                    continue
                r = resolve_set(fmt, species, sobj)
                if r is None:
                    dropped += 1
                    if verbose:
                        print(f"    drop {fmt}/{species} '{name}' (no legal lineup)")
                    continue
                new_s = {
                    "moves": r["moves"],
                    "nature": r["nature"],
                    "evs": _ev_dict(r["evs"]),
                }
                iv = _iv_dict(r["ivs"])
                if iv:
                    new_s["ivs"] = iv
                if r["item"]:
                    new_s["item"] = r["item"]
                if r["ability"]:
                    new_s["ability"] = r["ability"]
                if r["tera"]:
                    new_s["teratypes"] = r["tera"]
                clean[fmt][species][name] = new_s
                kept += 1
            if not clean[fmt][species]:
                del clean[fmt][species]
        if verbose:
            print(f"  {fmt}: {kept} legal sets kept, {dropped} dropped")
    return clean


def build_movesets(formats: List[str], out_path: str, validate: bool = True) -> None:
    """Fetch the competitive sets for each format into ``{format: {species: {set: ...}}}``.

    With ``validate`` (default), every set is then checked against the format's Showdown
    CLI oracle and illegal moves/items/abilities (and unusable sets) are stripped before
    writing -- see ``filter_illegal``.
    """
    existing: Dict = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, encoding="utf8") as fh:
                existing = json.load(fh)
        except Exception:
            existing = {}
    for fmt in formats:
        # pkmn.github.io serves some formats under a different slug (champions
        # -> 'championsou'); canonical map lives in legality.formats.SET_SLUG.
        try:
            import sys as _sys
            _sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from legality import formats as _fmts
            slug = _fmts.set_slug(fmt)
        except Exception:
            slug = fmt
        data = fetch_json(SET_URL.format(fmt=slug))
        if not isinstance(data, dict) or not data:
            raise RuntimeError(f"unexpected payload for {fmt}: not a non-empty object")
        n_species = len(data)
        n_sets = sum(len(v) for v in data.values() if isinstance(v, dict))
        existing[fmt] = data
        print(f"  {fmt}: {n_species} species, {n_sets} named sets (slug={slug})")
        time.sleep(0.15)
    if validate:
        print("[movesets] filtering illegal sets via the Showdown CLI oracle ...")
        existing = filter_illegal(existing, verbose=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf8") as fh:
        json.dump(existing, fh, indent=1)
    n_kept = sum(len(v) for fmt in existing.values() for v in fmt.values() if isinstance(v, dict)) \
        if existing else 0
    print(f"wrote {out_path} ({len(existing)} formats, {n_kept} legal sets)")


def migrate_teams(out_path: str) -> bool:
    """Copy the coherent teams out of ``team_pools.json`` into ``teams.json``.

    ``team_pools.json`` layout: ``{format: {"teams": [[mon x6], ...], "flat": [...], ...}}``
    (legacy formats may be a bare ``[mon, ...]`` list, which has NO coherent teams).
    """
    if not os.path.exists(POOLS_PATH):
        print(f"  (skipping teams.json -- {POOLS_PATH} not found)")
        return False
    with open(POOLS_PATH, encoding="utf8") as fh:
        pools = json.load(fh)
    teams: Dict = {}
    for fmt, entry in pools.items():
        if isinstance(entry, dict):
            t = entry.get("teams")
            if t:
                teams[fmt] = t
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf8") as fh:
        json.dump(teams, fh, indent=1)
    print(f"wrote {out_path} ({len(teams)} formats, "
          f"{sum(len(v) for v in teams.values())} coherent teams)")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the movesets.json + teams.json data files")
    ap.add_argument("--formats", default=",".join(DEFAULT_FORMATS),
                    help="comma-separated formats to fetch sets for")
    ap.add_argument("--skip-teams", action="store_true",
                    help="do not (re)migrate the coherent teams file")
    ap.add_argument("--no-validate", action="store_true",
                    help="skip the Showdown-CLI legality filter (faster; keep raw sets)")
    ap.add_argument("--out-dir", default=DATA_DIR)
    args = ap.parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[movesets] fetching {len(formats)} format(s) from pkmn.github.io/smogon ...")
    build_movesets(formats, os.path.join(args.out_dir, "movesets.json"),
                   validate=not args.no_validate)

    print("[teams] migrating coherent teams from team_pools.json ...")
    if not args.skip_teams:
        migrate_teams(os.path.join(args.out_dir, "teams.json"))


if __name__ == "__main__":
    main()
