"""Build server-verified team pools for tier formats.

The legality oracle is the local Showdown server's own TeamValidator, invoked via
the ``pokemon-showdown validate-team`` CLI -- the exact validator the server runs
when a battle starts, so anything verified here is guaranteed to be accepted in real
battles. No hand-guessed movepools: the server decides which moves/abilities are
legal per format and which mons are banned.

How a candidate is resolved (``resolve``):
  * The CLI only reports learn/ability errors for teams of <=4 moves, and only the
    *first* offending move per call (verified empirically). So ``resolve``:
      1. picks the first ability that produces no "can't have / invalid ability"
         error, bailing immediately on a hard format ban (e.g. "Mewtwo is tagged
         Uber, which is banned");
      2. iteratively builds a legal 4-move set -- each pass tries the next 4
         preference-ordered candidates, drops the move the CLI flags, and retries;
      3. does a final confirm that no legality error remains.
  A mon is only kept if the final confirm is clean, so an illegal mon can never enter
  a pool. This replaces the old resolver, which (a) passed >4 moves on the first check
  so the CLI never reported per-move learn errors, and (b) looked for the literal
  string "invalid ability" when the server actually says "can't have <ability>".

Pools are cached in ``bot/team_pools.json`` and consumed by ``bot/teams.py``.

    python bot/teamgen.py --formats gen6ou,gen7ou
    python bot/teamgen.py --formats gen7ou --refresh
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
POOLS_PATH = os.path.join(HERE, "team_pools.json")
NODE = os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"),
                    "nodejs", "node.exe")

# Candidate mons. ``moves`` is a *preference-ordered* candidate moveset -- the server
# decides which are legal per format and ``resolve`` keeps a legal 4 (it drops any the
# server rejects and refills from the rest in this order). ``abilities`` is fallback
# order; the first one the server accepts is used. Species all exist in Gen 6 and 7.
CANDIDATES: List[Dict] = [
    dict(species="Garchomp", item="Life Orb", abilities=["Rough Skin"],
         moves=["Earthquake", "Stone Edge", "Fire Fang", "Outrage", "Dragon Claw",
                "Dragon Tail", "Roost", "Toxic"],
         nature="Adamant", evs=[0, 252, 0, 4, 0, 252]),
    dict(species="Landorus-T", item="Life Orb", abilities=["Intimidate"],
         moves=["U-turn", "Earth Power", "Stealth Rock", "Knock Off", "Ice Punch",
                "Hypnosis", "Roost", "Swords Dance"],
         nature="Careful", evs=[0, 0, 0, 252, 4, 252]),
    dict(species="Keldeo", item="Life Orb", abilities=["Justified"],
         moves=["Secret Sword", "Aqua Jet", "Surf", "Calm Mind", "Hydro Pump",
                "Roost", "Signal Beam", "Ice Beam"],
         nature="Modest", evs=[0, 0, 0, 252, 4, 252]),
    dict(species="Talonflame", item="Flame Orb", abilities=["Gale Wings"],
         moves=["Brave Bird", "Acrobatics", "Roost", "Defog", "Tailwind",
                "Work Up", "Heat Wave"],
         nature="Adamant", evs=[0, 252, 0, 0, 4, 252]),
    dict(species="Amoonguss", item="Black Sludge", abilities=["Regenerator"],
         moves=["Spore", "Giga Drain", "Toxic", "Synthesis", "Knock Off",
                "Sludge Wave", "Leech Seed"],
         nature="Relaxed", evs=[252, 0, 252, 0, 0, 4]),
    dict(species="Clefable", item="Leftovers", abilities=["Magic Guard"],
         moves=["Calm Mind", "Moonblast", "Soft-Boiled", "Thunder Wave",
                "Toxic", "Wish", "Protect"],
         nature="Modest", evs=[252, 0, 0, 252, 4, 0]),
    dict(species="Ferrothorn", item="Light Clay", abilities=["Iron Barbs"],
         moves=["Spikes", "Leech Seed", "Power Whip", "Knock Off", "Toxic",
                "Wring Out", "Leech Life"],
         nature="Relaxed", evs=[252, 0, 252, 0, 0, 4]),
    dict(species="Gliscor", item="Toxic Orb", abilities=["Poison Heal"],
         moves=["Earthquake", "Swords Dance", "Acrobatics", "Roost", "Taunt",
                "Toxic", "Stealth Rock"],
         nature="Adamant", evs=[0, 252, 0, 0, 4, 252]),
    dict(species="Hippowdon", item="Rocky Helmet", abilities=["Sand Force"],
         moves=["Earthquake", "Stealth Rock", "Slack Off", "Roar", "Ice Beam",
                "Waterfall", "Crush Grip"],
         nature="Impish", evs=[252, 0, 252, 0, 0, 4]),
    dict(species="Mewtwo", item="Focus Sash", abilities=["Pressure"],
         moves=["Psystrike", "Ice Beam", "Aura Sphere", "Recover", "Calm Mind",
                "Flamethrower"],
         nature="Modest", evs=[0, 0, 0, 252, 0, 252]),
    dict(species="Aegislash", item="Focus Sash", abilities=["Stance Change"],
         moves=["Flash Cannon", "Shadow Ball", "Spirit Shield", "Aura Sphere",
                "Recover", "Calm Mind"],
         nature="Modest", evs=[0, 0, 0, 252, 4, 252]),
    dict(species="Greninja", item="Life Orb", abilities=["Protean"],
         moves=["Water Pulse", "Dark Pulse", "Ice Beam", "Substitution", "Surf",
                "Hydro Pump", "U-turn", "Sucker Punch"],
         nature="Timid", evs=[0, 0, 0, 252, 0, 252]),
    dict(species="Rotom-Wash", item="Leftovers", abilities=["Levitate", "Water Absorb"],
         moves=["Thunderbolt", "Hydro Pump", "Will-O-Wisp", "Volt Switch",
                "Pain Split", "Protect"],
         nature="Modest", evs=[0, 0, 0, 252, 4, 0]),
    dict(species="Kyurem", item="Life Orb", abilities=["Pressure"],
         moves=["Ice Beam", "Dragon Pulse", "Earth Power", "Roost",
                "Extreme Speed", "Substitute"],
         nature="Modest", evs=[0, 0, 0, 252, 0, 252]),
    dict(species="Tornadus-T", item="Life Orb", abilities=["Regenerator", "Prankster"],
         moves=["U-turn", "Defog", "Taunt", "Hurricane", "Knock Off", "Roost"],
         nature="Timid", evs=[0, 0, 0, 252, 0, 252]),
    dict(species="Skarmory", item="Leftovers", abilities=["Sturdy"],
         moves=["Spikes", "Brave Bird", "Roost", "Whirlwind", "Toxic",
                "Steel Wing"],
         nature="Impish", evs=[252, 0, 252, 0, 0, 4]),
    dict(species="Heatran", item="Leftovers", abilities=["Flame Body"],
         moves=["Lava Plug", "Earth Power", "Stealth Rock", "Roost",
                "Flare Blitz", "Toxic"],
         nature="Bold", evs=[252, 0, 252, 0, 0, 4]),
    dict(species="Lucario", item="Life Orb", abilities=["Justified", "Adaptability"],
         moves=["Close Combat", "Flash Cannon", "Swords Dance", "Extreme Speed",
                "Bullet Punch", "Ice Punch", "Roost"],
         nature="Jolly", evs=[0, 252, 0, 0, 0, 252]),
    dict(species="Scizor", item="Leftovers", abilities=["Technician"],
         moves=["U-turn", "Roost", "Swords Dance", "Bullet Punch", "Knock Off",
                "Brick Break"],
         nature="Adamant", evs=[0, 252, 0, 0, 4, 252]),
    dict(species="Milotic", item="Leftovers", abilities=["Marvel Scale"],
         moves=["Scald", "Ice Beam", "Toxic", "Recover", "Hydro Pump", "Protect"],
         nature="Bold", evs=[252, 0, 252, 0, 0, 4]),
    dict(species="Gyarados", item="Life Orb", abilities=["Intimidate"],
         moves=["Waterfall", "Bounce", "Roost", "Dragon Dance", "Ice Fang",
                "Earthquake", "Taunt"],
         nature="Jolly", evs=[0, 252, 0, 0, 0, 252]),
    dict(species="Slowbro", item="Leftovers", abilities=["Regenerator", "Own Tempo"],
         moves=["Scald", "Sludge Wave", "Ice Beam", "Toxic", "Calm Mind",
                "Psyshock"],
         nature="Bold", evs=[252, 0, 252, 0, 0, 4]),
    dict(species="Tyranitar", item="Leftovers", abilities=["Sand Stream"],
         moves=["Crunch", "Stone Edge", "Earthquake", "Dragon Dance",
                "Stealth Rock", "Roar", "Pursuit", "Taunt"],
         nature="Adamant", evs=[0, 252, 0, 0, 4, 252]),
    dict(species="Suicune", item="Leftovers", abilities=["Pressure"],
         moves=["Scald", "Ice Beam", "Toxic", "Roar", "Protect", "Calm Mind"],
         nature="Bold", evs=[252, 0, 0, 0, 252, 4]),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _mk(species: str, item: str, ability: str, nature: str, evs: List[int],
        moves: List[str]) -> Dict:
    return dict(species=species, item=item, ability=ability, moves=list(moves),
                nature=nature, evs=list(evs))


def _pack(mons: List[Dict]) -> str:
    """Pack a list of mon dicts the same way poke-env does (join_team)."""
    from poke_env.teambuilder import Teambuilder, TeambuilderPokemon

    tbs = [
        TeambuilderPokemon(
            species=m["species"], item=m["item"], ability=m["ability"],
            moves=list(m["moves"]), nature=m["nature"], evs=list(m["evs"]),
            ivs=[31, 31, 31, 31, 31, 31],
        )
        for m in mons
    ]
    return Teambuilder.join_team(tbs)


# Hard format bans (e.g. Uber/Ubers/BANL) -- a mon is unusable in this format.
BANNED_RE = re.compile(r"\bbanned\b|\bis illegal\b|is not allowed|is tagged", re.I)
# Errors that make a specific mon illegal (as opposed to team-level notes).
MON_ERROR_RE = re.compile(
    r"can't learn|can't have|invalid ability|invalid item|is banned|is illegal|"
    r"not a legal|doesn't exist|is not a valid|is not allowed|is tagged", re.I,
)


def validate(fmt: str, mons: List[Dict]) -> Tuple[bool, List[str]]:
    """Validate via the local server's own TeamValidator CLI. Returns (ok, errors)."""
    p = subprocess.run(
        [NODE, "pokemon-showdown", "validate-team", fmt],
        input=_pack(mons).encode("utf8"), capture_output=True, cwd=os.path.join(ROOT, "pokemon-showdown"),
        timeout=120,
    )
    errors = [l.strip() for l in p.stderr.decode("utf8", "replace").splitlines()
              if l.strip()]
    return p.returncode == 0, errors


def _bad_moves(errors: List[str]) -> set:
    """Move names the server flags as bad. It reports them two ways:
    ``<Mon> can't learn <X>.`` (not in the learnset) and
    ``"<x>" is an invalid move.`` (unknown / illegal move id). Catch BOTH so we
    drop the right move and refill, instead of silently keeping an illegal one."""
    bad = set()
    for e in errors:
        # <Mon> can't learn <X>.   (not in the learnset for this gen)
        m = re.search(r"can't learn (.+?)\.\s*$", e, re.I)
        if m:
            bad.add(_norm(m.group(1)))
            continue
        # "<x>" is an invalid move.   (unknown / removed / illegal move id)
        m = re.search(r'"(.+?)" is an invalid move', e, re.I)
        if m:
            bad.add(_norm(m.group(1)))
            continue
        # <Mon>'s move <X> can't be transferred from Gen A to B.   (gen-transfer lock)
        m = re.search(r"move (.+?) can't be transferred", e, re.I)
        if m:
            bad.add(_norm(m.group(1)))
            continue
    return bad


def resolve(fmt: str, cand: Dict) -> Optional[Dict]:
    """Resolve one candidate mon into a legal final mon for ``fmt`` (or None).

    Tries each listed ability (bailing on a hard ban), then iteratively finds a legal
    4-move set (the CLI reports one unlearnable move per call, so we drop & retry),
    and finally confirms the assembled mon is clean.
    """
    spec, item, nature, evs = cand["species"], cand["item"], cand["nature"], cand["evs"]
    moves_pool = list(cand["moves"])

    # 1. Pick the first legal ability; bail immediately on a hard format ban.
    ability = None
    for a in cand["abilities"]:
        ok, errors = validate(fmt, [_mk(spec, item, a, nature, evs, moves_pool[:4])])
        if any(BANNED_RE.search(e) for e in errors):
            return None
        if not any(("can't have" in e.lower()) or ("invalid ability" in e.lower())
                   for e in errors):
            ability = a
            break
    if ability is None:
        return None

    # 2. Iteratively build a legal 4-move set. The CLI reports the first offending
    #    move each call; we drop it (and any others flagged) and refill from the
    #    preference-ordered pool until the set validates. Success is the CLI ACCEPTING
    #    the trial -- not "we found no flagged move" -- so an unrecognised error form
    #    can never be silently accepted (the old bug: it accepted an illegal trial and
    #    only failed at the final gate). If no move is identified we still make forward
    #    progress by dropping the newest candidate, guaranteeing termination.
    chosen: List[str] = []
    pool = list(moves_pool)
    for _ in range(len(moves_pool) + 2):
        trial = (chosen + [m for m in pool if m not in chosen])[:4]
        if len(trial) < 4:
            break
        ok, errors = validate(fmt, [_mk(spec, item, ability, nature, evs, trial)])
        if ok:
            chosen = list(trial)
            break
        bad = _bad_moves(errors)
        if not bad:
            bad.add(_norm(trial[-1]))  # unknown error form: back off the newest pick
        pool = [m for m in pool if _norm(m) not in bad and m not in chosen]
        chosen = [m for m in chosen if _norm(m) not in bad]
    if len(chosen) < 4:
        return None

    # 3. Final confirm: for a single-mon team the CLI only fails when the mon is
    #    illegal (it does not enforce team size), so ANY failure means the assembled
    #    mon is not legal -> drop it. This is the ultimate gate: an illegal mon can
    #    never reach the pool, so no battle-start team can be rejected by the server.
    final = dict(cand, ability=ability, moves=chosen)
    ok, _errors = validate(fmt, [_mk(spec, item, ability, nature, evs, chosen)])
    if not ok:
        return None
    return final


def build_pool(fmt: str, min_size: int) -> List[Dict]:
    pool: List[Dict] = []
    for cand in CANDIDATES:
        try:
            r = resolve(fmt, cand)
        except Exception as ex:
            print(f"  ERR  {fmt}: {cand['species']}: {ex}")
            r = None
        if r:
            pool.append(r)
            print(f"  OK   {fmt}: {r['species']} | {', '.join(r['moves'])} [{r['ability']}]")
        else:
            print(f"  FAIL {fmt}: {cand['species']}")
    if len(pool) < min_size:
        raise SystemExit(
            f"[{fmt}] only {len(pool)} verified mons (< {min_size}); expand CANDIDATES"
        )
    return pool


def main() -> None:
    ap = argparse.ArgumentParser(description="Build server-verified team pools")
    ap.add_argument("--formats", default="gen6ou,gen7ou")
    ap.add_argument("--min-size", type=int, default=8)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]

    pools: Dict[str, List[Dict]] = {}
    if os.path.exists(POOLS_PATH) and not args.refresh:
        try:
            pools = json.load(open(POOLS_PATH, encoding="utf8"))
        except Exception:
            pools = {}

    for fmt in formats:
        if fmt in pools and not args.refresh:
            print(f"[{fmt}] using cached pool ({len(pools[fmt])} mons)")
            continue
        print(f"[{fmt}] verifying candidates against the server...")
        pools[fmt] = build_pool(fmt, args.min_size)

    with open(POOLS_PATH, "w", encoding="utf8") as fh:
        json.dump(pools, fh, indent=1)
    print(f"wrote {POOLS_PATH}")
    for f, p in pools.items():
        print(f"  {f}: {len(p)} mons -> {[m['species'] for m in p]}")


if __name__ == "__main__":
    main()