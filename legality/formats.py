"""Format registry — the 50 target formats (49 verified in server data, 2026-11-01,
plus the project's custom ``gen9bss3v3``; see PLAN.md §2–3).

Every entry resolves via ``Dex.formats.get`` with ``effectType == 'Format'`` in the
local Showdown build.

``min_team`` / ``max_team`` / ``default_level`` are intentionally None here: they are
format *rules* and get filled at legality-ingest time from the format's ruleTable
(VGC 2024 was 4-mon — never assume 6; LC is level 5; BSS 3v3 picks 3 of 6).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


def _f(id: str, name: str, gen: int, tier: str, game_type: str = "singles",
       mod: Optional[str] = None, section: Optional[str] = None) -> Dict:
    return {
        "id": id, "name": name, "gen": gen, "tier": tier,
        "game_type": game_type, "min_team": None, "max_team": None,
        "default_level": None, "section": section, "mod": mod,
        "exists_in_server": 1,
    }


# ── Ladder singles (24): gen1–9 × OU/UU; RU exists only gen3, 5–9 ──
LADDER: List[Dict] = (
    [_f(f"gen1ou", "[Gen 1] OU", 1, "OU", mod="gen1"),
     _f(f"gen1uu", "[Gen 1] UU", 1, "UU", mod="gen1")]
    + [_f(f"gen2ou", "[Gen 2] OU", 2, "OU", mod="gen2"),
       _f(f"gen2uu", "[Gen 2] UU", 2, "UU", mod="gen2")]
    + [_f(f"gen3ou", "[Gen 3] OU", 3, "OU", mod="gen3"),
       _f(f"gen3uu", "[Gen 3] UU", 3, "UU", mod="gen3"),
       _f(f"gen3ru", "[Gen 3] RU", 3, "RU", mod="gen3")]
    + [_f(f"gen4ou", "[Gen 4] OU", 4, "OU", mod="gen4"),
       _f(f"gen4uu", "[Gen 4] UU", 4, "UU", mod="gen4")]
    + [_f(f"gen5ou", "[Gen 5] OU", 5, "OU", mod="gen5"),
       _f(f"gen5uu", "[Gen 5] UU", 5, "UU", mod="gen5"),
       _f(f"gen5ru", "[Gen 5] RU", 5, "RU", mod="gen5")]
    + [_f(f"gen6ou", "[Gen 6] OU", 6, "OU", mod="gen6"),
       _f(f"gen6uu", "[Gen 6] UU", 6, "UU", mod="gen6"),
       _f(f"gen6ru", "[Gen 6] RU", 6, "RU", mod="gen6")]
    + [_f(f"gen7ou", "[Gen 7] OU", 7, "OU", mod="gen7"),
       _f(f"gen7uu", "[Gen 7] UU", 7, "UU", mod="gen7"),
       _f(f"gen7ru", "[Gen 7] RU", 7, "RU", mod="gen7")]
    + [_f(f"gen8ou", "[Gen 8] OU", 8, "OU", mod="gen8"),
       _f(f"gen8uu", "[Gen 8] UU", 8, "UU", mod="gen8"),
       _f(f"gen8ru", "[Gen 8] RU", 8, "RU", mod="gen8")]
    + [_f(f"gen9ou", "[Gen 9] OU", 9, "OU", mod="gen9"),
       _f(f"gen9uu", "[Gen 9] UU", 9, "UU", mod="gen9"),
       _f(f"gen9ru", "[Gen 9] RU", 9, "RU", mod="gen9")]
)

# ── Ubers (9): id is PLURAL ──────────────────────────────────────────
UBERS: List[Dict] = [
    _f(f"gen{g}ubers", f"[Gen {g}] Ubers", g, "Uber", mod=f"gen{g}") for g in range(1, 10)
]

# ── Little Cup (8): no gen2 ──────────────────────────────────────────
LITTLE_CUP: List[Dict] = [
    _f(f"gen{g}lc", f"[Gen {g}] LC", g, "LC", mod=f"gen{g}")
    for g in (1, 3, 4, 5, 6, 7, 8, 9)
]

# ── VGC (6): canonical latest per gen, all doubles ──────────────────
VGC: List[Dict] = [
    _f("gen4vgc2010", "[Gen 4] VGC 2010", 4, "VGC", "doubles", mod="gen4"),
    _f("gen5vgc2013", "[Gen 5] VGC 2013", 5, "VGC", "doubles", mod="gen5"),
    _f("gen6vgc2016", "[Gen 6] VGC 2016", 6, "VGC", "doubles", mod="gen6"),
    _f("gen7vgc2019", "[Gen 7] VGC 2019", 7, "VGC", "doubles", mod="gen7"),
    _f("gen8vgc2022", "[Gen 8] VGC 2022", 8, "VGC", "doubles", mod="gen8"),
    _f("gen9vgc2025regi", "[Gen 9] VGC 2025 Reg I", 9, "VGC", "doubles", mod="gen9"),
]

# ── Champions (2): gen9 mod; OU singles + VGC doubles ───────────────
CHAMPIONS: List[Dict] = [
    _f("gen9championsou", "[Gen 9 Champions] OU", 9, "OU", "singles", mod="champions"),
    _f("gen9championsvgc2026regmc", "[Gen 9 Champions] VGC 2026 Reg M-C",
       9, "VGC", "doubles", mod="champions"),
]

# ── BSS 3v3 (our AI target format; local-server custom) ────────────
# Fixed-team 3v3 Singles under Battle Spot Singles flat rules (bring 6,
# pick 3). Defined in pokemon-showdown/config/formats.ts (this project's
# server copy) — NOT a stock Smogon format, hence section tag.
BSS: List[Dict] = [
    _f("gen9bss3v3", "[Gen 9] BSS 3v3", 9, "BSS", mod="gen9", section="bss"),
]

# Known-but-not-targeted (available in server data; add via config if wanted):
#   older VGC years: gen4vgc2009, gen5vgc2011/2012, gen6vgc2014/2015,
#     gen7vgc2017/2018, gen8vgc2020/2021, gen9vgc2023regc/regd, gen9vgc2024regg
#   champions: gen9championsuu, gen9championsdoubles, gen9championsvgc2026regmb(+bo3),
#     gen9championsvgc2026regmcbo3
#   ladder variants: gen8bdspubers, gen9doublesubers, gen9nationaldex*, gen9ubersuu ...

TARGET_FORMATS: List[Dict] = LADDER + UBERS + LITTLE_CUP + VGC + CHAMPIONS + BSS

#: dependency order for FK-safe ingestion (children last)
TABLE_ORDER: List[str] = [
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


def format_ids() -> List[str]:
    return [f["id"] for f in TARGET_FORMATS]


def by_id(fmt: str) -> Dict:
    for f in TARGET_FORMATS:
        if f["id"] == fmt:
            return f
    raise KeyError(f"format '{fmt}' not in registry (known: {len(TARGET_FORMATS)})")


# ── @smogon/calc engine key (PLAN.md D32) ────────────────────────────
#
# @smogon/calc's ``calculate(gen, ...)`` indexes its MECHANICS table by this
# key: 1–9 are the standard gens, and 0 is the library's native *champions*
# mechanics set (verified: library champions Slash BP=80 == this fork's
# ``data/mods/champions/moves.ts``; gen-9 Slash=70). Champions formats are
# therefore NOT gen 9 for damage purposes — mapping them to 9 would
# silently mis-calc ~25 moves. The format's ``mod`` (not its ``gen``
# column) is the correct key source.
_MOD_ENGINE_KEYS: Dict[str, int] = {
    "champions": 0,
}
_GEN_MOD = re.compile(r"^gen(\d+)$")


def engine_key(fmt: str) -> int:
    """@smogon/calc engine key (the ``gen`` argument) for a target format id.

    ``genN`` mods map to ``N``; ``champions`` maps to ``0``. Raises
    ``KeyError`` for an unknown format id and ``ValueError`` for a mod we
    cannot map (add it to ``_MOD_ENGINE_KEYS`` when that happens).
    """
    f = by_id(fmt)
    mod = f.get("mod") or ""
    if mod in _MOD_ENGINE_KEYS:
        return _MOD_ENGINE_KEYS[mod]
    m = _GEN_MOD.match(mod)
    if m:
        return int(m.group(1))
    raise ValueError(f"no @smogon/calc engine key for mod {mod!r} (format {fmt!r})")


if __name__ == "__main__":
    print(f"{len(TARGET_FORMATS)} target formats")
    for f in TARGET_FORMATS:
        print(f"  {f['id']:32} {f['game_type']:8} tier={f['tier']}")
    assert len(format_ids()) == len(set(format_ids())), "duplicate format ids!"
    print("OK: ids unique")
