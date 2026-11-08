# Legality DB — module sheet (plan + rules, wipe-proof)

> **Read this first if the chat was wiped.** Mission, decisions, verified facts, schema,
> current state. Status is current as of the last line in §6.
>
> **Repo root:** `C:/Users/JustinCase/Documents/Projects/showdown AI-imitation`
> (Showdown server copy: `pokemon-showdown/`).
> **Parent roadmap:** [../PLAN.md](../PLAN.md) (10-step build plan; this module = Step 1).

---

## 1. Mission

A local **source of truth** database over this Showdown server's own data — the bot's
instant legality oracle and lookup layer:

1. **Gen 9 Pokémon data** — forms, types, abilities (incl. hidden), tier, base stats.
2. **Per-species move lists** (learnsets) **with move properties** (type/power/accuracy/
   PP/description + effect flags: hazard, screen, tailwind, boosts, drain, recoil).
3. **Common sets per Pokémon per format** (Smogon), with normalized probability.
4. **Legal species/moves/abilities/items per format** — the Fissure rule materialized.
5. **Type chart** exactly as *this build's* sim computes it.

**Scope (this project): Gen 9**, formats `gen9ou gen9uu gen9ru gen9bss3v3` (D22).
The schema still supports all gens (tables are gen-scoped) — but only gen 9 is populated
and tested. **Read-only at runtime; one rebuild/ingest path** (D13, D16).

**Status: STEP 1 COMPLETE (2026-11-07)** — 21/21 tests green, DB populated, API live.

---

## 2. Decision log

D1–D19 below are the module's foundation (2026-11-01, carried over from the old project —
still valid). D20–D23 are this project's Step 1 data decisions (assigned during the
Step 1 session; also in the root PLAN.md §4, where D25–D28 additionally cover the
project-level setup: new folder, plan adoption, format choice, JSON pipeline).

| # | Decision | Rationale / note | Date |
|---|----------|------------------|------|
| D1 | Storage = **SQLite** (single `legality.db`) | wipeable win-rate tables + fast sampling; no server; `sqlite3` built-in | 2026-11-01 |
| D2 | Legality oracle = **local Showdown repo sim** (this fork) | exact data the local server runs → no drift; used at *ingest* time | 2026-11-01 |
| D4 | **Forms are first-class** (team-level unit) | Rotom-Wash ≠ Rotom-Heat; forms banned independently | 2026-11-01 |
| D5 | Legality tables store **memberships only** (legal combos) | size + matches bot queries | 2026-11-01 |
| D7 | **Fissure rule:** move legal in format ⇔ learnable in the gen ∧ not format-banned | user-specified; verified (Dugtrio/Fissure; banned in gen9ou) | 2026-11-01 |
| D8 | **Form legal in format** ⇔ not explicitly banned ∧ tier not banned ∧ ≥ 1 legal ability | matches `checkSpecies` + Obtainable-Abilities | 2026-11-01 |
| D9 | **Per-gen tier per form** (`form_tier`) | per-gen tier shifts (Garchomp OU@6 → UUBL@9) | 2026-11-01 |
| D13 | DB **read-only at runtime**; one rebuild/ingest command; `meta` + `ingest_log` provenance | honest source of truth; drift detectable | 2026-11-01 |
| D16 | **Ingest protocol:** one upsert payload contract (`ingest_protocol.md`), source-tagged, transactional, logged | shapes future scrapers; any language can emit it | 2026-11-01 |
| D17 | **Sets = scraped strategies** (`strategy` + `strategy_move`), format-scoped, item/nature/EVs + source URL | Smogon "Strategies" parity; feeds team gen + belief (step 3) | 2026-11-01 |
| D19 | Sets table named **`strategy`** | `set` is a reserved SQL word | 2026-11-01 |
| D20 | **Schema v3:** `strategy.ability_id` added | Smogon sets always list an ability; the AI needs it | 2026-11-07 |
| D21 | **Type chart:** encoding `{0:1x, 1:2x, 2:0.5x, 3:0x}`; stored `[defending][attacking]` | verified vs `sim/dex.ts` + `dist/data/typechart.js` | 2026-11-07 |
| D22 | **Type chart = 18 real types only** (324 rows; `stellar` excluded) | pseudo-types are not playable types | 2026-11-07 |
| D23 | **Set ingester skips sets for species not legal in the format** | unplayable sets pollute step 3's belief space; gen9: 489 raw → 477 ingested | 2026-11-07 |
| D24 | `get_type_chart()` returns **display-name keys** (match `move.type`/`form_type.type`) | direct `chart[defender_type][move_type]` indexing for step 2 | 2026-11-07 |
| D29 | **Fork's data = ground truth.** Chart matches official (20 cells verified). Species typing deviates: `ninetalesalola` = Ice/Fairy. Earlier "chart deviations" note was an **axis mix-up** (Rock→Bug is 2×; the 1× cell was Bug→Rock) | tests assert fork-verified values, never memory of the official chart | 2026-11-07 |

(Full table incl. D3, D6, D10–D12, D14–D15, D18: old project `showdown AI/legality/PLAN.md`
— read-only reference.)

---

## 3. Verified facts (evidence — don't re-derive)

### 3.1 Type chart (the one axis trap)

- Source of truth: `pokemon-showdown/dist/data/typechart.js` →
  `TypeChart[defender_id].damageTaken[Attacker_name]`, encoding
  **0=1×, 1=2×, 2=0.5×, 3=immune** (confirmed in `sim/dex.ts getEffectiveness/
  getImmunity`). NOT log2.
- **Axis:** `Rock→Bug` = `bug.damageTaken.Rock` = 1 → **2× (super-effective)**.
  `rock.damageTaken.Bug` = 0 → 1× is **Bug→Rock**. One previous test failed on
  exactly this confusion (2026-11-07, fixed in test + this note).
- **20 cells verified** attacker→defender: Water→Fire 2 · Fire→Grass 2 · Ice→Dragon 2 ·
  Poison→Fairy 2 · Steel→Fairy 2 · **Rock→Bug 2** · Electric→Ground 0 · Dragon→Fairy 0 ·
  Ghost→Normal 0 · Psychic→Dark 0 · Poison→Steel 0 · Ground→Flying 0 · Fighting→Ghost 0 ·
  Normal→Ghost 0 · Grass→Grass 0.5 · Ground→Bug 0.5 · Fairy→Fire 0.5 · Fire→Fairy 1 ·
  Bug→Bug 1 · Bug→Rock 1. DB `type_chart` (324 rows, gen 9) matches all of them.
- Chart matches the **official** chart on these cells. The fork's *species* data does
  deviate (below) — so never assume either.

### 3.2 Fork data quirks

- **`ninetalesalola` = Ice/Fairy** (official: Fire/Fairy) — `dist/data/pokedex.js`.
  Dex ids here have **no dash**: `ninetalesalola`, not `ninetales-ala`.
- `dex.species.all()/moves.all()/…` return **effect objects** — use `.id`.
- `dex.data.Types` is undefined → enumerate the 18 real types explicitly.
- `dex.types.get('fire')` works; `damageTaken` inner keys are **display names**,
  top-level keys are **ids**.
- Showdown `gen` field = gen **introduced**, not removed.
- **BSS format:** `pickedTeamSize=3`, `maxLevel=100`, `maxTeam` **unset** in this fork
  (assert `in (None, 6)`).
- Legacy `Hidden Power <type>` (16 variants) all carry server id `hiddenpower`;
  extractor skips name/id mismatches; set ingester maps variant names → `hiddenpower`.
- **Ingest pitfall (fixed):** rows with only PK columns → empty `DO UPDATE SET`
  ("incomplete input"); `api.ingest` falls back to `DO NOTHING`.

### 3.3 Environment

- Node v24 (project-local `pokemon-showdown`), Python 3.12, SQLite built-in.
- Extract: `node legality/extract_gen9.js` (uses the fork's `Dex` — D2 ground truth).
- Sets source: `bot/data/movesets.json` (12 formats; gen9 targets: gen9ou 108 species /
  gen9uu 93 / gen9ru 93; **489 raw gen9 sets → 477 ingested**, 2,033 raw across all 12).

---

## 4. Legality model (two layers — core conceptual rule)

### T1 — Element legality (static; **the DB owns this**)
- **Form legal** ⇔ not explicitly banned ∧ `tier(form, gen)` not banned by format ∧ ≥ 1 legal ability (D8).
- **Move legal** ⇔ learnable in the gen ∧ not banned by format (D7, Fissure rule).
- **Ability legal** ⇔ exists for the form ∧ not banned. **Item legal** ⇔ not banned (format-level).

### T2 — Set-level context (**live validator stays arbiter**)
- Shiny-locked event moves (Entei Adamant), egg-move breeding chains, nature-locked
  distros, delearn constraints — no static table can encode these.
- **Bot impact:** we only play scraped Smogon sets + generated normal sets ⇒ T1 suffices;
  any team still passes the live validator before a battle.

---

## 5. Schema (v3 — see `schema.sql` for DDL)

```
── 1. universal entities ─────────────────────────────────────────────
pokemon(id PK, name, num)
form(id PK, name, pokemon_id, forme, is_mega, is_primal, is_gmax, is_nonstandard, reason)
move(id PK, name, type, category, power, accuracy, pp, priority, desc, side_condition,
     boosts, drain, recoil, ...)            -- side_condition drives is_hazard/is_screen/is_tailwind
move_gen(move_id, gen, ...)                 -- per-gen overrides (empty for now)
ability(id PK, name, desc)
item(id PK, name, desc)

── 2. per-gen attributes ─────────────────────────────────────────────
form_type(form_id, gen, type)               -- type = display name ("Fire")
form_stats(form_id, gen, hp, atk, def, spa, spd, spe)
form_tier(form_id, gen, tier, source)
form_ability(form_id, gen, ability_id, is_hidden)
type_chart(gen, attacking, defending, factor)   -- 18x18, factor in {0, 0.5, 1, 2} (D21–D22, D24)

── 3. learnset ───────────────────────────────────────────────────────
form_learnset(form_id, move_id, gen, sources)

── 4. format registry + T1 legality ─────────────────────────────────
format(id PK, name, gen, tier, game_type, min_team, max_team, default_level, ...)
format_form / format_form_ability / format_form_move / format_item / form_item

── 5. strategies (Smogon sets) ──────────────────────────────────────
strategy(id PK, format_id, form_id, title, source, source_url, scraped_at,
         item_id, ability_id,           -- D20
         nature, evs, ivs, tera_type, ...)
strategy_move(set_id, ord, move_id)

── 6. win rates (WIPEABLE, later) ── 7. provenance (ingest_log, meta)
```

**Counts (2026-11-07, `api.status()`):** pokemon 1120 · form 1517 · move 938 ·
ability 321 · item 587 · form_type 2374 · form_stats 1517 · form_tier 1517 ·
form_ability 3305 · form_learnset 65872 · **type_chart 324** · format 50 ·
format_form 3611 · format_form_ability 7949 · **format_form_move 201350** ·
format_item 986 · **strategy 477** · strategy_move 1893. `PRAGMA integrity_check` ok.

---

## 6. Current state (Step 1 — COMPLETE)

| Item | Status |
|------|--------|
| Extractor (`extract_gen9.js`) → JSON payload | ✅ gen9 × 4 formats, hidden-power + type-chart fixes in |
| Bootstrap + ingest (`build.py`, `api.ingest`) | ✅ schema v3, `DO NOTHING` fallback |
| Sets ingester (`ingest_movesets.py`) | ✅ 477 sets, per-format transactions, D27 filter, Hidden Power fallback |
| Lookup API (`lookup.py`) | ✅ `get_move` / `get_species` / `legal_moves` / `get_common_sets` / `get_type_chart` |
| **Tests (`tests/test_step1.py`)** | ✅ **21/21 green (2026-11-07)** |
| Docs (root PLAN.md / README.md / this sheet) | ✅ 2026-11-07 |

**Next:** STOP — per the one-step-at-a-time rule, **Step 2 (damage calculator wrapper)**
starts only when the user says go. Its gate: 15–20 hand-checked matchups vs the Smogon
calculator within rounding; screens/boosts/immunities; speed helper. The step 2 damage
math will index `get_type_chart()` directly (D26).

**Rebuild recipe** (only if extractor/data changes):
```
node legality/extract_gen9.js --gen 9 --formats gen9ou,gen9uu,gen9ru,gen9bss3v3 --out legality/data/extract_gen9.json
python -m legality.build bootstrap --force
python -c "from legality import api; api.ingest_file('legality/data/extract_gen9.json', source='showdown')"
python -m legality.ingest_movesets
python -m pytest legality/tests -v
```

---

## 7. Working conventions

1. **The fork's sim/data is the oracle** (D2, D29); the DB *materializes* it, never
   re-derives from memory of the official games.
2. **No invented clauses** — if the server allows it, the DB allows it.
3. **One ingest command** per source; all rows source-tagged; `ingest_log` + `meta`.
4. **Every gate logs evidence** here (counts, sample size, mismatch list = empty).
5. **DB keys:** entity ids are Showdown ids (lowercase); *type strings* in
   `move.type` / `form_type` / `get_type_chart()` are display names (D24).
6. `legality.db` is generated (git-ignored); code + schema + this sheet are the source.

## 8. File layout

```
legality/
├─ PLAN.md               ← this sheet
├─ schema.sql            ← single DDL source of truth (v3)
├─ formats.py            ← 50-format registry (49 + gen9bss3v3) + TABLE_ORDER
├─ db.py                 ← connection + bootstrap (SCHEMA_VERSION=3)
├─ api.py                ← status/reads/ingest (D16 protocol) + win-rate stubs
├─ ingest_protocol.md    ← payload contract
├─ build.py              ← `python -m legality.build bootstrap|status`
├─ extract_gen9.js       ← Node extractor (fork Dex → JSON payload)
├─ ingest_movesets.py    ← Smogon sets → strategy tables (D27)
├─ lookup.py             ← Step 1 API (the bot-facing surface)
├─ data/extract_gen9.json← extracted payload (regenerable)
├─ tests/test_step1.py   ← 21-test gate
└─ legality.db           ← generated (git-ignored)
```
