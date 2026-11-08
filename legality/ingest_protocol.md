# Ingest protocol (D16) — how data enters `legality.db`

> **Single payload contract.** Every source — the server-extract (Step 1),
> the Smogon sets ingester (Step 1), anything else — emits the *same* JSON payload.
> Language-agnostic: any tool can write the JSON; `api.ingest` /
> `api.ingest_file` performs the upsert in one transaction.
>
> Rules live in PLAN.md §7 (conventions). This document is the normative
> data contract. If code and this document disagree, fix the code.

---

## 1. Payload shape

```json
{
  "source": "server",
  "note":   "legality extract — gen1ou sample",
  "rows":   [
    { "table": "pokemon",        "data": { "id": "onix", "name": "Onix", "num": 95 } },
    { "table": "form",           "data": { "id": "onix", "name": "Onix", "pokemon_id": "onix" } },
    { "table": "form_tier",      "data": { "form_id": "onix", "gen": 1, "tier": "OU" } },
    { "table": "format_form",    "data": { "format_id": "gen1ou", "form_id": "onix" } }
  ]
}
```

| key      | type   | required | meaning |
|----------|--------|----------|---------|
| `source` | string | yes (here or in envelope) | one of the **source tags** (§3) |
| `note`   | string | no       | free-text batch description; copied to `ingest_log` |
| `rows`   | list   | yes, non-empty | `[{table, data}]` — one row each |

`source` may be passed to `api.ingest(payload, source=...)` instead; the
argument wins. `meta` and `ingest_log` are **not** ingestable (protected).

## 2. Rules

1. **One transaction.** All rows commit or none do (rollback on any error).
2. **Upsert semantics.** `INSERT … ON CONFLICT(<PK>) DO UPDATE` on the table's
   primary key (§4). Re-running a batch is idempotent — same data ⇒ same DB.
3. **FK order is enforced for you.** Rows are sorted against
   `formats.TABLE_ORDER` before writing (parents before children). Still,
   write parents first in the payload — it reads better in `ingest_log`.
4. **Strict validation.** Unknown table, unknown column, or a missing PK
   column ⇒ the whole batch is rejected (nothing written, exception raised).
5. **Referential integrity.** `PRAGMA foreign_keys=ON`. A row referencing a
   `pokemon`/`form`/`move`/`item`/`format`/`set` id that doesn't exist ⇒
   batch rejected. Ingest the parent row in the same (or an earlier) batch.
6. **Provenance.** Each table in the batch gets one `ingest_log` row
   (ts, source, table, n_rows, note). Wipes are logged the same way.
7. **No derived legality.** Legality tables (§4.4) must contain the
   *validator's verdicts* (PLAN.md §7 rule 1) — never tier arithmetic done
   ad hoc in a scraper.
8. **Win rates** are normally bumped by `api.record_battle`, not ingested.
   Direct `win_form`/`win_move` payloads are allowed only for bulk
   import/export of historical stats.

## 3. Source tags

| tag      | meaning |
|----------|---------|
| `server` | extracted from the local Showdown repo (`pokemon-showdown/`) — identity, per-gen attributes, learnsets, T1 legality (Step 1) |
| `smogon` | scraped from smogon.com — strategies/sets (Step 1), optionally reference text |
| `net`    | any other web/manual source |
| `reset`  | (written by the DB itself) wipe/audit events in `ingest_log` |

## 4. Table catalog (PKs + required columns)

### 4.1 Universal entities

| table | PK | required | notes |
|---|---|---|---|
| `pokemon` | `id` | `id`, `name` | base species; `id` = showdown `toID`; `num` = dex number |
| `form` | `id` | `id`, `name` | forme or base; `pokemon_id` → `pokemon.id`; flags `is_mega/is_primal/is_gmax` (0/1), `is_nonstandard` (e.g. `"Past"`, `null` = standard) |
| `move` | `id` | `id`, `name` | current-gen properties in `type/category/power/accuracy/pp/priority/desc` |
| `move_gen` | `move_id, gen` | `move_id`, `gen` | per-gen property overrides (only rows that differ) |
| `ability` | `id` | `id`, `name` | |
| `item` | `id` | `id`, `name` | |

### 4.2 Per-gen attributes (Smogon parity)

| table | PK | required | notes |
|---|---|---|---|
| `form_type` | `form_id, gen, type` | all three | one row per type slot (dual types ⇒ 2 rows) |
| `form_stats` | `form_id, gen` | `form_id`, `gen` | `hp atk def spa spd spe` |
| `form_tier` | `form_id, gen` | `form_id`, `gen`, `tier` | e.g. `OU`, `UUBL`, `NFE`, `Uber`; `source` defaults `server` |
| `form_ability` | `form_id, gen, ability_id` | all three | `is_hidden` 0/1 (Hidden Ability ⇒ 1) |

### 4.3 Learnset (gen-scoped)

| table | PK | required | notes |
|---|---|---|---|
| `form_learnset` | `form_id, move_id, gen` | all three, `sources` | **absent in a gen = move unavailable that gen** ("gen 3 switch" behavior). `sources` = raw showdown codes, comma-joined (e.g. `9M,8L1,6E`) — these are also the **T2 hints** (event `S`, egg `E`, …) |
| `form_learnset_constraints` | `form_id, move_id, gen` | all three, `note` | free-text T2 notes (e.g. "event-only, nature-locked") |

### 4.4 Formats + T1 legality (memberships only — D5)

| table | PK | required | notes |
|---|---|---|---|
| `format` | `id` | `id`, `name`, `gen` | pre-filled at bootstrap from the 50-format registry; ingest updates `min_team/max_team/default_level` from the ruleTable at legality-ingest time |
| `format_form` | `format_id, form_id` | both | **legal species/forms** (D8: not banned ∧ tier allowed ∧ ≥1 legal ability) |
| `format_form_ability` | `format_id, form_id, ability_id` | all three | legal abilities for that form in that format |
| `format_form_move` | `format_id, form_id, move_id` | all three | **Fissure rule (D7):** learnable in the gen ∧ not format-banned |
| `format_item` | `format_id, item_id` | both | format-legal items |
| `form_item` | `form_id, item_id` | both | species-locked item grants (not yet built) |

### 4.5 Strategies (scraped sets — Step 1, done)

> The tables are named `strategy` / `strategy_move` because `set` is a SQL
> keyword; the `set_id` column name (and `api.sets()`) is kept — in Showdown
> terms the object is still a *set*.

| table | PK | required | notes |
|---|---|---|---|
| `strategy` | `id` | `id`, `format_id`, `form_id` | `id` = stable slug (e.g. `smogon-gen9uu-aerodactyl-1`); `source_url` **required** for `smogon` rows; `evs`/`ivs` = raw strings (`"252 Atk / 4 Def / 252 Spe"`); `scraped_at` = ISO-8601 UTC |
| `strategy_move` | `set_id, ord` | `set_id`, `ord`, `move_id` | `ord` = 1..n, order as listed on the page |

### 4.6 Win rates (wipeable)

| table | PK | required | notes |
|---|---|---|---|
| `win_form` | `format_id, form_id` | both | usually via `api.record_battle` |
| `win_move` | `format_id, form_id, move_id` | all three | usually via `api.record_battle` |

`api.wipe_win_rates(fmt|None)` deletes **only** these two tables (D11).

## 5. Conventions (id/encoding)

- **ids** = Showdown `toID` (lowercase, no spaces/hyphens-as-needed): `garchomp`,
  `rotom-wash`, `fissure`, `leftovers`.
- **gen** = integer 1–9.
- **booleans** = integers 0/1.
- **timestamps** = ISO-8601 UTC (`2026-11-01T12:00:00+00:00`).
- **tier** = exact tag string from the server (`OU`, `UUBL`, `NFE`, `Uber`, …).
- `null` (JSON) = column stays NULL / "not set" — never use `""` for absence.

## 6. Examples

### 6.1 Server legality batch

```json
{
  "source": "server",
  "note": "legality — gen1ou sample",
  "rows": [
    { "table": "pokemon",  "data": { "id": "onix", "name": "Onix", "num": 95 } },
    { "table": "form",     "data": { "id": "onix", "name": "Onix", "pokemon_id": "onix" } },
    { "table": "form_tier","data": { "form_id": "onix", "gen": 1, "tier": "OU" } },
    { "table": "ability",  "data": { "id": "rockhead", "name": "Rock Head" } },
    { "table": "move",     "data": { "id": "thundershock", "name": "Thunder Shock" } },
    { "table": "format_form",          "data": { "format_id": "gen1ou", "form_id": "onix" } },
    { "table": "format_form_ability",  "data": { "format_id": "gen1ou", "form_id": "onix", "ability_id": "rockhead" } },
    { "table": "format_form_move",     "data": { "format_id": "gen1ou", "form_id": "onix", "move_id": "thundershock" } }
  ]
}
```

### 6.2 Smogon strategy batch

```json
{
  "source": "smogon",
  "note": "Smogon Gen 9 UU Aerodactyl strategies",
  "rows": [
    { "table": "strategy", "data": {
        "id": "smogon-gen9uu-aerodactyl-1",
        "format_id": "gen9uu", "form_id": "aerodactyl",
        "title": "Choice Band", "source": "smogon",
        "source_url": "https://www.smogon.com/dex/uu/pokemon/aerodactyl",
        "scraped_at": "2026-11-01T12:00:00+00:00",
        "item_id": "choicescarf", "nature": "Naive",
        "evs": "252 Atk / 4 Def / 252 Spe", "notes": "main set" } },
    { "table": "strategy_move", "data": { "set_id": "smogon-gen9uu-aerodactyl-1", "ord": 1, "move_id": "crunch" } },
    { "table": "strategy_move", "data": { "set_id": "smogon-gen9uu-aerodactyl-1", "ord": 2, "move_id": "stoneedge" } },
    { "table": "strategy_move", "data": { "set_id": "smogon-gen9uu-aerodactyl-1", "ord": 3, "move_id": "earthquake" } },
    { "table": "strategy_move", "data": { "set_id": "smogon-gen9uu-aerodactyl-1", "ord": 4, "move_id": "fly" } }
  ]
}
```

## 7. Verification (after every batch)

1. `api.ingest` returns per-table counts — check they match what was emitted.
2. `SELECT … FROM ingest_log ORDER BY id DESC LIMIT 10` — one row per table.
3. `PRAGMA foreign_key_check;` — must be empty.
4. Spot-read via the API (`api.legal_forms`, `api.tier`, `api.sets`, …).
