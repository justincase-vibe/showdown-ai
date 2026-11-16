-- ============================================================
-- legality.db — schema v3 (single DDL source of truth)
-- See PLAN.md §5 for the design rationale.
-- Sections: 1 entities · 2 per-gen attrs · 3 learnset · 3b type chart ·
--           4 format+legality · 5 strategies · 6 win rates · 7 provenance
-- v2 (2026-11-07): move effect flags (side_condition/side_status/boosts/
--   drain/recoil), format.picked_team, type_chart table.
-- v3 (2026-11-07): strategy.ability_id — Smogon sets always list the
--   ability; the AI needs it for team generation (D20).
-- ============================================================

PRAGMA foreign_keys = ON;

-- ── 1. universal entities ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS pokemon (
  id   TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  num  INTEGER
);

CREATE TABLE IF NOT EXISTS form (
  id                 TEXT PRIMARY KEY,
  name               TEXT NOT NULL,
  pokemon_id         TEXT NOT NULL REFERENCES pokemon(id),
  forme              TEXT,
  is_mega            INTEGER NOT NULL DEFAULT 0,
  is_primal          INTEGER NOT NULL DEFAULT 0,
  is_gmax            INTEGER NOT NULL DEFAULT 0,
  is_nonstandard     TEXT,
  nonstandard_reason TEXT
);
CREATE INDEX IF NOT EXISTS idx_form_pokemon ON form(pokemon_id);

CREATE TABLE IF NOT EXISTS move (
  id         TEXT PRIMARY KEY,
  name       TEXT NOT NULL,
  type       TEXT,
  category   TEXT,
  power      INTEGER,
  accuracy   INTEGER,
  pp         INTEGER,
  priority   INTEGER,
  desc       TEXT,
  -- v2: plain-data effect flags from the server (NULL = not present)
  side_condition TEXT,   -- e.g. stealthrock/spikes/stickybarbedsteel (hazards),
                         --      reflect/lightscreen (screens), tailwind
  side_status    TEXT,   -- side-status move, if any (e.g. aquaring)
  boosts         TEXT,   -- JSON object e.g. {"atk":2} / {"atk":-1}; NULL if none
  drain          INTEGER,-- heal/recoil fraction x100 (50 = 50%); NULL if none
  recoil         INTEGER -- recoil fraction x100; NULL if none
);

-- v5 (A1) / v6 (A2): per-gen move overrides. Its ``gen`` column is the
-- @smogon/calc ENGINE KEY (0 = champions, 1-9 = gen), NOT the raw generation:
-- move data is mechanics-scoped (champions reuses gen-9 species but different
-- moves — e.g. Slash bp 70 -> 80), matching formats.engine_key() and the calc's
-- per-engine-key tables (special_bp_moves.json / recoil_drain_moves.json).
-- Populated per engine key from the fork (extract_move_gen.js) ONLY where a move
-- DIFFERS from the gen-9 base row. Tracked columns:
--   A1 (v5): type / category / power / accuracy / pp / priority
--   A2 (v6): side_condition / side_status / boosts / drain / recoil  (effect flags)
-- lookup.get_move(name, gen) merges the override onto the gen-9 ``move`` row
-- (the base row is the fallback). The calc models recoil/drain per-gen via its
-- own JSON tables; the DB flags here serve the public API + team-building.
CREATE TABLE IF NOT EXISTS move_gen (
  move_id        TEXT NOT NULL REFERENCES move(id),
  gen            INTEGER NOT NULL,
  type           TEXT,
  category       TEXT,
  power          INTEGER,
  accuracy       INTEGER,
  pp             INTEGER,
  priority       INTEGER,
  desc           TEXT,
  -- A2 (v6): per-gen effect flags (override the gen-9 base where they differ —
  -- e.g. gen-1 Light Screen is a stat boost, NOT a side condition; Tail Glow +2)
  side_condition TEXT,
  side_status    TEXT,
  boosts         TEXT,
  drain          INTEGER,
  recoil         INTEGER,
  PRIMARY KEY (move_id, gen)
);

CREATE TABLE IF NOT EXISTS ability (
  id   TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  desc TEXT
);

CREATE TABLE IF NOT EXISTS item (
  id   TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  desc TEXT
);

-- ── 2. per-gen attributes (Smogon parity) ───────────────────────────
CREATE TABLE IF NOT EXISTS form_type (
  form_id   TEXT NOT NULL REFERENCES form(id),
  gen       INTEGER NOT NULL,
  type      TEXT NOT NULL,
  PRIMARY KEY (form_id, gen, type)
);

CREATE TABLE IF NOT EXISTS form_stats (
  form_id TEXT NOT NULL REFERENCES form(id),
  gen     INTEGER NOT NULL,
  hp      INTEGER,
  atk     INTEGER,
  def     INTEGER,
  spa     INTEGER,
  spd     INTEGER,
  spe     INTEGER,
  PRIMARY KEY (form_id, gen)
);

CREATE TABLE IF NOT EXISTS form_tier (
  form_id TEXT NOT NULL REFERENCES form(id),
  gen     INTEGER NOT NULL,
  tier    TEXT NOT NULL,
  source  TEXT NOT NULL DEFAULT 'server',
  PRIMARY KEY (form_id, gen)
);

CREATE TABLE IF NOT EXISTS form_ability (
  form_id    TEXT NOT NULL REFERENCES form(id),
  gen        INTEGER NOT NULL,
  ability_id TEXT NOT NULL REFERENCES ability(id),
  is_hidden  INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (form_id, gen, ability_id)
);

-- ── 3. learnset (gen-scoped; T2 hints in sources) ──────────────────
CREATE TABLE IF NOT EXISTS form_learnset (
  form_id TEXT NOT NULL REFERENCES form(id),
  move_id TEXT NOT NULL REFERENCES move(id),
  gen     INTEGER NOT NULL,
  sources TEXT NOT NULL,
  PRIMARY KEY (form_id, move_id, gen)
);
CREATE INDEX IF NOT EXISTS idx_fl_move ON form_learnset(move_id);

CREATE TABLE IF NOT EXISTS form_learnset_constraints (
  form_id TEXT NOT NULL,
  move_id TEXT NOT NULL,
  gen     INTEGER NOT NULL,
  note    TEXT NOT NULL,
  PRIMARY KEY (form_id, move_id, gen)
);

-- ── 3b. type chart (per gen; factor 0 = immune) ────────────────────
CREATE TABLE IF NOT EXISTS type_chart (
  attacking TEXT NOT NULL,
  defending TEXT NOT NULL,
  factor    REAL NOT NULL,
  gen       INTEGER NOT NULL,
  PRIMARY KEY (attacking, defending, gen)
);
CREATE INDEX IF NOT EXISTS idx_tc_defending ON type_chart(defending, gen);

-- ── 4. format registry + T1 legality ───────────────────────────────
CREATE TABLE IF NOT EXISTS format (
  id               TEXT PRIMARY KEY,
  name             TEXT NOT NULL,
  gen              INTEGER NOT NULL,
  tier             TEXT,
  game_type        TEXT NOT NULL DEFAULT 'singles',
  min_team         INTEGER,
  max_team         INTEGER,
  picked_team      INTEGER,   -- v2: team-preview pick size (e.g. 3 for BSS); NULL = all
  -- v4 (D33): level rules from the fork's ruleTable (team-validator ground truth).
  -- default_level = CANONICAL BATTLE LEVEL for a standard team (ruleTable
  -- defaultLevel after adjustLevelDown/adjustLevel), NOT the raw defaultLevel:
  -- BSS 3v3 plays level 50 (Flat Rules = Adjust Level Down 50), VGC 50, LC 5,
  -- ladder 100, champions 50 (the @smogon/calc champions engine hard-codes 50).
  min_level        INTEGER,
  max_level        INTEGER,
  adjust_level     INTEGER,   -- exact level forced by the format (e.g. champions 50, gen1lc 5)
  adjust_level_down INTEGER,  -- levels at/above this are lowered to it (e.g. BSS/VGC 50)
  default_level    INTEGER,
  section          TEXT,
  mod              TEXT,
  exists_in_server INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS format_form (
  format_id TEXT NOT NULL REFERENCES format(id),
  form_id   TEXT NOT NULL REFERENCES form(id),
  PRIMARY KEY (format_id, form_id)
);
CREATE INDEX IF NOT EXISTS idx_ff_form ON format_form(form_id);

CREATE TABLE IF NOT EXISTS format_form_ability (
  format_id  TEXT NOT NULL,
  form_id    TEXT NOT NULL,
  ability_id TEXT NOT NULL,
  PRIMARY KEY (format_id, form_id, ability_id)
);

CREATE TABLE IF NOT EXISTS format_form_move (
  format_id TEXT NOT NULL,
  form_id   TEXT NOT NULL,
  move_id   TEXT NOT NULL,
  PRIMARY KEY (format_id, form_id, move_id)
);
CREATE INDEX IF NOT EXISTS idx_fmf_move ON format_form_move(move_id);

CREATE TABLE IF NOT EXISTS format_item (
  format_id TEXT NOT NULL,
  item_id   TEXT NOT NULL,
  PRIMARY KEY (format_id, item_id)
);

CREATE TABLE IF NOT EXISTS form_item (
  form_id TEXT NOT NULL,
  item_id TEXT NOT NULL,
  PRIMARY KEY (form_id, item_id)
);

-- ── 5. strategies (scraped sets) ────────────────────────────────────
-- NB: named `strategy` because SET is a SQL keyword (breaks bare table refs).
CREATE TABLE IF NOT EXISTS strategy (
  id          TEXT PRIMARY KEY,
  format_id   TEXT NOT NULL REFERENCES format(id),
  form_id     TEXT NOT NULL REFERENCES form(id),
  title       TEXT,
  source      TEXT,
  source_url  TEXT,
  scraped_at  TEXT,
  item_id     TEXT,
  ability_id  TEXT,   -- v3: the set's ability (Smogon sets always list one)
  nature      TEXT,
  evs         TEXT,
  ivs         TEXT,
  gender      TEXT,
  tera_type   TEXT,
  notes       TEXT
);
CREATE INDEX IF NOT EXISTS idx_strategy_form ON strategy(format_id, form_id);

CREATE TABLE IF NOT EXISTS strategy_move (
  set_id    TEXT NOT NULL REFERENCES strategy(id),
  ord     INTEGER NOT NULL,
  move_id TEXT NOT NULL REFERENCES move(id),
  PRIMARY KEY (set_id, ord)
);

-- ── 6. win rates (WIPEABLE — never touch T1 on reset) ──────────────
CREATE TABLE IF NOT EXISTS win_form (
  format_id     TEXT NOT NULL,
  form_id       TEXT NOT NULL,
  battles       INTEGER NOT NULL DEFAULT 0,
  wins          INTEGER NOT NULL DEFAULT 0,
  last_updated  TEXT,
  PRIMARY KEY (format_id, form_id)
);

CREATE TABLE IF NOT EXISTS win_move (
  format_id TEXT NOT NULL,
  form_id   TEXT NOT NULL,
  move_id   TEXT NOT NULL,
  uses      INTEGER NOT NULL DEFAULT 0,
  wins      INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (format_id, form_id, move_id)
);

-- ── 7. provenance ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ingest_log (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  ts         TEXT NOT NULL,
  source     TEXT NOT NULL,
  table_name TEXT NOT NULL,
  n_rows     INTEGER NOT NULL,
  note       TEXT
);

CREATE TABLE IF NOT EXISTS meta (
  key   TEXT PRIMARY KEY,
  value TEXT
);
