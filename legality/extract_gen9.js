#!/usr/bin/env node
/**
 * extract_gen9.js — dump a generation's Showdown data as an ingest payload.
 *
 * Contract: ingest_protocol.md (source tag: "server"). Uses the *same* Dex the
 * local server runs (dist/sim), so the data is exactly what the server uses —
 * no drift, no client pings. All T1 legality verdicts come from the server's
 * own TeamValidator (checkSpecies / checkMove / checkItem / checkTagRules),
 * per ingest protocol rule 7.
 *
 * Usage (from the project root):
 *   node legality/extract_gen9.js \
 *     --gen 9 \
 *     --formats gen9ou,gen9uu,gen9ru,gen9bss3v3 \
 *     --out legality/data/extract_gen9.json
 *
 * Then ingest (Python):
 *   python -c "from legality import api; print(api.ingest_file('legality/data/extract_gen9.json'))"
 *
 * Notes
 * -----
 * - `move` holds CURRENT-gen properties. Gen 9 is the newest gen, so the base
 *   Dex is already gen-correct; for older gens pass a gen-N format so the mod
 *   applies (move_gen overrides are a later phase).
 * - form_learnset is gen-scoped: a move absent there is unavailable that gen.
 * - type_chart uses the root data chart (gen 9). factor = damage multiplier,
 *   0 = immune.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const SIM = path.join(ROOT, 'pokemon-showdown', 'dist', 'sim', 'index.js');

// ── CLI ──────────────────────────────────────────────────────────────
function arg(name, def) {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}
const GEN = parseInt(arg('--gen', '9'), 10);
const FORMATS = arg('--formats', 'gen9ou,gen9uu,gen9ru,gen9bss3v3')
  .split(',').map(s => s.trim()).filter(Boolean);
const OUT = arg('--out', path.join(__dirname, 'data', `extract_gen${GEN}.json`));

const {Dex, toID, TeamValidator} = require(SIM);

// Always go through a format: the base Dex is a gen-0 template; the modded
// dex (Dex.forFormat) carries the gen's data. Data is identical for all
// formats of the same gen — only rule tables differ (handled per-format below).
const fmt0 = Dex.formats.get(FORMATS[0]);
if (!fmt0 || fmt0.effectType !== 'Format') throw new Error(`unknown format: ${FORMATS[0]}`);
const dex = Dex.forFormat(FORMATS[0]);
if (dex.gen !== GEN) throw new Error(`expected gen ${GEN} dex, got gen ${dex.gen}`);

const rows = [];
const add = (table, data) => rows.push({table, data});
const t0 = Date.now();
/** [num, denom] fraction (or plain number) -> integer percent, or null. */
const frac100 = (v) => {
  if (v == null) return null;
  if (Array.isArray(v) && v.length === 2 && v[1]) return Math.round(100 * v[0] / v[1]);
  if (typeof v === 'number') return v;
  return null;
};

// NB: in this Showdown build, *.all() returns EFFECT OBJECTS, not id strings.
// Iterate the objects and use `.id` for all id-keyed lookups/sets.

// ── 1. pokemon + form ────────────────────────────────────────────────
const baseSeen = new Set();
for (const sp of dex.species.all()) {
  if (!sp || !sp.num) continue;
  const spId = sp.id;
  const baseId = toID(sp.baseSpecies || sp.name);
  const base = dex.species.get(baseId);
  if (!baseSeen.has(baseId)) {
    baseSeen.add(baseId);
    add('pokemon', {id: baseId, name: base.name, num: base.num});
  }
  const ns = sp.isNonstandard;
  add('form', {
    id: spId, name: sp.name, pokemon_id: baseId,
    forme: sp.forme || null,
    is_mega: sp.isMega ? 1 : 0,
    is_primal: sp.isPrimal ? 1 : 0,
    is_gmax: sp.canGigantamax ? 1 : 0,
    is_nonstandard: typeof ns === 'string' ? ns : (ns ? 'true' : null),
    nonstandard_reason: null,
  });
}

// ── 2. moves (current-gen properties + plain-data effect flags) ──────
for (const m of dex.moves.all()) {
  if (!m || !m.num) continue;
  // Generated variants can share a base id (in this build all 16
  // 'Hidden Power <type>' variants have id 'hiddenpower'). Keep the base
  // entry only — variants would collide in the move table's PK.
  if (toID(m.name) !== m.id) continue;
  add('move', {
    id: m.id, name: m.name,
    type: m.type || null,
    category: m.category || null,
    power: m.basePower || null,  // status moves have basePower 0 -> NULL (protocol §5: null = not set)
    accuracy: m.accuracy === true ? 100 : (m.accuracy == null ? null : m.accuracy),
    pp: m.pp == null ? null : m.pp,
    priority: m.priority == null ? 0 : m.priority,
    desc: m.desc || null,
    side_condition: m.sideCondition || null,
    side_status: m.sideStatus || null,
    boosts: (m.boosts && Object.keys(m.boosts).length) ? JSON.stringify(m.boosts) : null,
    // drain/recoil are [num, denom] fractions in this dataset (Absorb = [1, 2])
    drain: frac100(m.drain),
    recoil: frac100(m.recoil),
  });
}

// ── 3. abilities + items ─────────────────────────────────────────────
for (const a of dex.abilities.all()) {
  if (!a || !a.name) continue;
  add('ability', {id: a.id, name: a.name, desc: a.desc || null});
}
for (const it of dex.items.all()) {
  if (!it || !it.name) continue;
  add('item', {id: it.id, name: it.name, desc: it.desc || null});
}

// ── 4. per-gen attributes + gen-scoped learnsets ─────────────────────
const learnsetCache = new Map(); // formId -> {moveId: [code...]}
// Real (current-gen) moves only — a species can only be credited with a move
// that actually exists in this gen's move table (prevents orphan move ids).
const realMoves = new Set();
for (const m of dex.moves.all()) { if (m && m.id) realMoves.add(m.id); }
for (const sp of dex.species.all()) {
  if (!sp || !sp.num) continue;
  const spId = sp.id;
  for (const t of (sp.types || [])) {
    add('form_type', {form_id: spId, gen: GEN, type: t});
  }
  const bs = sp.baseStats || {};
  add('form_stats', {
    form_id: spId, gen: GEN,
    hp: bs.hp ?? null, atk: bs.atk ?? null, def: bs.def ?? null,
    spa: bs.spa ?? null, spd: bs.spd ?? null, spe: bs.spe ?? null,
  });
  if (sp.tier) add('form_tier', {form_id: spId, gen: GEN, tier: sp.tier, source: 'server'});
  for (const [slot, aName] of Object.entries(sp.abilities || {})) {
    const aId = toID(aName);
    if (!aId) continue;
    add('form_ability', {form_id: spId, gen: GEN, ability_id: aId, is_hidden: slot === 'H' ? 1 : 0});
  }

  // Learnset: union over base+forme parts. A move is usable in THIS gen if the
  // species can learn it in this gen OR any earlier gen >= 3 — moves carry
  // forward via transfer/relearner. This mirrors the fork's own checkCanLearn
  // (noFutureGen caps sources at dex.gen; minSourceGen floors them at 3). The
  // old "c[0] === GEN" filter was too strict: it dropped ~399 species whose
  // learnset data in this fork caps at gen 8 (e.g. machamp, abra), leaving them
  // with zero legal moves and breaking build_prior.
  // Code shape: "9M" (gen 9, method M), "5S0" (gen 5 event, level 0), "4-" = illegal.
  const learn = {};
  try {
    for (const part of dex.species.getFullLearnset(spId)) {
      for (const [mId, codes] of Object.entries(part.learnset || {})) {
        const okCodes = (codes || []).filter(
          c => typeof c === 'string' && c[1] !== '-'
            && parseInt(c[0], 10) >= 3 && parseInt(c[0], 10) <= GEN);
        if (okCodes.length && realMoves.has(mId)) {
          learn[mId] = (learn[mId] || []).concat(okCodes);
        }
      }
    }
  } catch (e) {
    console.error(`warn: getFullLearnset(${spId}) failed: ${e.message}`);
  }
  learnsetCache.set(spId, learn);
  for (const [mId, codes] of Object.entries(learn)) {
    add('form_learnset', {form_id: spId, move_id: mId, gen: GEN, sources: codes.join(',')});
  }
}

// ── 5. formats + T1 legality (validator verdicts) ───────────────────
const TIER = {gen9ou: 'OU', gen9uu: 'UU', gen9ru: 'RU', gen9bss3v3: 'BSS'};
for (const fmtId of FORMATS) {
  const fmt = Dex.formats.get(fmtId);
  if (!fmt || fmt.effectType !== 'Format') throw new Error(`unknown format: ${fmtId}`);
  const tv = new TeamValidator(fmtId);
  const rt = tv.ruleTable;

  // D33: default_level must equal the format's BATTLE level, exactly as
  // test_step1.test_format_levels_match_fork re-derives it (start at
  // defaultLevel, then apply adjustLevelDown, then adjustLevel). The old
  // `rt.maxLevel` was wrong for formats like BSS (Adjust Level Down 50 →
  // battle level 50, not maxLevel 100) and clobbered the D33 rule on rebuild.
  let battleLevel = rt.defaultLevel;
  if (rt.adjustLevelDown && battleLevel >= rt.adjustLevelDown) battleLevel = rt.adjustLevelDown;
  if (rt.adjustLevel) battleLevel = rt.adjustLevel;
  if (typeof battleLevel !== 'number') battleLevel = null;

  add('format', {
    id: fmtId, name: fmt.name, gen: tv.gen, tier: TIER[fmtId] || null,
    game_type: /doubles/i.test(fmt.name) ? 'doubles' : 'singles',
    min_team: fmt.minTeam ?? null,
    max_team: fmt.maxTeam ?? null,
    picked_team: rt.pickedTeamSize == null ? null : rt.pickedTeamSize,
    default_level: battleLevel,
    section: TIER[fmtId] === 'BSS' ? 'bss' : 'ladder',
    mod: fmt.mod || null,
    exists_in_server: 1,
  });

  // Legal elements per format — computed once, then joined with the
  // capability tables (D5: legality tables store memberships only).
  const legalMoves = new Set();
  for (const m of dex.moves.all()) {
    if (!m || !m.num) continue;
    if (!tv.checkMove({}, m, {})) legalMoves.add(m.id);
  }
  const legalAbilities = new Set();
  for (const a of dex.abilities.all()) {
    if (!a || !a.name) continue;
    if (rt.check('ability:' + a.id)) continue;         // explicit ban ('' = whitelisted = ok)
    if (tv.checkTagRules({}, a, {})) continue;         // tag rules
    legalAbilities.add(a.id);
  }
  const legalItems = new Set();
  for (const it of dex.items.all()) {
    if (!it || !it.name || it.isNonstandard) continue;
    if (!tv.checkItem({}, it, {})) legalItems.add(it.id);
  }
  for (const iId of legalItems) add('format_item', {format_id: fmtId, item_id: iId});

  for (const sp of dex.species.all()) {
    if (!sp || !sp.num) continue;
    const spId = sp.id;
    const base = dex.species.get(toID(sp.baseSpecies || sp.name));
    let problem;
    try { problem = tv.checkSpecies({}, sp, base, {}); }
    catch (e) { problem = `error: ${e.message}`; }
    // D47 (Option B): this fork marks ~454 REAL species `isNonstandard: "Past"`
    // (tier "Illegal") purely because their learnset data caps at gen 8 — a data
    // artifact, not a ban. The fork's own TeamValidator.validateTeam ACCEPTS them
    // (BSS "Flat Rules" allows all species), so the "Past" tag is internally
    // inconsistent. Allow a real species (valid num, exists, not a placeholder)
    // that is blocked ONLY by the "Past" tag. CAP / LGPE / Legends / Unobtainable
    // are NOT "Past", so they stay rejected (D8) — this override cannot un-ban them.
    // Base form only: base species have forme="" (falsy); Mega/Primal/Gmax
    // formes carry forme="Mega"/"Mega-X"/"Primal"/"Gmax" (and gmax also set
    // placeholderFor). We do NOT want to un-ban those formes into BSS.
    const pastArtifact = !!(problem
      && sp.isNonstandard === 'Past'
      && sp.num
      && sp.exists
      && !sp.placeholderFor
      && !sp.forme);
    if (problem && !pastArtifact) continue; // validator says no → not in format_form (D8)
    add('format_form', {format_id: fmtId, form_id: spId});

    for (const [slot, aName] of Object.entries(sp.abilities || {})) {
      const aId = toID(aName);
      if (aId && legalAbilities.has(aId)) {
        add('format_form_ability', {format_id: fmtId, form_id: spId, ability_id: aId});
      }
    }
    // Fissure rule (D7): learnable in the gen AND not format-banned.
    for (const mId of Object.keys(learnsetCache.get(spId) || {})) {
      if (legalMoves.has(mId)) {
        add('format_form_move', {format_id: fmtId, form_id: spId, move_id: mId});
      }
    }
  }
}

// ── 6. type chart (root data chart; gen 9) ──────────────────────────
// Standard 18-type matrix only. The compiled chart object also carries a
// 'stellar' pseudo-row (gen-9 tera) — excluded on purpose: the lookup layer
// wants the canonical matchup table (18×18 = 324 cells, one row per pair).
try {
  const chartMod = require(path.join(ROOT, 'pokemon-showdown', 'dist', 'data', 'typechart.js'));
  const TypeChart = chartMod.TypeChart || (chartMod.default && chartMod.default.TypeChart);
  const REAL_TYPES = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting',
    'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon',
    'dark', 'steel', 'fairy'];
  if (TypeChart) {
    for (const defType of REAL_TYPES) {
      if (!TypeChart[toID(defType)]) {
        console.error(`warn: type '${defType}' missing from TypeChart — skipping row`);
        continue;
      }
      // damageTaken keys are display names ('Water', 'Fighting', ...) — rekey by id.
      const takenRaw = TypeChart[toID(defType)].damageTaken || {};
      const taken = {};
      for (const k of Object.keys(takenRaw)) taken[toID(k)] = takenRaw[k];
      for (const attType of REAL_TYPES) {
        // Encoding (verified against sim/dex.ts getEffectiveness/getImmunity):
        //   0 = neutral (1x) · 1 = super-effective (2x) · 2 = resist (0.5x) · 3 = immune (0x)
        const v = taken[attType];
        const factor = (v == null) ? 1 : [1, 2, 0.5, 0][v];
        add('type_chart', {attacking: toID(attType), defending: toID(defType), factor, gen: GEN});
      }
    }
  } else {
    console.error('warn: TypeChart not found in dist/data/typechart.js');
  }
} catch (e) {
  console.error(`warn: type chart extraction failed: ${e.message}`);
}

// ── out ──────────────────────────────────────────────────────────────
fs.mkdirSync(path.dirname(OUT), {recursive: true});
const payload = {
  source: 'server',
  note: `extract gen${GEN} formats=[${FORMATS.join(',')}] at ${new Date().toISOString()}`,
  rows,
};
fs.writeFileSync(OUT, JSON.stringify(payload));
const byTable = {};
for (const r of rows) byTable[r.table] = (byTable[r.table] || 0) + 1;
console.log(`rows: ${rows.length} in ${Date.now() - t0} ms -> ${OUT}`);
for (const [t, n] of Object.entries(byTable).sort()) console.log(`  ${t.padEnd(24)} ${n}`);