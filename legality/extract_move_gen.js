/**
 * extract_move_gen.js — dump PER-GEN move overrides as an ingest payload.
 *
 * The ``move`` table holds current-gen (gen 9) properties. Moves CHANGE across
 * generations — category (Dragon Claw is Special in gen 1-3, Physical gen 4+;
 * Ice moves were "special" in the type-based gen 1-2 split), priority (Protect
 * pri 2 -> 3 -> 4; Extreme Speed pri 1 -> 2), base power (Slash 70 -> 80 in
 * champions), type, accuracy, PP. The belief state reads category (the reflector
 * gate) and priority (the speed-order evidence); the Step-4 policy will read
 * base power / type. So move metadata must be per-gen, and it lives in the DB
 * (fast local lookup) — the user's call (2026-11-16, "the move data should live
 * in the database and change per generation").
 *
 * Source of truth: the FORK (Dex.forFormat bakes in the mod's overrides), which
 * is the raw-data authority (D37/D40) and the exact data the battle sim uses.
 * The @smogon/calc library carries the same per-gen category/priority (verified)
 * but the DB is our fast runtime store.
 *
 * KEY: ``move_gen.gen`` is the @smogon/calc ENGINE KEY (0 = champions, 1-9 =
 * gen) — the same key the damage calc and its JSON tables use, and what
 * formats.engine_key(fmt) returns. Move data is mechanics-scoped (champions
 * reuses gen-9 species but different moves), so it is keyed by mechanics, not
 * the raw generation.
 *
 * We emit an override row ONLY where a move's data DIFFERS from the gen-9 base
 * row. That is minimal, and it automatically excludes phantom rows: a move that
 * does not exist in an early gen carries base data (no diff -> no row).
 *
 * Usage (from the project root):
 *   node legality/extract_move_gen.js [--out legality/data/move_gen.json]
 * Then ingest (Python):
 *   python -c "from legality import api; api.ingest_file('legality/data/move_gen.json', source='showdown')"
 */
'use strict';
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const OUT = (() => {
  const i = process.argv.indexOf('--out');
  return (i >= 0 && process.argv[i + 1]) ? process.argv[i + 1]
    : path.join(__dirname, 'data', 'move_gen.json');
})();

const { Dex, toID } = require(path.join(ROOT, 'pokemon-showdown', 'dist', 'sim', 'index.js'));

// engine key (0 = champions, 1-9 = gen) -> a representative format whose mod
// carries that engine's move data. Data is identical for all formats of the same
// engine key; only rule tables differ (irrelevant to move properties).
const ENGINE_FMT = {
  0: 'gen9championsou', 1: 'gen1ou', 2: 'gen2ou', 3: 'gen3ou', 4: 'gen4ou',
  5: 'gen5ou', 6: 'gen6ou', 7: 'gen7ou', 8: 'gen8ou', 9: 'gen9ou',
};

// The data columns we track per gen. A1 (v5): type/category/power/accuracy/pp/
// priority. A2 (v6): the effect flags side_condition/side_status/boosts/drain/
// recoil — these DO change per gen in early gens (gen-1 Light Screen is a stat
// boost, not a side condition; Tail Glow +2 not +3; Double Edge 1/4 not 1/3
// recoil). desc is prose (not logic) so it stays excluded. ``boosts`` key order
// is normalized (sorted) so the "does it differ?" compare is order-stable.
const frac100 = (v) => {
  if (v == null) return null;
  if (Array.isArray(v) && v.length === 2 && v[1]) return Math.round(100 * v[0] / v[1]);
  if (typeof v === 'number') return v;
  return null;
};
const boostsStr = (m) => (m.boosts && Object.keys(m.boosts).length)
  ? JSON.stringify(m.boosts, Object.keys(m.boosts).sort()) : null;
const norm = (m) => ({
  type: m.type || null,
  category: m.category || null,
  power: m.basePower || null,
  accuracy: m.accuracy === true ? 100 : (m.accuracy == null ? null : m.accuracy),
  pp: m.pp == null ? null : m.pp,
  priority: m.priority == null ? 0 : m.priority,
  // A2 (v6): effect flags
  side_condition: m.sideCondition || null,
  side_status: m.sideStatus || null,
  boosts: boostsStr(m),
  drain: frac100(m.drain),
  recoil: frac100(m.recoil),
});
const same = (a, b) =>
  a.type === b.type && a.category === b.category && a.power === b.power &&
  a.accuracy === b.accuracy && a.pp === b.pp && a.priority === b.priority &&
  a.side_condition === b.side_condition && a.side_status === b.side_status &&
  a.boosts === b.boosts && a.drain === b.drain && a.recoil === b.recoil;

// gen-9 base values (the reference for "does this change per gen?").
const baseDex = Dex.forFormat(ENGINE_FMT[9]);
const base = new Map();
for (const m of baseDex.moves.all()) {
  if (!m || !m.num) continue;
  if (toID(m.name) !== m.id) continue; // Hidden Power <type> variants share an id
  base.set(m.id, norm(m));
}

const rows = [];
let emitted = 0, considered = 0, byGen = {};
for (const [key, fmt] of Object.entries(ENGINE_FMT)) {
  const dex = Dex.forFormat(fmt);
  for (const m of dex.moves.all()) {
    if (!m || !m.num) continue;
    if (toID(m.name) !== m.id) continue;
    considered++;
    const baseRow = base.get(m.id);
    if (!baseRow) continue; // not in the gen-9 base (shouldn't happen)
    const per = norm(m);
    if (same(per, baseRow)) continue; // no per-gen change -> the base row is correct
    rows.push({ table: 'move_gen', data: { move_id: m.id, gen: Number(key), ...per } });
    emitted++;
    byGen[key] = (byGen[key] || 0) + 1;
  }
}

const payload = {
  source: 'showdown',
  note: 'A1+A2: per-gen move overrides (type/category/power/accuracy/pp/priority ' +
        '+ side_condition/side_status/boosts/drain/recoil) from the fork, keyed by ' +
        'engine key (0=champions,1-9=gen); base row is gen 9',
  rows,
};
fs.mkdirSync(path.dirname(OUT), { recursive: true });
fs.writeFileSync(OUT, JSON.stringify(payload, null, 2));
console.log(`move_gen: ${emitted} override rows (of ${considered} move×gen considered)`);
console.log('  per engine key: ' + Object.keys(byGen).sort().map(k => `k${k}=${byGen[k]}`).join('  '));
console.log(`  -> ${OUT}`);
