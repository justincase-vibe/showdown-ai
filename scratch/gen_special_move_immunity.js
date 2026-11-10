'use strict';
// Precompute fork-exact type-immunity facts for the special-damage (non-basePower)
// moves, per engine key, from the fork's own dex data (source of truth for raw
// game data: type chart + move type/ignoreImmunity/ohko).
//
// The fork's getDamage dispatch (battle-actions.js) runs target.runImmunity(move)
// for EVERY move before any formula. runImmunity (pokemon.js:1664):
//   * move.ignoreImmunity === true (or per-type)  -> NOT immune
//   * type "Ground" -> grounded check (Flying type / Levitate / Eelevate / Iron Ball)
//   * else dex.getImmunity(move.type, defenderTypes)  (chart damageTaken === 3)
// So for a static (gen, move) lookup we need: the move's type, its ignoreImmunity
// flag, and the chart's 0x rows. Ground's ability/item nuances stay in node_calc.js
// (defender ability/item are request data). sheercold's "ohko: Ice" immunity is
// NOT a chart immunity (Ice resists Ice 0.5x) — emitted as its own flag.
const path = require('path');
const fs = require('fs');
const Sim = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));

const MOVES = [
  'superfang', 'ruination', 'naturesmadness', 'guardianofalola', 'finalgambit',
  'endeavor', 'psywave', 'counter', 'mirrorcoat', 'metalburst', 'comeuppance',
  'nightshade', 'seismictoss', 'fissure', 'guillotine', 'horndrill', 'sheercold',
];

// engine key -> fork format id (D32: champions mod -> 0, genN -> N)
const FORMATS = {
  '0': 'bss',
  '1': 'gen1randombattle', '2': 'gen2randombattle', '3': 'gen3randombattle',
  '4': 'gen4randombattle', '5': 'gen5randombattle', '6': 'gen6randombattle',
  '7': 'gen7randombattle', '8': 'gen8randombattle', '9': 'gen9randombattle',
};

const out = {};
for (const [key, fmtId] of Object.entries(FORMATS)) {
  const dex = Sim.Dex.forFormat(fmtId);
  const atkTypes = Object.keys(dex.types.get('normal').damageTaken);
  out[key] = {};
  for (const id of MOVES) {
    const m = dex.moves.get(id);
    if (!m || !m.exists || !m.type) continue;
    const ignoresAll = m.ignoreImmunity === true || (m.ignoreImmunity && m.ignoreImmunity[m.type]);
    const immune = ignoresAll ? [] : atkTypes.filter(t => dex.types.get(t).damageTaken[m.type] === 3);
    const entry = { immune };
    if (m.ohko === 'Ice') entry.ohkoIce = true;
    // Ground moves also get the fork's isGrounded rule (Flying type /
    // Levitate / Eelevate / Air Balloon make the defender airborne unless it
    // holds Iron Ball) — flagged here, applied in node_calc.js where the
    // defender's ability/item are request data.
    if (m.type === 'Ground') entry.ground = true;
    out[key][id] = entry;
  }
}

const target = path.join(__dirname, '..', 'damage', 'special_move_immunity.json');
fs.writeFileSync(target, JSON.stringify(out, null, 2) + '\n');
console.log('wrote', target);
for (const [k, v] of Object.entries(out)) {
  const summary = Object.entries(v).map(([m, e]) => `${m}=${e.immune.length ? e.immune.join('|') : '-'}`).join(' ');
  console.log(`gen ${k}: ${summary}`);
}
