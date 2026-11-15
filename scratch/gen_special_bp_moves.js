'use strict';
// Generate damage/special_bp_moves.json — per-ENGINE-KEY base power (no-state)
// for the 9 T5 state-dependent bpCallback moves that are BSS-legal and NOT
// modeled in @smogon/calc's mechanics (it returns the raw basePower, 0 for the
// bp=0 ones). The fork (Sim.Dex.forFormat) is the source of truth for raw move
// data, so we read the base BP per engine key straight from it — same pattern as
// special_move_immunity.js (D38).
//
// Engine key mapping (D32): champions -> 0, genN -> N (1..9).
//
// Schema: { "<engineKey>": { "<moveId>": <baseBP:number | null> } }
//   null = move unusable/undefined in that engine (not in data, or isNonstandard).
//
// Run: node scratch/gen_special_bp_moves.js
const path = require('path');
const fs = require('fs');
const Sim = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));

const MOVES = ['avalanche', 'beatup', 'firepledge', 'grasspledge', 'waterpledge',
  'lastrespects', 'spitup', 'stompingtantrum', 'temperflare'];

// engine key -> fork format id (same table as gen_special_move_immunity.js)
const FORMATS = {
  '0': 'bss',
  '1': 'gen1randombattle', '2': 'gen2randombattle', '3': 'gen3randombattle',
  '4': 'gen4randombattle', '5': 'gen5randombattle', '6': 'gen6randombattle',
  '7': 'gen7randombattle', '8': 'gen8randombattle', '9': 'gen9randombattle',
};

const out = {};
for (const [key, fmtId] of Object.entries(FORMATS)) {
  const dex = Sim.Dex.forFormat(fmtId);
  out[key] = {};
  for (const id of MOVES) {
    const m = dex.moves.get(id);
    if (!m || !m.exists) { out[key][id] = null; continue; }
    if (m.isNonstandard) { out[key][id] = null; continue; }
    out[key][id] = (m.basePower === undefined) ? null : m.basePower;
  }
}

const dest = path.join(__dirname, '..', 'damage', 'special_bp_moves.json');
fs.writeFileSync(dest, JSON.stringify(out, null, 2) + '\n');
console.log('wrote', dest);
for (const [k, v] of Object.entries(out)) {
  console.log(`key ${k}: ` + Object.entries(v).map(([m, bp]) => `${m}=${bp}`).join('  '));
}
