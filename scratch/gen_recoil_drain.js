'use strict';
// T6 (recoil/drain) — precompute fork-exact recoil/drain facts per engine key
// from the fork's own move data (the fork is the source of truth for raw game
// data). Values DO differ per engine (e.g. Double-Edge 25% gen1/2 vs 33%
// BSS/gen3+; Struggle 50%-of-damage gen1 vs 25%-of-damage gen2/3 vs 25%-of-
// max-HP gen4+), so the calc must not assume a single value.
//
// Fork application semantics (verified in dist code):
//   recoil (battle-actions.js applyRecoilDamage, called only when
//   move.totalDamage > 0, i.e. the move DEALT damage):
//     struggleRecoil        -> clamp(round(baseMaxhp/4), 1)   [baseMaxhp = user's actual max HP]
//     mindBlown/chloroblast -> round(maxhp/2)
//     recoil:[n,d]          -> clamp(round(damageDealt*n/d), 1)
//   drain (battle.js:1793-1800, inside battle.damage() so only on damage dealt):
//     gen<=4 -> clamp(floor(damageDealt*n/d), 1)
//     gen>4  -> round(damageDealt*n/d)
const path = require('path');
const fs = require('fs');
const Sim = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));

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
  for (const id of dex.moves.all()) {
    const m = dex.moves.get(id);
    if (!m || !m.exists) continue;
    const e = {};
    if (m.recoil) e.recoil = m.recoil;
    if (m.drain) e.drain = m.drain;
    if (m.struggleRecoil) e.struggleRecoil = true;
    if (m.mindBlownRecoil) e.mindBlownRecoil = true;
    if (m.chloroblastRecoil) e.chloroblastRecoil = true;
    if (Object.keys(e).length) out[key][Sim.toID(id)] = e; // id keys, like the other tables
  }
}

const target = path.join(__dirname, '..', 'damage', 'recoil_drain_moves.json');
fs.writeFileSync(target, JSON.stringify(out, null, 2) + '\n');
console.log('wrote', target);
for (const [k, v] of Object.entries(out)) {
  console.log(`engine ${k}: ${Object.keys(v).length} moves`);
}
