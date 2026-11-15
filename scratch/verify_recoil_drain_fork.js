'use strict';
// T6 (recoil/drain) — LIVE FORK BATTLE verification (the fork = the battle
// environment the calc must match). For each case: run a real fork battle,
// measure the ATTACKER's actual HP delta, and compare it to the wrapper's
// recoil/heal/lifeOrb blocks (fork-exact formulas, damage-range dependent).
//
// Fork semantics under test (verified in dist code):
//   recoil: only when the move DEALT damage (battle-actions.js:905 guard);
//     struggleRecoil -> clamp(round(baseMaxhp/4), 1)  [g5-9/champ]
//                      clamp(floor(baseMaxhp/4), 1)  [g1-4: gen4 script override]
//     mindBlown/chloroblast -> round(maxhp/2)
//     recoil:[n,d] -> clamp(round(dmg*n/d), 1)   [g5-9/champ]
//                    clamp(floor(dmg*n/d), 1)    [g1-4: gen4 script override]
//   drain: gen<=4 clamp(floor(dmg*n/d), 1) ; gen>4 round(dmg*n/d)
//   Life Orb: baseMaxhp/10 (all gens in this fork), non-Status moves only
const path = require('path');
const { execFileSync } = require('child_process');
const Sim = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));

const NODE = process.execPath;
const NODE_CALC = path.join(__dirname, '..', 'damage', 'node_calc.js');
const LEVEL = 50;

const fmt = (id) => Sim.Dex.formats.get(id, true);
const FMT_CHAMP = (() => {
  for (const id of ['bss', 'gen9championsbssregmc', 'gen9championsou']) {
    try { const f = Sim.Dex.formats.get(id, true); if (f && f.name && f.mod === 'champions') return f; } catch (e) {}
  }
  throw new Error('no champions format');
})();
const FMT_G9 = fmt('gen9randombattle');
const FMT_G1 = fmt('gen1randombattle');

function wrapper(req) {
  const out = execFileSync(NODE, [NODE_CALC], { input: JSON.stringify(req), encoding: 'utf8' });
  const j = JSON.parse(out);
  if (j.error) throw new Error(j.error);
  return j;
}
const wmon = (sp, extra = {}) => ({ name: sp, level: LEVEL, nature: 'Docile', evs: { atk: 252, spa: 252 }, ...extra });

function attackerDelta(battle, startHp) {
  // HP change of p1's active mon: delta>0 = lost HP (recoil/life orb), <0 = net healed.
  const a = battle.p1.active[0];
  const start = startHp === undefined ? a.maxhp : startHp;
  const end = a.hp;
  return { start, end, maxhp: a.maxhp, delta: start - end };
}

function runBattle({ fmt, p1, p2, seed, setup, c1, c2 }) {
  const opts = { format: fmt, debug: true, strictChoices: false, seed: `gen5,${seed}` };
  // Per-mon EV override (s.evs) — default 252 Atk/SpA like the rest of the suite;
  // gen1/2 cases pass explicit zero EVs (see the gen1 case note).
  const team = (s) => { const t = { species: s.sp, level: LEVEL, nature: 'Docile', evs: s.evs || { atk: 252, spa: 252 }, moves: s.moves }; if (s.item) t.item = s.item; return t; };
  opts.p1 = { team: p1.map(team) };
  opts.p2 = { team: p2.map(team) };
  const battle = new Sim.Battle(opts);
  try {
    if (battle.requestState === 'teampreview') battle.makeChoices('team', 'team');
    if (setup) setup(battle);
    if (c1 !== undefined || c2 !== undefined) {
      battle.makeChoices(...battle.sides.map((side, i) => (side.requestState ? (([c1, c2][i]) || 'pass') : null)));
    }
    return battle;
  } finally { /* caller destroys after reading */ }
}

let pass = 0, fail = 0;
const check = (label, cond, detail) => {
  if (cond) { pass++; console.log(`  PASS  ${label}`); }
  else { fail++; console.log(`  FAIL  ${label}  ${detail}`); }
};

// ── cases ───────────────────────────────────────────────────────────────
function caseBraveBirdChamps() {
  console.log('Champions (gen 0) Brave Bird (recoil 33% of damage dealt):');
  const b = runBattle({
    fmt: FMT_CHAMP, seed: 1,
    p1: [{ sp: 'charizard', moves: ['bravebird', 'sleeptalk'] }],
    p2: [{ sp: 'blastoise', moves: ['sleeptalk'] }],
    c1: 'move 1', c2: 'move 1',
  });
  const delta = attackerDelta(b);
  const dmgToDef = b.p2.active[0].maxhp - b.p2.active[0].hp;
  const w = wrapper({ type: 'damage', gen: 0, attacker: wmon('charizard'), defender: wmon('blastoise'), move: 'brave bird' });
  check('attacker delta within wrapper recoil block',
    w.recoil && delta.delta >= w.recoil.min && delta.delta <= w.recoil.max,
    `delta=${delta.delta} recoil=${JSON.stringify(w.recoil)} dmg=${dmgToDef}`);
  check('fork value == round(dmg*33/100)', delta.delta === Math.round(dmgToDef * 33 / 100),
    `delta=${delta.delta} expect=${Math.round(dmgToDef * 33 / 100)}`);
  b.destroy();
}
function caseGigaDrainG9() {
  console.log('Gen9 Giga Drain (heal 50% of damage dealt, attacker at 50% HP):');
  const b = runBattle({
    fmt: FMT_G9, seed: 1,
    p1: [{ sp: 'venusaur', moves: ['gigadrain', 'sleeptalk'] }],
    p2: [{ sp: 'charizard', moves: ['sleeptalk'] }],
    setup: (bt) => { bt.p1.active[0].hp = Math.floor(bt.p1.active[0].maxhp / 2); },
    c1: 'move 1', c2: 'move 1',
  });
  const startHp = Math.floor(b.p1.active[0].maxhp / 2);
  const delta = attackerDelta(b, startHp);
  const dmgDealt = b.p2.active[0].maxhp - b.p2.active[0].hp;
  const expectedHeal = Math.min(Math.round(dmgDealt / 2), b.p1.active[0].maxhp - startHp);
  check('attacker net heal == min(round(dmg/2), missing HP)', -delta.delta === expectedHeal,
    `heal=${-delta.delta} expect=${expectedHeal} dmg=${dmgDealt}`);
  const w = wrapper({ type: 'damage', gen: 9, attacker: wmon('venusaur'), defender: wmon('charizard'), move: 'giga drain' });
  check('wrapper heal block bracketed the fork heal',
    w.heal && w.heal.min <= expectedHeal && expectedHeal <= w.heal.max,
    `heal=${expectedHeal} heal-block=${JSON.stringify(w.heal)}`);
  b.destroy();
}
function caseStruggleG9() {
  console.log('Gen9 Struggle (recoil = 25% of user max HP, exact):');
  const b = runBattle({
    fmt: FMT_G9, seed: 1,
    p1: [{ sp: 'charizard', moves: ['struggle', 'sleeptalk'] }],
    p2: [{ sp: 'blastoise', moves: ['sleeptalk'] }],
    c1: 'move 1', c2: 'move 1',
  });
  const delta = attackerDelta(b);
  const expect = Math.max(1, Math.round(b.p1.active[0].maxhp / 4));
  check('attacker delta == round(maxhp/4)', delta.delta === expect, `delta=${delta.delta} expect=${expect} maxhp=${b.p1.active[0].maxhp}`);
  const w = wrapper({ type: 'damage', gen: 9, attacker: wmon('charizard'), defender: wmon('blastoise'), move: 'struggle' });
  check('wrapper recoil matches', w.recoil && w.recoil.min === expect && w.recoil.max === expect,
    `recoil=${JSON.stringify(w.recoil)} expect=${expect}`);
  b.destroy();
}
function caseMindBlownG9() {
  console.log('Gen9 Mind Blown (recoil = 50% of user max HP, exact):');
  const b = runBattle({
    fmt: FMT_G9, seed: 1,
    p1: [{ sp: 'charizard', moves: ['mindblown', 'sleeptalk'] }],
    p2: [{ sp: 'blastoise', moves: ['sleeptalk'] }],
    c1: 'move 1', c2: 'move 1',
  });
  const delta = attackerDelta(b);
  const expect = Math.round(b.p1.active[0].maxhp / 2);
  check('attacker delta == round(maxhp/2)', delta.delta === expect, `delta=${delta.delta} expect=${expect} maxhp=${b.p1.active[0].maxhp}`);
  const w = wrapper({ type: 'damage', gen: 9, attacker: wmon('charizard'), defender: wmon('blastoise'), move: 'mind blown' });
  check('wrapper recoil matches', w.recoil && w.recoil.min === expect && w.recoil.max === expect,
    `recoil=${JSON.stringify(w.recoil)} expect=${expect}`);
  b.destroy();
}
function caseStruggleG1() {
  console.log('Gen1 Struggle (fork gen1 data: recoil = 50% of damage dealt):');
  // Gen 1 stat-model note: the fork computes gen1 stats with the MODERN formula
  // (2*base + iv + ev/4), while @smogon/calc (damage-model truth) uses the
  // authentic RBY DV/stat-EXP model ((base+dv)*2 + sqrt-ish statexp). With IV 31
  // and zero EVs the two agree exactly at this level (50), so the fork-vs-calc
  // bracket below is valid. (With EVs the models can diverge by ~1 — a known
  // gen1/2 limitation, documented; the BSS/champions target engine is
  // unaffected because both models use base + EV there.)
  const ZERO = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
  const b = runBattle({
    fmt: FMT_G1, seed: 1,
    p1: [{ sp: 'pikachu', moves: ['struggle'], evs: ZERO }],
    p2: [{ sp: 'sandshrew', moves: ['sleeptalk'], evs: ZERO }],
    c1: 'move 1', c2: 'move 1',
  });
  const delta = attackerDelta(b);
  const dmgDealt = b.p2.active[0].maxhp - b.p2.active[0].hp;
  // Gen 1-4 recoil uses FLOOR (fork gen4 script override, inherited by gens 1-4):
  // floor(14.5) = 14, where round would give 15 — this is why the earlier
  // round-based assertion was off by one.
  const expect = Math.max(1, Math.floor(dmgDealt / 2));
  check('attacker delta == clamp(floor(dmg/2),1) [gen1-4 floor]', delta.delta === expect,
    `delta=${delta.delta} expect=${expect} dmg=${dmgDealt}`);
  const w = wrapper({ type: 'damage', gen: 1, attacker: wmon('pikachu', { evs: ZERO }), defender: wmon('sandshrew', { evs: ZERO }), move: 'struggle' });
  check('wrapper recoil block bracketed', w.recoil && w.recoil.min <= expect && expect <= w.recoil.max,
    `expect=${expect} recoil=${JSON.stringify(w.recoil)}`);
  b.destroy();
}
function caseLifeOrbG9() {
  console.log('Gen9 Life Orb (recoil = 10% of user max HP, exact):');
  const b = runBattle({
    fmt: FMT_G9, seed: 1,
    p1: [{ sp: 'charizard', item: 'lifeorb', moves: ['tackle', 'sleeptalk'] }],
    p2: [{ sp: 'blastoise', moves: ['sleeptalk'] }],
    c1: 'move 1', c2: 'move 1',
  });
  const delta = attackerDelta(b);
  const expect = Math.floor(b.p1.active[0].maxhp / 10);
  check('attacker delta == floor(maxhp/10)', delta.delta === expect,
    `delta=${delta.delta} expect=${expect} maxhp=${b.p1.active[0].maxhp}`);
  const w = wrapper({ type: 'damage', gen: 9, attacker: wmon('charizard', { item: 'lifeorb' }), defender: wmon('blastoise'), move: 'tackle' });
  check('wrapper lifeOrb matches', w.lifeOrb && w.lifeOrb.min === expect && w.lifeOrb.max === expect,
    `lifeOrb=${JSON.stringify(w.lifeOrb)} expect=${expect}`);
  b.destroy();
}
function caseImmuneNoRecoil() {
  console.log('Gen9 Take Down (Normal) vs Ghost — immune, NO recoil:');
  const b = runBattle({
    fmt: FMT_G9, seed: 1,
    p1: [{ sp: 'charizard', moves: ['takedown', 'sleeptalk'] }],
    p2: [{ sp: 'gengar', moves: ['sleeptalk'] }],
    c1: 'move 1', c2: 'move 1',
  });
  const delta = attackerDelta(b);
  check('attacker took no recoil', delta.delta === 0, `delta=${delta.delta}`);
  const w = wrapper({ type: 'damage', gen: 9, attacker: wmon('charizard'), defender: wmon('gengar'), move: 'take down' });
  check('wrapper reports no recoil block', !('recoil' in w) && w.immune, `resp=${JSON.stringify({ immune: w.immune, recoil: w.recoil })}`);
  b.destroy();
}

try {
  caseBraveBirdChamps();
  caseGigaDrainG9();
  caseStruggleG9();
  caseMindBlownG9();
  caseStruggleG1();
  caseLifeOrbG9();
  caseImmuneNoRecoil();
} catch (e) {
  console.error('HARNESS ERROR:', e.message);
  process.exitCode = 2;
}
console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
