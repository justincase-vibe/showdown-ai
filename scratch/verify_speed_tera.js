'use strict';
// KEEPERS HARNES (PLAN.md D42) — speed mirror + Tera-aware special-move immunity.
//
// Part A: our `effectiveSpeed` endpoint (damage/node_calc.js) vs the LIBRARY's
// own getFinalSpeed (node_modules/@smogon/calc/dist/mechanics/util.js, deep
// require — scratch only, the shipped code mirrors it by hand). Every scenario
// must match EXACTLY, except the documented gen-5 paralysis divergence (fork
// x0.5 = battle environment wins over the library's x0.25).
//
// Part B: Tera-aware immunity of the special-move gate. Fork ground truth:
// dex.getImmunity (dist/sim/dex.js:226) uses target.getTypes(), which returns
// the terastallized type when terastallized and not Stellar
// (dist/sim/pokemon.js:1583); the library's effectiveness overrides with
// defender.teraType the same way (dist/mechanics/gen789.js:292-294). We assert
// the engine outputs (immune -> 0; control -> hits) for gen 9 + champions, and
// that a teraType is IGNORED in gen 1 (no Tera there).

const { spawnSync } = require('child_process');
const path = require('path');
const { Pokemon, Field, Side, Generations } = require('@smogon/calc');
const util = require('@smogon/calc/dist/mechanics/util');

const ROOT = path.join(__dirname, '..');
function ourSpeed(req) {
  const r = spawnSync('node', [path.join(ROOT, 'damage', 'node_calc.js')],
    { input: JSON.stringify(req), encoding: 'utf8', timeout: 30000 });
  if (r.status !== 0) throw new Error('node_calc failed: ' + r.stdout + r.stderr);
  return JSON.parse(r.stdout.trim()).speed;
}
function ourDamage(req) {
  const r = spawnSync('node', [path.join(ROOT, 'damage', 'node_calc.js')],
    { input: JSON.stringify(req), encoding: 'utf8', timeout: 30000 });
  if (r.status !== 0) throw new Error('node_calc failed: ' + r.stdout + r.stderr);
  return JSON.parse(r.stdout.trim());
}
function libSpeed(gen, mon, tailwind, field) {
  const g = Generations.get(gen);
  const { name, ...opts } = mon;
  const p = new Pokemon(g, name, opts);
  const f = new Field({
    weather: (field && field.weather) || '',
    terrain: (field && field.terrain) || '',
    attackerSide: new Side({ isTailwind: !!tailwind }),
    defenderSide: new Side({}),
  });
  return util.getFinalSpeed(g, p, f, f.attackerSide);
}

let pass = 0, fail = 0;
function check(label, got, want) {
  const ok = got === want;
  if (ok) pass++; else fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}  got=${got} want=${want}`);
}
function checkEq(label, ours, libs) {
  const ok = ours === libs;
  if (ok) pass++; else fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}  ours=${ours} library=${libs}`);
}

const J = { name: 'Jolteon', level: 50 };
const GARCH = { name: 'Garchomp' };
// [label, gen, mon, tailwind, field, expectedLibDiverges]
const SPEED = [
  ['gen9 Jolteon par', 9, { ...J, status: 'par' }, false, {}],
  ['gen1 Jolteon par', 1, { ...J, status: 'par' }, false, {}],
  ['gen2 Jolteon par', 2, { ...J, status: 'par' }, false, {}],
  ['gen3 Jolteon par', 3, { ...J, status: 'par' }, false, {}],
  ['gen4 Jolteon par', 4, { ...J, status: 'par' }, false, {}],
  ['gen5 Jolteon par (fork x0.5 vs lib x0.25)', 5, { ...J, status: 'par' }, false, {}, true],
  ['gen6 Jolteon par', 6, { ...J, status: 'par' }, false, {}],
  ['champs Jolteon par', 0, { name: 'Jolteon', level: 50, status: 'par' }, false, {}],
  ['gen9 Pelipper Swift Swim Rain', 9, { name: 'Pelipper', level: 50, ability: 'Swift Swim' }, false, { weather: 'Rain' }],
  ['gen9 Sceptile Chlorophyll Sun', 9, { name: 'Sceptile', level: 50, ability: 'Chlorophyll' }, false, { weather: 'Sun' }],
  ['gen9 Beartic Slush Rush Hail', 9, { name: 'Beartic', level: 50, ability: 'Slush Rush' }, false, { weather: 'Hail' }],
  ['gen9 Cramorant Surge Surfer Electric', 9, { name: 'Cramorant', level: 50, ability: 'Surge Surfer' }, false, { terrain: 'Electric' }],
  ['gen9 Joltik Iron Ball', 9, { name: 'Joltik', level: 50, item: 'Iron Ball' }, false, {}],
  ['gen9 Machamp Slow Start on', 9, { name: 'Machamp', level: 50, ability: 'Slow Start', abilityOn: true }, false, {}],
  ['gen9 Carnivine Quick Feet par', 9, { name: 'Carnivine', level: 50, ability: 'Quick Feet', status: 'par' }, false, {}],
  ['gen9 Carnivine Quick Feet par + Scarf', 9, { name: 'Carnivine', level: 50, ability: 'Quick Feet', status: 'par', item: 'Choice Scarf' }, false, {}],
  ['gen9 Great Tusk QP spe', 9, { name: 'Great Tusk', level: 50, ability: 'Quark Drive', boostedStat: 'spe' }, false, {}],
  ['gen9 Great Tusk QP auto + Electric', 9, { name: 'Great Tusk', level: 50, ability: 'Quark Drive', boostedStat: 'auto' }, false, { terrain: 'Electric' }],
  ['gen9 Great Tusk QP off (no flag)', 9, { name: 'Great Tusk', level: 50, ability: 'Quark Drive' }, false, {}],
  ['gen9 Carnivine Unburden on', 9, { name: 'Carnivine', level: 50, ability: 'Unburden', abilityOn: true }, false, {}],
  ['gen9 Carnivine Unburden on + Scarf (suppressed)', 9, { name: 'Carnivine', level: 50, ability: 'Unburden', abilityOn: true, item: 'Choice Scarf' }, false, {}],
  ['gen9 Garchomp Tailwind', 9, GARCH, true, {}],
  ['gen9 Garchomp par + Scarf', 9, { ...GARCH, status: 'par', item: 'Choice Scarf' }, false, {}],
  ['gen2 Jolteon +1 boost', 2, { ...J, boosts: { spe: 1 } }, false, {}],
  ['gen2 Jolteon -1 boost', 2, { ...J, boosts: { spe: -1 } }, false, {}],
  ['gen2 Jolteon +4 boost', 2, { ...J, boosts: { spe: 4 } }, false, {}],
  ['champs Garchomp Tailwind', 0, GARCH, true, {}],
  ['champs Sceptile Chlorophyll Sun', 0, { name: 'Sceptile', level: 50, ability: 'Chlorophyll' }, false, { weather: 'Sun' }],
];
for (const [label, gen, mon, tw, field, diverges] of SPEED) {
  const ours = ourSpeed({ type: 'speed', gen, pokemon: mon, tailwind: tw, field });
  const libs = libSpeed(gen, mon, tw, field);
  if (diverges) {
    // documented fork-vs-library divergence: assert BOTH values explicitly
    const ok = (ours === 75 && libs === 37);
    if (ok) pass++; else fail++;
    console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}  ours=${ours} (fork x0.5) library=${libs} (lib x0.25)`);
  } else {
    checkEq(label, ours, libs);
  }
}

console.log('─'.repeat(60));
console.log('Part B: Tera-aware special-move immunity (fork getTypes ground truth)');
const SAND = { name: 'Sandshrew', level: 50 };
const B = { name: 'Marowak', level: 50 };
function spf(gen, def) { return ourDamage({ type: 'damage', gen, attacker: B, defender: def, move: 'superfang' }); }
function imm(label, r) {
  const ok = r.immune === true && r.min === 0 && r.max === 0;
  if (ok) pass++; else fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}  ${JSON.stringify({ min: r.min, max: r.max, immune: r.immune })}`);
}
function hits(label, r) {
  const ok = r.immune === false && r.max > 0;
  if (ok) pass++; else fail++;
  console.log(`${ok ? 'PASS' : 'FAIL'}  ${label}  ${JSON.stringify({ min: r.min, max: r.max, immune: r.immune })}`);
}
imm('gen9 superfang vs Tera-Ghost Sandshrew', spf(9, { ...SAND, teraType: 'Ghost' }));
hits('gen9 superfang vs Sandshrew (control)', spf(9, SAND));
// champs note: the gen-0 engine only whitelists 359 BSS-legal species —
// Sandshrew/Marowak are NOT in that table (clean error from node_calc), so the
// champs case uses legal species: Farfetchd attacking Gengar (Ghost type is
// what the Tera-Ghost gate is about).
imm('champs superfang vs Tera-Ghost Gengar', ourDamage({ type: 'damage', gen: 0, attacker: { name: 'Farfetchd', level: 50 }, defender: { name: 'Gengar', level: 50, teraType: 'Ghost' }, move: 'superfang' }));
hits('champs superfang vs Beartic (control: not Ghost, hits)', ourDamage({ type: 'damage', gen: 0, attacker: { name: 'Farfetchd', level: 50 }, defender: { name: 'Beartic', level: 50 }, move: 'superfang' }));
hits('gen9 superfang vs Stellar Sandshrew (Stellar keeps base types)', spf(9, { ...SAND, teraType: 'Stellar' }));
hits('gen1 superfang vs Sandshrew + teraType Ghost (no Tera in gen1)', spf(1, { ...SAND, teraType: 'Ghost' }));
imm('gen9 fissure vs Tera-Flying Onix', ourDamage({ type: 'damage', gen: 9, attacker: B, defender: { name: 'Onix', level: 50, teraType: 'Flying' }, move: 'fissure' }));
hits('gen9 fissure vs Onix (control)', ourDamage({ type: 'damage', gen: 9, attacker: B, defender: { name: 'Onix', level: 50 }, move: 'fissure' }));
imm('gen9 sheercold vs Tera-Ice Gengar', ourDamage({ type: 'damage', gen: 9, attacker: B, defender: { name: 'Gengar', level: 50, teraType: 'Ice' }, move: 'sheercold' }));
hits('gen9 sheercold vs Gengar (control)', ourDamage({ type: 'damage', gen: 9, attacker: B, defender: { name: 'Gengar', level: 50 }, move: 'sheercold' }));
// levitate still works alongside Tera (ground check is independent of types)
imm('gen9 fissure vs Tera-Ghost Farfetchd + Levitate', ourDamage({ type: 'damage', gen: 9, attacker: B, defender: { name: 'Farfetchd', level: 50, teraType: 'Ghost', ability: 'Levitate' }, move: 'fissure' }));
// Iron Ball still grounds a Tera-Flying mon (fork isGrounded checks Iron Ball first)
hits('gen9 fissure vs Tera-Flying Onix + Iron Ball', ourDamage({ type: 'damage', gen: 9, attacker: B, defender: { name: 'Onix', level: 50, teraType: 'Flying', item: 'Iron Ball' }, move: 'fissure' }));

console.log('─'.repeat(60));
console.log(`${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
