'use strict';
// One-off cross-check: @smogon/calc (Step 2 engine) vs the LIVE fork simulator.
//
// For each of 11 randomized Gen 9 matchups (level 50 = BSS battle level) we:
//   1. run a REAL headless battle in the fork (Sim.Battle, BSS format) and read
//      the defender's actual HP loss, and
//   2. compute the expected [min, max] (+ crit range) with @smogon/calc,
// then verify actual damage falls inside one of those ranges.
//
// This closes the one untested link from PLAN.md: library numbers vs the sim
// that actually resolves battles. Mechanics covered: plain physical/special,
// multi-hit, burn, Life Orb, Choice Band, +2 boosts, rain, sun, Reflect,
// Adaptability (2x STAB).
//
// Run: node scratch_crosscheck_calc_vs_sim.js

const path = require('path');
const Sim = require(path.join(__dirname, 'pokemon-showdown/dist/sim'));
const { calculate, Pokemon, Move, Field, Side } = require('@smogon/calc');
const Learnsets = require(path.join(__dirname, 'pokemon-showdown/dist/data/learnsets')).Learnsets;
const _movesMod = require(path.join(__dirname, 'pokemon-showdown/dist/data/moves'));
const MovesData = _movesMod.Moves || _movesMod.moves;
const PokedexData = (function () {
  const d = require(path.join(__dirname, 'pokemon-showdown/dist/data/pokedex'));
  return d.Pokedex || d;
})();
// @smogon/calc resolves `item` by DISPLAY NAME ("Life Orb"), not id ("lifeorb")
// — passing an id is silently ignored. The sim (fork) uses ids. So the calc side
// must translate id -> name (ground truth: the fork's own items table). This is
// also the exact contract gap in damage/node_calc.js (see report).
const ItemsData = (function () {
  const m = require(path.join(__dirname, 'pokemon-showdown/dist/data/items'));
  const I = m.Items || m.items;
  const id2name = {};
  for (const id of Object.keys(I)) if (I[id] && I[id].name) id2name[id] = I[id].name;
  return id2name;
})();
const itemName = (id) => (id && ItemsData[id]) || id; // id -> display name (or passthrough)
// Same contract for ABILITIES: the library resolves them by display name
// ("Adaptability"), silently ignoring ids ("adaptability"). Neutral placeholders
// (e.g. 'limiter') that the fork table doesn't name are passed through as-is.
const AbilityData = (function () {
  const m = require(path.join(__dirname, 'pokemon-showdown/dist/data/abilities'));
  const A = m.Abilities || m.abilities;
  const id2name = {};
  for (const id of Object.keys(A)) if (A[id] && A[id].name) id2name[id] = A[id].name;
  return id2name;
})();
const abilityName = (id) => (id && AbilityData[id]) || id;

// ── seeded RNG (reproducible) ─────────────────────────────────────────
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const SEED = parseInt(process.env.SEED || '0xC0FFEE', 16); // env SEED to re-roll all matchups
const rng = mulberry32(SEED);
const pick = (arr) => arr[Math.floor(rng() * arr.length)];

// ── pools ──────────────────────────────────────────────────────────────
const SPECIES = ['garchomp', 'tyranitar', 'dragonite', 'hydreigon', 'charizard', 'chandelure',
  'gengar', 'lucario', 'machamp', 'samurott', 'kingdra', 'zeraora', 'raikou', 'suicune',
  'milotic', 'landorus', 'torkoal', 'mamoswine', 'salamence', 'krookodile', 'hippowdon',
  'tangrowth', 'corviknight', 'dragapult', 'clefable', 'slowkinggalar', 'clodsire',
  'porygonz', 'alakazam', 'steelix', 'bisharp', 'scizor', 'swampert'];
const ATT_NATURES = ['Adamant', 'Jolly', 'Hasty', 'Naive', 'Modest', 'Timid', 'Quiet', 'Serious'];
const DEF_NATURES = ['Bold', 'Impish', 'Calm', 'Careful', 'Relaxed', 'Docile', 'Gentle', 'Lax'];

// Moves with non-standard damage models (variable power, charging, forced
// switches, self-destruction, typing changes, ...) — out of scope for this
// cross-check; the standard pipeline is what we're verifying.
const EXCLUDE_MOVES = new Set([
  'struggle', 'explosion', 'selfdestruct', 'mindblown', 'finalgambit', 'psyblade', 'venoshock',
  'fly', 'bounce', 'solarbeam', 'skyattack', 'metronome', 'mimic', 'beatup', 'assist', 'copycat', 'technic',
  'shellsidearm', 'teeterdance', 'transform', 'naturepower', 'crushgrip', 'wringout', 'musclejet',
  'propulsionpush', 'integratedswords', 'tropicaisland', 'dynamxpunch', 'heavyslam', 'lowkick', 'foulplay',
  'partingshot', 'uturn', 'voltswitch', 'shedtail', 'batonpass', 'rollout', 'trumpcard', 'boltbeak',
  'firstimpression', 'bonerush', 'wrap', 'bind', 'squeeze', 'whirlpool', 'block', 'clamp', 'strangegame',
  // partial-trapping moves (separate damage model; known calc-vs-sim divergence, see PLAN.md)
  'infestation',
  // charging moves (no damage on the setup turn)
  'dig', 'shadowforce', 'phantomforce',
]);

function canLearn9(sp, mv) {
  const ls = Learnsets[sp];
  if (!ls || !ls.learnset) return false;
  const ways = ls.learnset[mv];
  return !!ways && ways.some((w) => w.startsWith('9'));
}

function candidateMoves(sp) {
  const ls = Learnsets[sp];
  if (!ls || !ls.learnset) return [];
  return Object.keys(ls.learnset).filter((mv) => {
    if (!ls.learnset[mv].some((w) => w.startsWith('9'))) return false;
    const m = MovesData[mv];
    if (!m || !m.basePower || m.basePower <= 0) return false;
    if (m.category !== 'Physical' && m.category !== 'Special') return false;
    if (m.target !== 'normal') return false; // explicit single-target only (clean choice syntax)
    if (EXCLUDE_MOVES.has(mv) || /^(max|gmax)/.test(mv)) return false;
    const acc = m.accuracy;
    if (acc !== null && acc !== true && acc !== 100) return false; // avoid miss-noise
    return true;
  });
}

function defaultAbility(sp) {
  const p = PokedexData[sp];
  return (p && p.abilities && (p.abilities['1'] || p.abilities['0'])) || null;
}

// ── scenarios (mechanic coverage; species/move/natures randomized) ─────
const SCENARIOS = [
  { name: 'plain physical', category: 'Physical' },
  { name: 'plain special', category: 'Special' },
  { name: 'multi-hit', multiHit: true },
  { name: 'burned physical attacker', category: 'Physical', attackerStatus: 'brn' },
  { name: 'life orb (physical)', category: 'Physical', attackerItem: 'lifeorb' },
  { name: 'choice band (physical)', category: 'Physical', attackerItem: 'choiceband' },
  { name: 'attacker +2 boost', boost: true },
  { name: 'rain + water move', type: 'Water', weather: 'raindance', calcWeather: 'Rain' },
  { name: 'sun + fire move', type: 'Fire', weather: 'sunnyday', calcWeather: 'Sun' },
  { name: 'reflect + physical', category: 'Physical', reflect: true },
  { name: 'adaptability (2x STAB)', stabAbility: 'adaptability' },
  { name: 'guts + burned physical', category: 'Physical', guts: true, attackerStatus: 'brn' },
  { name: 'aurora veil + special', category: 'Special', veil: true },
];
// Guts users: the sim needs the ability ON the set, and a burned Guts user must
// use a physical move (Guts only boosts physical; the sim's guts.onModifyAtk is
// category-agnostic but the burn-halving skip is physical-only — keep the
// scenario physical so the two effects are both exercised).
const GUTS_SPECIES = ['machamp', 'hariyama', 'ursaring', 'tyranitar', 'heracross', 'mawile', 'marowakalola'];

// ── matchup generation ─────────────────────────────────────────────────
function genMatchup(sc) {
  let attacker = null, move = null;
  for (let i = 0; i < 400 && !move; i++) {
    const sp = sc.guts ? pick(GUTS_SPECIES) : pick(SPECIES);
    const pool = candidateMoves(sp).filter((mv) => {
      const m = MovesData[mv];
      if (sc.category && m.category !== sc.category) return false;
      if (sc.type && m.type !== sc.type) return false;
      if (sc.multiHit && !(Array.isArray(m.multihit) && m.multihit[0] >= 2)) return false; // multihit is [min, max]
      if (sc.stabAbility && !PokedexData[sp].types.includes(m.type)) return false;
      return true;
    });
    if (pool.length) { attacker = sp; move = pick(pool); }
  }
  if (!move) throw new Error(`no move found for scenario: ${sc.name}`);
  let defender = pick(SPECIES);
  if (defender === attacker) defender = pick(SPECIES);
  const m = MovesData[move];
  return {
    sc,
    attacker, move, defender,
    attNature: pick(ATT_NATURES), defNature: pick(DEF_NATURES),
    attEvs: m.category === 'Physical' ? { atk: 252 } : { spa: 252 },
    defEvs: m.category === 'Physical' ? { def: 252 } : { spd: 252 },
    // Neutral abilities on both sides: no damage interaction (no Drought/Drizzle
    // weather, no Thick Fat / Multiscale etc.) so sim and calc stay in sync.
    // The ability pipeline itself is tested by the Adaptability scenario.
    attAbility: sc.stabAbility || (sc.guts ? 'guts' : 'limiter'),
    defAbility: 'stench',
    attItem: sc.attackerItem || null, attStatus: sc.attackerStatus || null,
    boostStat: m.category === 'Physical' ? 'atk' : 'spa',
    level: 50,
  };
}

// ── sim side: real headless battle in the fork ─────────────────────────
const FORMAT = (function () {
  // BSS format is the target; '@@@!teampreview' drops Team Preview (a choice-flow
  // rule, not a damage mechanic) so the headless battle runs turn 1 immediately.
  // Fallback: the fork's standard test format (same vanilla gen-9 mechanics).
  const candidates = ['gen9bssregi@@@!teampreview', 'gen9randombattle'];
  for (const id of candidates) {
    try {
      const f = Sim.Dex.formats.get(id, id.includes('@@@') ? true : false);
      if (f && f.name) return f; // Battle wants a format object, not an id
    } catch (e) { /* try next */ }
  }
  throw new Error('no usable gen-9 format found');
})();

function runSim(m, seed) {
  const team = (sp, o) => {
    // Note: this sim version ignores set.status — statuses are applied
    // imperatively below via trySetStatus (the fork's own test pattern).
    const t = { species: sp, level: 50, nature: o.nature, evs: o.evs, ability: o.ability };
    if (o.item) t.item = o.item;
    return t;
  };
  const opts = { format: FORMAT, debug: true, strictChoices: true, seed: `gen5,${(seed + 1).toString(16).padStart(16, '0')}` }; // format = object
  opts.p1 = { team: [Object.assign(team(m.attacker, { nature: m.attNature, evs: m.attEvs, ability: m.attAbility, item: m.attItem, status: m.attStatus }), { moves: [m.move, 'sleeptalk'] })] };
  opts.p2 = { team: [Object.assign(team(m.defender, { nature: m.defNature, evs: m.defEvs, ability: m.defAbility }), { moves: m.reflect ? ['reflect', 'sleeptalk'] : ['sleeptalk'] })] };

  const battle = new Sim.Battle(opts);
  try {
    // BSS has pickedTeamSize=3 -> a Team Preview phase first; each side picks
    // min(teamSize, 3) mons. Our 1-mon teams pick their single lead.
    if (battle.requestState === 'teampreview') battle.makeChoices('team', 'team');
    if (m.sc.weather) battle.field.setWeather(m.sc.weather, 'debug'); // 'debug' source = p1 lead (sim shortcut)
    if (m.sc.veil) battle.p2.addSideCondition('auroraveil', battle.p2.active[0]); // 5-turn special-damage veil (battle.p2 IS the Side)
    if (m.attStatus) battle.p1.active[0].trySetStatus(m.attStatus, battle.p1.active[0]);
    const statusApplied = !m.attStatus || battle.p1.active[0].status === m.attStatus;
    if (m.sc.boost) battle.p1.active[0].boostBy({ [m.boostStat]: 2 });
    if (m.reflect) battle.makeChoices('move sleeptalk', 'move reflect'); // turn 1: set screen
    const maxhp = battle.p2.active[0].maxhp;
    battle.makeChoices(`move ${m.move} 1`, 'move sleeptalk');
    // Actual hit count (multi-hit moves roll a random count; the sim logs it
    // on a |-hitcount| line). Drives the calc so totals are like-for-like.
    let hitCount = 1;
    for (const line of battle.log) {
      if (typeof line === 'string' && line.startsWith('|-hitcount|') && line.includes('p2a:')) {
        const n = parseInt(line.split('|')[3], 10); if (n) hitCount = n;
      }
    }
    // Measure the MOVE's damage from the log. Two filters are both required:
    //   * skip [from] lines  -> excludes residual/status damage (brn/psn/sand/recoil),
    //     e.g. a move's secondary poison tick, or a burn on either side;
    //   * skip percentage lines -> the sim logs each damage event TWICE: absolute
    //     ("130/170") and as a % of max ("77/100"); the denominator of the real
    //     line always equals the defender's maxhp, the percentage line does not.
    // Multi-hit moves log one cumulative line, so summing is correct.
    let total = 0, prevHp = maxhp;
    for (const line of battle.log) {
      if (typeof line !== 'string' || !line.startsWith('|-damage|')) continue;
      if (!line.includes('p2a:')) continue;
      if (line.includes('[from]')) continue;
      const [hpStr, maxStr] = line.split('|')[3].split('/');
      if (parseInt(maxStr, 10) !== maxhp) continue; // drop the "% of max" line
      const hp = parseInt(hpStr, 10);
      total += prevHp - hp;
      prevHp = hp;
    }
    return { damage: total, statusApplied, hitCount };
  } catch (e) {
    console.error(`\n[DEBUG] ${m.sc.name}: ${m.attacker} ${m.move} vs ${m.defender}`);
    console.error(battle.log.slice(-30).join('\n'));
    throw e;
  } finally {
    battle.destroy();
  }
}

// ── calc side: @smogon/calc (same code path as damage/node_calc.js) ───
// attStatusOverride: drive the calc from the status the SIM actually applied
// (a status-immune lead rejects the burn; compare against reality, not intent).
function runCalc(m, attStatusOverride, hitCount) {
  const ivs = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
  const clean = (o) => { const out = {}; for (const k of Object.keys(o)) if (o[k] !== null && o[k] !== undefined && !(k === 'boosts' && !Object.keys(o[k] || {}).length)) out[k] = o[k]; return out; };
  const mk = (sp, o) => new Pokemon(9, sp, clean({
    level: 50, nature: o.nature, evs: o.evs, ivs,
    ability: abilityName(o.ability), item: itemName(o.item), status: o.status, boosts: o.boosts || {},
  }));
  const attStatus = attStatusOverride === undefined ? m.attStatus : attStatusOverride;
  const att = mk(m.attacker, { nature: m.attNature, evs: m.attEvs, ability: m.attAbility, item: m.attItem, status: attStatus, boosts: m.sc.boost ? { [m.boostStat]: 2 } : {} });
  const def = mk(m.defender, { nature: m.defNature, evs: m.defEvs, ability: m.defAbility });
  const field = new Field({
    weather: m.sc.calcWeather,
    attackerSide: new Side({}),
    defenderSide: new Side((m.reflect && { isReflect: true }) || (m.sc.veil && { isAuroraVeil: true }) || {}),
  });
  const range = (crit) => {
    const mvOpts = {}; if (crit) mvOpts.isCrit = true;
    if (hitCount && hitCount > 1) mvOpts.hits = hitCount; // match the sim's rolled hit count
    const r = calculate(9, att, def, new Move(9, m.move, mvOpts), field);
    return r.range();
  };
  const [min, max] = range(false);
  const [critMin, critMax] = range(true);
  return { min, max, critMin, critMax, maxhp: def.maxHP() };
}

// ── main ───────────────────────────────────────────────────────────────
console.log(`format: ${FORMAT.name}${FORMAT.customRules ? ` + custom rules [${FORMAT.customRules}]` : ''} | level: 50 (BSS) | rng seed: 0x${SEED.toString(16).toUpperCase()}`);
let pass = 0, fail = 0;
const results = [];
SCENARIOS.forEach((sc, idx) => {
  // Reject any matchup whose normal OR crit band reaches/overkills the defender:
  // the sim clamps HP at 0, so a crit that OHKOs reports maxhp (clamped), which
  // falls BELOW the true crit band and would read as "out of range".
  const ohko = (c) => c.max >= c.maxhp || c.critMin >= c.maxhp;
  // Run the sim FIRST so the calc is driven by the status the sim ACTUALLY
  // applied (a status-immune lead rejects a burn) — compare reality, not intent.
  const calcFor = (mm, r) => runCalc(mm, (r.statusApplied ? mm.attStatus : null), r.hitCount);
  let m, c, actual = null, tries = 0;
  do {
    m = genMatchup(sc);
    const r = runSim(m, idx + tries);
    actual = r.damage;
    c = calcFor(m, r);
    tries++;
  } while (ohko(c) && tries < 12);
  if (ohko(c)) {
    fail++;
    results.push({ sc: sc.name, ok: false, note: 'SKIPPED (OHKO matchup)', actual, c, m });
    return;
  }
  // 100%-accuracy pool => damage should never be 0; retry defensively.
  let guard = 0;
  while (actual <= 0 && ++guard < 5) { const r = runSim(m, idx + 20 + guard); actual = r.damage; c = calcFor(m, r); }
  const isMulti = Array.isArray(MovesData[m.move].multihit);
  const inNormal = actual >= c.min && actual <= c.max;
  const inCrit = actual >= c.critMin && actual <= c.critMax;
  // Multi-hit moves can partially crit (some hits crit, some not) -> any total
  // between the all-normal and all-crit bands is legitimate.
  const ok = isMulti ? (actual >= c.min && actual <= c.critMax) : (inNormal || inCrit);
  ok ? pass++ : fail++;
  results.push({ sc: sc.name, ok, note: inNormal ? 'normal' : inCrit ? 'CRIT' : isMulti ? 'partial-crit' : 'OUT OF RANGE', actual, c, m });
});

for (const r of results) {
  const { m, c } = r;
  const md = MovesData[m.move];
  const line = [
    `${r.ok ? 'PASS' : 'FAIL'}  [${r.sc}]  ${m.attacker} ${m.attNature} ${md.type} ${m.move} (${md.basePower}BP${Array.isArray(md.multihit) ? `, ${md.multihit[0]}-${md.multihit[1]} hits` : ''})`,
    `      vs ${m.defender} ${m.defNature} | att ab:${m.attAbility} item:${m.attItem || '-'} status:${m.attStatus || '-'} | def ab:${m.defAbility}`,
    `      sim: ${r.actual}  calc: ${c.min}-${c.max}${r.note === 'CRIT' ? ` (crit ${c.critMin}-${c.critMax})` : ''}  -> ${r.actual / c.maxhp * 100 | 0}% of maxhp ${c.maxhp}  [${r.note}]`,
  ].join('\n');
  console.log(line + '\n');
}
console.log(`\n${pass}/${pass + fail} matchups: sim damage within @smogon/calc range (normal or crit)`);
process.exit(fail ? 1 : 0);
