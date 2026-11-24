'use strict';
// B1 — ALL-FORMAT regression test (Claude small-addition #1; see GATE_CONTROLS_C1_C5.md).
//
// For EVERY format in the registry (legality/formats.py TARGET_FORMATS) this:
//   (a) resolves the @smogon/calc engine key from the registry (no silent fallbacks —
//       the Python side asserts this per format; this file consumes the resolved key), and
//   (b) runs a REAL headless fork battle per scenario and verifies the sim's actual
//       damage falls inside the @smogon/calc [min, max] band (normal or crit; multi-hit
//       anywhere between the all-normal and all-crit bands).
//
// Ground truth = @smogon/calc (user ruling: the Smogon damage calc is the source of
// truth, NOT the fork). The sim (fork) is the thing being checked: if the two
// disagree for a format, the AI's damage belief for that format is wrong.
//
// DOUBLES SCREEN RULE (discovered while fixing the 4 doubles Reflect/Veil FAILs):
// In doubles, Reflect / Light Screen / Aurora Veil reduce damage by 2/3
// (2732/4096), NOT 1/2, and screens do NOT apply to crits. Both the fork and the
// calc library implement this — the library selects the 2/3 ratio only when
// Field.gameType !== 'Singles' (gen789.js:1263, champions.js:735) and defaults
// gameType to 'Singles' (field.js:23). So runCalc MUST pass the format's gameType
// into new Field({...}); otherwise every doubles screen scenario silently compares
// the fork's 2/3 damage against a singles 1/2 band and FAILs. (The production
// damage path had the same latent bug — fixed separately in damage/node_calc.js.)
//
// Mechanics covered per format: plain physical/special, burned attacker, multi-hit,
// +2 stat boost, rain+water, sun+fire, Reflect, Choice Band (gen2+), Life Orb (gen4+),
// Adaptability (gen4+), Guts+burn (gen3+), Aurora Veil (gen4+). Singles AND doubles
// (VGC / champions VGC) with 2-mon teams and explicit targeting.
//
// Species/moves are filtered to what is legal-ish in that format's generation via the
// fork's own data tables (learnsets + pokedex), so the same code path the AI's damage
// layer uses (damage/node_calc.js) is exercised with that format's engine key and level.
//
// Verdict policy (recorded in OPEN_DESIGN_CHOICES.md):
//   * must-pass set: engine key 9 (gen9, incl. BSS 3v3) and 0 (champions) — the formats
//     the AI actually plays. Any divergence there => FAIL (exit 1), UNLESS it is a
//     verified @smogon/calc library gap (see KNOWN_ENGINE_GAPS below), which is a WARN.
//   * engine keys 1-8 (old gens): the calc library's per-gen data tables have known gaps
//     (e.g. gen1 Karate Chop lacks `category` and carries the wrong type) that we cannot
//     patch locally. Divergences there are reported as WARN with full detail (move, fork
//     damage, calc band) but do not fail the suite, so they stay visible instead of
//     hiding in the numbers.
//   * KNOWN_ENGINE_GAPS: verified per-engine @smogon/calc gaps (NOT fork bugs, NOT
//     band-loosening) that are classified as WARN instead of FAIL so the must-pass set
//     stays meaningful. Each is documented with evidence in OPEN_DESIGN_CHOICES.md.
//
// Driver:  python -m pytest tools/tests/test_b1_all_formats.py
// Manual:  node tools/tests/b1_calc_vs_sim_all_formats.js  (sanity: gen9bss3v3 only)
//          node tools/tests/b1_calc_vs_sim_all_formats.js '<json payload>'
//          payload = [{ "id", "engineKey", "level", "gameType" }, ...]

const path = require('path');
const ROOT = path.join(__dirname, '..', '..');
const Sim = require(path.join(ROOT, 'pokemon-showdown/dist/sim'));
const { calculate, Pokemon, Move, Field, Side, toID } = require('@smogon/calc');

// ── @smogon/calc id->display-name contract (same as damage/node_calc.js) ───
function loadId2Name(file, tableKey) {
  try {
    const mod = require(path.join(ROOT, 'pokemon-showdown/dist/data', file));
    const table = (mod && (mod[tableKey])) || {};
    const m = {};
    for (const id of Object.keys(table)) if (table[id] && table[id].name) m[id] = table[id].name;
    return m;
  } catch (e) { return {}; }
}
const ITEMS = loadId2Name('items', 'Items');
const ABILITIES = loadId2Name('abilities', 'Abilities');
const itemName = (id) => (id && ITEMS[id]) || id;
const abilityName = (id) => (id && ABILITIES[id]) || id;

// ── seeded RNG (reproducible test) ─────────────────────────────────────
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(0xB1F0B5 & 0x7fffffff); // fixed seed: reproducible test
const pick = (arr) => arr[Math.floor(rng() * arr.length)];
const shuffle = (arr) => { const a = arr.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };

// ── species pools ────────────────────────────────────────────────────────
// Base pool: exists in gen1 AND gen9 (verified: all in both the fork data and the
// calc engines). Modern pool (gen5+): a few stronger types for variety. Champions
// pool: the engine-0 (BSS/champions) species whitelist is restrictive (359 species);
// these are verified present in it.
const POOL_BASE = ['machop', 'geodude', 'onix', 'grimer', 'magnemite', 'poliwag',
  'pidgey', 'farfetchd', 'rhyhorn', 'zubat', 'doduo', 'pikachu', 'eevee'];
const POOL_MODERN = ['charizard', 'gengar', 'lucario', 'machamp', 'scizor', 'swampert', 'dragonite'];
const POOL_CHAMPIONS = ['tyranitar', 'dragonite', 'charizard', 'gengar', 'lucario', 'machamp',
  'steelix', 'scizor', 'swampert', 'pikachu', 'farfetchd', 'rhyperior', 'victreebel',
  'slowking', 'heracross', 'persian', 'clefable'];

// Abilities provably NEUTRAL for the damage we measure (full HP, no sleep applied,
// no OHKO matchups, defender never attacks, no paralysis source):
const NEUTRAL = new Set(['sturdy', 'vitalspirit', 'steadfast', 'torrent', 'overgrow',
  'keeneye', 'swarm', 'blaze', 'shadowtag', 'lightningrod', 'noguard', 'levitate',
  'oblivious', 'innerfocus', 'cutcharm', 'rockhead', 'limber']);

// Moves with non-standard damage models — out of scope (same list as the Step-2
// scratch cross-check, plus the variable/random-basePower and stateful ones found
// while generalizing to old gens):
const EXCLUDE_MOVES = new Set([
  'struggle', 'explosion', 'selfdestruct', 'mindblown', 'finalgambit', 'psyblade', 'venoshock',
  'fly', 'bounce', 'solarbeam', 'skyattack', 'metronome', 'mimic', 'beatup', 'assist', 'copycat', 'technic',
  'shellsidearm', 'teeterdance', 'transform', 'naturepower', 'crushgrip', 'wringout', 'musclejet',
  'propulsionpush', 'integratedswords', 'tropicaisland', 'dynamxpunch', 'heavyslam', 'lowkick', 'foulplay',
  'partingshot', 'uturn', 'voltswitch', 'shedtail', 'batonpass', 'rollout', 'trumpcard', 'boltbeak',
  'firstimpression', 'bonerush', 'wrap', 'bind', 'squeeze', 'whirlpool', 'block', 'clamp', 'strangegame',
  'infestation',
  'dig', 'shadowforce', 'phantomforce',
  // variable/random base power (sim and calc can't be like-for-like):
  'magnitude', 'boneclub',
  // stateful power growth across turns:
  'furycutter', 'tailslap',
  // stateful: doubles BP if the ATTACKER's last move failed (fork: basePowerCallback on
  // moveLastTurnResult===false). The test's Sleep-Talk filler fails, so the sim inflates
  // the hit while the static calc (which can't see battle history) does not -> desync.
  'stompingtantrum',
  // charging in old gens (no damage on the setup turn):
  'skullbash',
]);

// ── scenarios (gen-gated; `ability` = required attacker ability) ─────────
const SCENARIOS = [
  { name: 'plain-physical', minGen: 1, category: 'Physical' },
  { name: 'plain-special', minGen: 1, category: 'Special' },
  { name: 'burned-attacker', minGen: 1, category: 'Physical', attStatus: 'brn' },
  { name: 'multihit', minGen: 1, multihit: true },
  { name: 'boost2', minGen: 1, boost: true },
  { name: 'rain-water', minGen: 1, type: 'Water', weather: 'raindance', calcWeather: 'Rain' },
  { name: 'sun-fire', minGen: 1, type: 'Fire', weather: 'sunnyday', calcWeather: 'Sun' },
  { name: 'reflect', minGen: 1, category: 'Physical', reflect: true },
  { name: 'choiceband', minGen: 2, category: 'Physical', attItem: 'choiceband' },
  { name: 'lifeorb', minGen: 4, category: 'Physical', attItem: 'lifeorb' },
  { name: 'adaptability', minGen: 4, ability: 'adaptability', stab: true },
  { name: 'guts-burn', minGen: 3, ability: 'guts', attStatus: 'brn', category: 'Physical' },
  { name: 'auroraveil', minGen: 4, category: 'Special', veil: true },
];

// ── per-format context ───────────────────────────────────────────────────
function loadFormats() {
  if (process.argv[2]) {
    const list = JSON.parse(process.argv[2]);
    if (!Array.isArray(list) || !list.length) throw new Error('empty format payload');
    return list;
  }
  // standalone sanity: the primary target format only
  return [{ id: 'gen9bss3v3', engineKey: 9, level: 50, gameType: 'singles' }];
}

function makeCtx(fmtInfo) {
  const fmt = Sim.Dex.formats.get(fmtInfo.id);
  if (!fmt || !fmt.name) throw new Error(`format '${fmtInfo.id}' not found in fork data`);
  const fmtDex = Sim.Dex.forFormat(fmtInfo.id);
  const gen = fmtInfo.engineKey === 0 ? 9 : fmtInfo.engineKey; // engine 0 = champions (gen9-based data)
  const engine = require('@smogon/calc').Generations.get(fmtInfo.engineKey);

  // Species pool for this format: must exist in BOTH the fork's data and the
  // calc engine (the champions engine has a 359-species whitelist; the gen9
  // engine table is missing e.g. 'tentooler' — filtering is per-engine).
  let base = POOL_BASE.slice();
  if (fmtInfo.engineKey !== 0 && gen >= 5) base = base.concat(POOL_MODERN);
  if (fmtInfo.engineKey === 0) base = POOL_CHAMPIONS.slice();
  const pool = base.filter((sp) =>
    !!fmtDex.data.Pokedex[toID(sp)] && !!engine.species.get(toID(sp)));
  if (pool.length < 2) throw new Error(`format ${fmtInfo.id}: species pool too small (${pool.length})`);

  return { fmtInfo, fmt, fmtDex, gen, pool };
}

// Ability for a species in a scenario:
//  * gen<3 (engine): no abilities in-game -> null (both engines agree: none).
//  * scenario requires a specific ability (guts/adaptability): must be in the
//    species' table, else the species is blocked for that scenario.
//  * otherwise: first NEUTRAL ability, else blocked (a non-neutral ability would
//    desync sim vs calc and pollute the damage comparison).
// gen1-2 have ONE special stat: @smogon/calc enforces spa === spd on the passed
// EV/IV values, and the fork's separate SpA/SpD bases differ per species. Using
// species with equal special bases makes fork and library agree by construction,
// whatever the library's merge rule is.
function specialEqualBase(sp, ctx) {
  const spData = ctx.fmtDex.data.Pokedex[toID(sp)];
  return !!(spData && spData.baseStats && spData.baseStats.spa === spData.baseStats.spd);
}

function resolveAbility(sp, sc, ctx) {
  if (ctx.gen < 3) return null;
  const spData = ctx.fmtDex.data.Pokedex[toID(sp)];
  // Fork lists abilities by DISPLAY NAME ("Steadfast"); normalize to ids for
  // the sim set (ids) — the calc side re-translates id -> name (node_calc contract).
  const abs = Object.values((spData && spData.abilities) || {}).map((a) => toID(a));
  if (sc && sc.ability) return abs.includes(toID(sc.ability)) ? toID(sc.ability) : null; // null = blocked
  for (const a of abs) if (NEUTRAL.has(a)) return a;
  return null; // blocked
}

function candidateMoves(sp, sc, ctx) {
  const ls = ctx.fmtDex.data.Learnsets && ctx.fmtDex.data.Learnsets[toID(sp)];
  if (!ls || !ls.learnset) return [];
  const mvData = ctx.fmtDex.data.Moves;
  const types = ((ctx.fmtDex.data.Pokedex[toID(sp)]) || {}).types || [];
  return Object.keys(ls.learnset).filter((mv) => {
    const ways = ls.learnset[mv];
    if (!ways.some((w) => w && w[0] === String(ctx.gen))) return false; // learnable in this format's gen
    const m = mvData[mv];
    if (!m || !m.basePower || m.basePower <= 0) return false;
    // Exclude moves whose power is CONDITIONAL (a basePowerCallback): e.g. Stomping
    // Tantrum / Temper Flare double BP when a move failed last turn. That state is
    // battle-history, invisible to a static calc, and the test's Sleep-Talk filler
    // triggers it -> sim inflates the hit and desyncs the band. A principled filter
    // (catches all such moves), not a per-move list.
    if (typeof m.basePowerCallback === 'function') return false;
    if (sc.category && m.category !== sc.category) return false;
    if (sc.type && m.type !== sc.type) return false;
    if (sc.stab && !types.includes(m.type)) return false; // Adaptability only matters on STAB
    if (sc.multihit && !(Array.isArray(m.multihit) && m.multihit[0] >= 2)) return false;
    if (m.target !== 'normal') return false; // single-target only (clean choice syntax)
    const acc = m.accuracy;
    if (acc !== null && acc !== true && acc !== 100) return false; // avoid miss-noise
    if (EXCLUDE_MOVES.has(mv) || /^(max|gmax)/.test(mv)) return false;
    return true;
  });
}

function genMatchup(sc, ctx) {
  const { pool } = ctx;
  const species = shuffle(pool);
  for (const sp of species) {
    const attAbility = resolveAbility(sp, sc, ctx);
    if (sc && sc.ability && !attAbility) continue; // required ability (guts/adaptability) missing
    if (ctx.gen >= 3 && !attAbility) continue; // no neutral ability -> sim/calc desync risk
    let moves = candidateMoves(sp, sc, ctx);
    if (!moves.length) continue;
    if (ctx.gen < 3) {
      // drop special moves unless the species' special bases are equal (see specialEqualBase)
      moves = moves.filter((mv) => {
        const m = ctx.fmtDex.data.Moves[mv];
        return m.category === 'Physical' || specialEqualBase(sp, ctx);
      });
      if (!moves.length) continue;
    }
    const move = pick(moves);
    const moveDef = ctx.fmtDex.data.Moves[move];

    if (ctx.gen < 3 && moveDef.category === 'Special' && !specialEqualBase(sp, ctx)) continue; // attacker special bases must be equal
    // Defender selection: retry until the defender is ACCEPTABLE (not the attacker,
    // a neutral-ability holder for gen3+, not Ground-immune vs a Ground move).
    // Old behaviour accepted the first non-self pick, so a shared module RNG could
    // hand every valid attacker a neutral-less defender and make genMatchup return
    // null mid-sweep (the gen9lc rain-water FAIL) while it passed standalone.
    // Retrying the defender makes the SKIP/FAIL outcome deterministic (null only
    // when the pool genuinely has no acceptable defender).
    let defender = null;
    for (let i = 0; i < 60; i++) {
      const d = pick(pool);
      if (d === sp) continue;
      if (ctx.gen < 3 && moveDef.category === 'Special' && !specialEqualBase(d, ctx)) continue; // defender too (library merges one special stat)
      const da = resolveAbility(d, null, ctx);
      if (ctx.gen >= 3 && !da) continue; // defender needs a neutral ability (else sim/calc desync)
      if (da === 'levitate' && moveDef.type === 'Ground') continue; // ground immunity: keep the band non-degenerate
      defender = d;
      break;
    }
    if (!defender) continue;
    const defAbility = resolveAbility(defender, null, ctx);

    const useNature = ctx.gen >= 4; // natures are a gen4+ mechanic
    const ATT_NATURES = ['Adamant', 'Jolly', 'Hasty', 'Naive', 'Modest', 'Timid', 'Quiet', 'Serious'];
    const DEF_NATURES = ['Bold', 'Impish', 'Calm', 'Careful', 'Relaxed', 'Docile', 'Gentle', 'Lax'];
    const isPhys = moveDef.category === 'Physical';
    let attEvs = isPhys ? { atk: 252 } : { spa: 252 };
    let defEvs = isPhys ? { def: 252 } : { spd: 252 };
    if (ctx.gen < 3 && !isPhys) {
      // @smogon/calc enforces spa === spd on the passed EV values in gen1-2.
      // Species were pre-filtered to equal special bases, so pairing the EVs
      // keeps fork and library like-for-like (base + 63 on both stats).
      attEvs = { spa: 252, spd: 252 };
      defEvs = { spa: 252, spd: 252 };
    }
    return {
      sc, ctx, move, moveDef,
      attacker: sp, defender,
      attNature: useNature ? pick(ATT_NATURES) : null,
      defNature: useNature ? pick(DEF_NATURES) : null,
      attAbility, defAbility,
      attEvs, defEvs,
      boostStat: isPhys ? 'atk' : 'spa',
      fillerA: pool[0] === sp ? pool[1] : pool[0],
      fillerB: pool[0] === defender ? pool[1] : pool[0],
      fillAbility: ctx.gen < 3 ? null : (resolveAbility(pool[0] === sp ? pool[1] : pool[0], null, ctx) || null),
    };
  }
  return null; // no candidate for this scenario in this format (legitimate SKIP)
}

// ── sim side: real headless battle in the fork ───────────────────────────
function runSim(m, seed) {
  const { fmt, fmtInfo } = m.ctx;
  const singles = fmtInfo.gameType === 'singles';
  const mkSet = (sp, o) => {
    const s = { species: sp, level: fmtInfo.level, moves: o.moves };
    if (o.evs) s.evs = o.evs;
    if (o.nature) s.nature = o.nature;
    if (o.ability) s.ability = o.ability;
    if (o.item) s.item = o.item;
    return s;
  };
  const attSet = mkSet(m.attacker, { evs: m.attEvs, nature: m.attNature, ability: m.attAbility, item: m.sc.attItem, moves: [m.move, 'sleeptalk'] });
  const defSet = mkSet(m.defender, { evs: m.defEvs, nature: m.defNature, ability: m.defAbility, moves: m.sc.reflect ? ['reflect', 'sleeptalk'] : ['sleeptalk'] });
  const opts = { format: fmt, debug: true, strictChoices: true, seed: `gen5,${(seed + 1).toString(16).padStart(16, '0')}` };
  if (singles) {
    opts.p1 = { team: [attSet] };
    opts.p2 = { team: [defSet] };
  } else {
    opts.p1 = { team: [attSet, mkSet(m.fillerA, { ability: m.fillAbility, moves: ['sleeptalk'] })] };
    opts.p2 = { team: [defSet, mkSet(m.fillerB, { ability: m.fillAbility, moves: ['sleeptalk'] })] };
  }
  const battle = new Sim.Battle(opts);
  try {
    if (battle.requestState === 'teampreview') battle.makeChoices('team', 'team');
    if (m.sc.weather) battle.field.setWeather(m.sc.weather, 'debug'); // 'debug' source = sim shortcut
    if (m.sc.veil) battle.p2.addSideCondition('auroraveil', battle.p2.active[0]);
    if (m.attStatus) battle.p1.active[0].trySetStatus(m.attStatus, battle.p1.active[0]);
    const statusApplied = !m.attStatus || battle.p1.active[0].status === m.attStatus;
    if (m.sc.boost) battle.p1.active[0].boostBy({ [m.boostStat]: 2 });

    const ch = (a, b) => singles ? [a, b] : [a + ', move sleeptalk', b + ', move sleeptalk'];
    if (m.sc.reflect) battle.makeChoices(...ch('move sleeptalk', 'move reflect')); // turn 1: screen
    const maxhp = battle.p2.active[0].maxhp;
    battle.makeChoices(...ch(`move ${m.move} 1`, 'move sleeptalk'));

    // Actual hit count (multi-hit rolls; the sim logs it on |-hitcount|).
    let hitCount = 1;
    for (const line of battle.log) {
      if (typeof line === 'string' && line.startsWith('|-hitcount|') && line.includes('p2a:')) {
        const n = parseInt(line.split('|')[3], 10); if (n) hitCount = n;
      }
    }
    // Measure the MOVE's damage from the log: skip [from] lines (residual damage)
    // and the "% of max" duplicate lines (denominator must equal the defender's
    // maxhp). Multi-hit moves log one cumulative line, so summing is correct.
    let total = 0, prevHp = maxhp;
    for (const line of battle.log) {
      if (typeof line !== 'string' || !line.startsWith('|-damage|')) continue;
      if (!line.includes('p2a:')) continue;
      if (line.includes('[from]')) continue;
      const parts = line.split('|')[3].split('/');
      const hpStr = parts[0], maxStr = parts[1];
      if (maxStr && parseInt(maxStr, 10) !== maxhp) continue; // drop the "% of max" line
      const hp = parseInt(hpStr, 10);
      total += prevHp - hp;
      prevHp = hp;
    }
    return { damage: total, statusApplied, hitCount };
  } catch (e) {
    console.error(`\n[DEBUG] ${m.ctx.fmtInfo.id} ${m.sc.name}: ${m.attacker} ${m.move} vs ${m.defender}`);
    console.error(battle.log.slice(-30).join('\n'));
    throw e;
  } finally {
    battle.destroy();
  }
}

// ── calc side: @smogon/calc (same code path as damage/node_calc.js) ──────
function runCalc(m, attStatusOverride, hitCount) {
  const ctx = m.ctx;
  const fmtInfo = ctx.fmtInfo;
  const gen = fmtInfo.engineKey;
  const ivs = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
  const mk = (sp, o) => new Pokemon(gen, sp, (() => {
    const out = { level: fmtInfo.level, evs: o.evs || {}, ivs, boosts: o.boosts || {} };
    if (o.nature) out.nature = o.nature;
    if (o.ability) out.ability = abilityName(o.ability);
    if (o.item) out.item = itemName(o.item);
    if (o.status) out.status = o.status;
    return out;
  })());
  const attStatus = attStatusOverride === undefined ? m.attStatus : attStatusOverride;
  const att = mk(m.attacker, { evs: m.attEvs, nature: m.attNature, ability: m.attAbility, item: m.sc.attItem, status: attStatus, boosts: m.sc.boost ? { [m.boostStat]: 2 } : {} });
  const def = mk(m.defender, { evs: m.defEvs, nature: m.defNature, ability: m.defAbility });
  const field = new Field({
    // The calc library defaults gameType to 'Singles' (field.js:23) -> singles
    // screen ratio 0.5. The official doubles rule is a 2/3 reduction (2732/4096)
    // that only activates when gameType !== 'Singles' (gen789.js:1263,
    // champions.js:735). Pass the format's game type so the doubles screen
    // scenarios match the fork, which implements the 2/3 rule.
    gameType: fmtInfo.gameType === 'doubles' ? 'Doubles' : 'Singles',
    weather: m.sc.calcWeather,
    attackerSide: new Side({}),
    defenderSide: (m.sc.reflect && { isReflect: true }) || (m.sc.veil && { isAuroraVeil: true }) || {},
  });
  const mkMove = (crit) => {
    const mvOpts = {}; if (crit) mvOpts.isCrit = true;
    if (hitCount && hitCount > 1) mvOpts.hits = hitCount;
    return new Move(gen, m.move, mvOpts);
  };
  const range = (crit) => calculate(gen, att, def, mkMove(crit), field).range();
  let min, max, critMin, critMax;
  try { [min, max] = range(false); } catch (e) { return { error: String(e) }; }
  try { [critMin, critMax] = range(true); } catch (e) { return { error: String(e) }; }
  return { min, max, critMin, critMax, immune: min === 0 && max === 0, maxhp: def.maxHP() };
}

// ── main ─────────────────────────────────────────────────────────────────
const FORMATS = loadFormats();
const MUST_PASS = (f) => f.engineKey === 9 || f.engineKey === 0; // gen9 + champions
// Verified @smogon/calc LIBRARY gaps (not fork bugs, not fixable in node_modules), keyed
// by engine -> scenario names. Each is a WARN (documented), not a FAIL, so the must-pass
// set keeps meaning while the gap stays visible. Evidence (all verified 2026-11-24):
//   * engine 0 (champions) 'choiceband': champions.js has ZERO choice-item handling
//     (Choice Band/Specs/Scarf all ignored — band identical with/without the item),
//     while the fork and the gen789 engine apply the 1.5x. PRODUCTION IMPACT: the AI's
//     champions damage belief is off by 1.5x for Choice items. See OPEN_DESIGN_CHOICES.md.
const KNOWN_ENGINE_GAPS = { 0: new Set(['choiceband']) };
const isKnownGap = (f, sc) => !!(KNOWN_ENGINE_GAPS[f.engineKey] && KNOWN_ENGINE_GAPS[f.engineKey].has(sc.name));
const results = { pass: 0, warn: 0, fail: 0, skip: 0, rows: [] };

console.log('B1 — all-format calc-vs-sim regression (ground truth: @smogon/calc)');
console.log(`formats: ${FORMATS.length} | must-pass set: engine 9 + 0 (gen9, champions)`);

for (const fmtInfo of FORMATS) {
  let ctx;
  try {
    ctx = makeCtx(fmtInfo);
  } catch (e) {
    results.fail++;
    results.rows.push({ id: fmtInfo.id, scenario: '(setup)', ok: false, note: String(e) });
    console.log(`FAIL  ${fmtInfo.id}: ${e.message}`);
    continue;
  }
  const scenarios = SCENARIOS.filter((sc) => ctx.gen >= sc.minGen);
  let fmtPass = 0, fmtSkip = 0, fmtWarn = 0, fmtFail = 0;
  scenarios.forEach((sc, idx) => {
    let m = null;
    for (let t = 0; t < 30 && !m; t++) m = genMatchup(sc, ctx);
    if (!m) { fmtSkip++; results.skip++; return; } // no legal candidate: documented skip
    let actual = null, c = null, tries = 0;
    let guard = 0;
    while (guard < 6) {
      const r = runSim(m, idx + tries * 7 + guard);
      actual = r.damage;
      c = runCalc(m, (r.statusApplied ? m.attStatus : null), r.hitCount);
      if (c.error) break;
      const ohko = !c.immune && (c.max >= c.maxhp || c.critMin >= c.maxhp);
      if (!ohko && actual > 0) break;
      m = genMatchup(sc, ctx);
      if (!m) break;
      tries++; guard++;
    }
    if (!m || c.error) {
      if (MUST_PASS(fmtInfo)) { fmtFail++; results.fail++; } else { fmtWarn++; results.warn++; }
      results.rows.push({ id: fmtInfo.id, scenario: sc.name, ok: false, note: c.error || 'no usable matchup' });
      return; // forEach: return === continue
    }
    const isMulti = Array.isArray(m.moveDef.multihit);
    let ok, note;
    if (c.immune) { ok = actual === 0; note = ok ? 'IMMUNE (both 0)' : `IMMUNE desync (sim ${actual})`; }
    else if (isMulti) { ok = actual >= c.min && actual <= c.critMax; note = ok ? 'multi-hit band' : 'OUT OF RANGE'; }
    else {
      const inNormal = actual >= c.min && actual <= c.max;
      const inCrit = actual >= c.critMin && actual <= c.critMax;
      ok = inNormal || inCrit;
      note = inNormal ? 'normal' : inCrit ? 'CRIT' : 'OUT OF RANGE';
    }
    if (ok) { fmtPass++; results.pass++; }
    else if (isKnownGap(fmtInfo, sc)) { fmtWarn++; results.warn++; } // verified @smogon/calc library gap (documented WARN, not a fork bug)
    else if (MUST_PASS(fmtInfo)) { fmtFail++; results.fail++; }
    else { fmtWarn++; results.warn++; }
    const gapTag = (!ok && isKnownGap(fmtInfo, sc)) ? ' [KNOWN LIBRARY GAP: @smogon/calc engine-' + fmtInfo.engineKey + ' omits this mechanic (verified); see OPEN_DESIGN_CHOICES.md]' : '';
    results.rows.push({
      id: fmtInfo.id, scenario: sc.name, ok, note: note + gapTag,
      detail: `${m.attacker} ${m.move}(${m.moveDef.basePower}BP,${m.moveDef.category},${m.moveDef.type}) vs ${m.defender}` +
        ` | att ab:${m.attAbility || '-'} item:${m.sc.attItem || '-'} status:${m.attStatus || '-'} def ab:${m.defAbility || '-'}` +
        ` | sim=${actual} calc=${c.min}-${c.max}(crit ${c.critMin}-${c.critMax}) maxhp=${c.maxhp}`,
    });
  });
  const verdict = fmtFail ? 'FAIL' : (fmtWarn ? 'WARN' : 'PASS');
  const tag = MUST_PASS(fmtInfo) ? 'must-pass' : 'best-effort';
  console.log(`${verdict}  ${fmtInfo.id.padEnd(28)} eng=${fmtInfo.engineKey} L=${fmtInfo.level} ${fmtInfo.gameType.padEnd(8)} [${tag}]  pass=${fmtPass} warn=${fmtWarn} fail=${fmtFail} skip=${fmtSkip}`);
}

console.log('\n────────────────────────────────────────────────────────────');
for (const r of results.rows) {
  if (r.ok) continue;
  console.log(`${r.id.padEnd(28)} ${r.scenario.padEnd(16)} ${r.note}${r.detail ? ' | ' + r.detail : ''}`);
}
console.log(`\nTOTAL: ${results.pass} pass, ${results.warn} warn(old-gen/known-gap), ${results.fail} FAIL, ${results.skip} skip(no candidate)`);
console.log(results.fail ? 'VERDICT: FAIL' : 'VERDICT: PASS (old-gen + known library gaps reported as warn)');
process.exit(results.fail ? 1 : 0);
