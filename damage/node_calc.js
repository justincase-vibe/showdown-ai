'use strict';
// Step 2 — Damage calculator wrapper (Node side).
//
// Thin CLI over the official @smogon/calc damage model. It reads ONE JSON request
// from stdin and writes ONE JSON response to stdout, so the Python side
// (calc.py) can treat it as a plain subprocess. The @smogon/calc library is the
// single source of truth for the numbers; this file only marshals the request
// into Pokemon/Move/Field objects and normalises the result.
//
// Request types:
//   { type:"damage", gen, attacker:{...}, defender:{...}, move:{name,isCrit?}, field?{...} }
//     -> { min,max,expected, min_pct,max_pct,expected_pct,
//          crit:{min,max,expected}, immune, maxhp, desc }
//   { type:"speed", gen, pokemon:{...}, tailwind?, field?{weather?,terrain?} }
//     -> { base, speed }
//
// attacker/defender/pokemon option keys follow @smogon/calc: level, nature, evs,
// ivs, item, ability, status, boosts, teraType, moves, plus the belief-state
// flags abilityOn (bool: Unburden/Slow Start active, QP-eligible) and
// boostedStat ('auto' or a stat name: Quark Drive / Protosynthesis). field keys:
// weather (capitalised: "Rain"/"Sun"/"Sand"/"Hail"), terrain, attackerSide,
// defenderSide.

const path = require('path');
const { calculate, Pokemon, Move, Field, Side, Generations, toID } = require('@smogon/calc');
// NOTE: `Generations` is an INSTANCE (not a class) exposing .get(gen).
function speciesInEngine(gen, name) {
  return !!Generations.get(gen).species.get(toID(name));
}

// @smogon/calc resolves `item` and `ability` by DISPLAY NAME (its item/ability
// tables are keyed by name, not id). The AI / legality DB and the sim fork all
// carry IDS ("lifeorb", "adaptability"), so passing an id is SILENTLY IGNORED
// by the library -> wrong damage. Species and move names are normalised by the
// library itself (id or name both work), so only item/ability need translating.
// Ground truth for id -> name: the fork's own items/abilities tables.
function loadId2Name(file, tableKey) {
  try {
    const mod = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'data', file));
    const table = (mod && (mod[tableKey])) || {};
    const m = {};
    for (const id of Object.keys(table)) if (table[id] && table[id].name) m[id] = table[id].name;
    return m;
  } catch (e) { return {}; }
}
const ITEM_ID2NAME = loadId2Name('items', 'Items');
const ABILITY_ID2NAME = loadId2Name('abilities', 'Abilities');
// Accept ids AND display names: translate id->name when known, else pass through
// (so a display name that isn't a fork id still works, and the Step 2 tests that
// pass names keep passing).
const normItem = (x) => (x && ITEM_ID2NAME[x]) || x;
const normAbility = (x) => (x && ABILITY_ID2NAME[x]) || x;

function clean(obj) {
  if (!obj) return {};
  const out = {};
  for (const k of Object.keys(obj)) if (obj[k] !== undefined) out[k] = obj[k];
  return out;
}

function buildPokemon(gen, o) {
  if (!o || !o.name) throw new Error('pokemon requires a "name"');
  // Engines with a legal-species whitelist (e.g. the champions/BSS engine, gen
  // key 0: 359 BSS-legal species) have NO entry for species outside the list;
  // the library then crashes opaquely inside its Pokemon constructor (reading
  // 'hp' of undefined). Fail loudly with a clean error instead — an illegal
  // species must never silently produce numbers.
  if (!speciesInEngine(gen, o.name)) {
    throw new Error(`pokemon "${o.name}" is not legal in the gen-${gen} engine (not in its species table)`);
  }
  const opts = clean({
    level: o.level,
    nature: o.nature,
    evs: o.evs || {},
    ivs: o.ivs || { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 },
    item: normItem(o.item),
    ability: normAbility(o.ability),
    status: o.status,
    boosts: o.boosts || {},
    teraType: o.teraType,
    moves: o.moves,
    // belief-state flags (library options; absent = off, as before):
    abilityOn: o.abilityOn,
    boostedStat: o.boostedStat,
  });

  // Current HP (optional). @smogon/calc gates several damage-affecting effects on
  // a mon's CURRENT HP, which the library otherwise assumes to be FULL:
  //   attacker  Defeatist (<=1/2, Atk&SpA x0.5), Overgrow/Blaze/Torrent/Swarm
  //             (<=1/3, matching-type x1.5), Eruption/Water Spout/Flail/Reversal
  //             (base power scales with own HP);
  //   defender  Multiscale/Shadow Shield/Tera Shell (only AT FULL HP, x0.5),
  //             Brine (<=1/2, x1.5), Crush Grip/Wring Out/Hard Press (BP scales
  //             with defender HP).
  // So with no curHP the low-HP attacker effects never fire and the full-HP-only
  // defender effects always do. The AI's belief state knows each mon's HP as a
  // fraction, so we accept:
  //   hp     -> 0.0-1.0 fraction of THIS mon's max HP  (AI-facing; level/IV/EV-agnostic)
  //   curHP  -> absolute HP (clamped to [1, maxHP])      (library-facing alias)
  // Omitting both preserves full-HP semantics (unchanged from before).
  const maxHP = new Pokemon(gen, o.name, opts).maxHP();
  if (o.hp !== undefined && o.hp !== null) {
    const frac = Number(o.hp);
    if (Number.isNaN(frac) || frac < 0 || frac > 1) {
      throw new Error('hp must be a 0.0-1.0 fraction (got ' + o.hp + ')');
    }
    opts.curHP = Math.max(1, Math.floor(frac * maxHP));
  } else if (o.curHP !== undefined && o.curHP !== null) {
    const abs = Number(o.curHP);
    if (Number.isNaN(abs) || abs <= 0) {
      throw new Error('curHP must be > 0 (got ' + o.curHP + ')');
    }
    opts.curHP = Math.max(1, Math.min(maxHP, Math.floor(abs)));
  }
  return new Pokemon(gen, o.name, opts);
}

function buildField(f) {
  f = f || {};
  return new Field({
    weather: f.weather,
    terrain: f.terrain,
    attackerSide: new Side(f.attackerSide || {}),
    defenderSide: new Side(f.defenderSide || {}),
  });
}

// ── Speed ─────────────────────────────────────────────────────────────────────
// FULL mirror of @smogon/calc `getFinalSpeed` (dist/mechanics/util.js:131) —
// same tables, same order, same rounding (chainMods + pokeRound) — so the AI's
// turn-order predictions match the official calc exactly:
//   boosts (getModifiedStat, gen-aware) -> Tailwind (x2) -> ability chain
//   (Unburden+abilityOn / Chlorophyll+Sun / Sand Rush+Sand / Swift Swim+Rain /
//   Slush Rush+Hail|Snow / Surge Surfer+Electric terrain: x2; else Quick Feet
//   +status: x1.5; else Slow Start+abilityOn: x0.5; else Quark Drive /
//   Protosynthesis: x1.5 when the boosted stat is Spe) -> item chain (Choice
//   Scarf x1.5; else Iron Ball or an EV item x0.5; else Quick Powder on Ditto
//   x2) -> THEN paralysis (Quick Feet exempt) -> cap 999 (gen 1-2) / 10000.
//
// Paralysis factor per ENGINE — FORK-EXACT (fork `par.onModifySpe`):
//   base conditions.js x0.5 (gens 5-9 + champions), mods/gen4/conditions.js:30
//   x0.25 (gens 1-4, inherited), mods/gen6/conditions.js:31 x0.25 (gen 6).
//   @smogon/calc agrees on every engine EXCEPT gen 5 (library x0.25 vs fork
//   x0.5) — the fork is the battle environment the AI fights in, so the fork
//   value wins (documented in calc.py / PLAN.md D42).
const PAR_SPEED_PCT = { '0': 50, '1': 25, '2': 25, '3': 25, '4': 25, '5': 50, '6': 25, '7': 50, '8': 50, '9': 50 };
// @smogon/calc util.js EV_ITEMS (the "Iron Ball or an EV item" speed x0.5 branch).
const EV_ITEMS = ['Macho Brace', 'Power Anklet', 'Power Band', 'Power Belt', 'Power Bracer', 'Power Lens', 'Power Weight'];
// library pokeRound (util.js:696) — half-up; OF16 (util.js:700) — 16-bit overflow.
const pokeRound = (n) => (n % 1 > 0.5 ? Math.ceil(n) : Math.floor(n));
const OF16 = (n) => (n > 65535 ? n % 65536 : n);

// Mirrors @smogon/calc `getModifiedStat` (util.js:58) exactly: gen 1-2 use the
// explicit % numerators (and cap 999); modern gens use the [num,den] table.
function modifiedStat(stat, boost, gen) {
  const b = Math.max(-6, Math.min(6, boost || 0));
  if (Number(gen) === 1 || Number(gen) === 2) {
    if (b >= 0) stat = Math.floor(stat * [1, 1.5, 2, 2.5, 3, 3.5, 4][b]);
    else stat = Math.floor((stat * [100, 66, 50, 40, 33, 28, 25][-b]) / 100);
    return Math.min(999, Math.max(1, stat));
  }
  const [n, d] = [[2, 8], [2, 7], [2, 6], [2, 5], [2, 4], [2, 3], [2, 2], [3, 2], [4, 2], [5, 2], [6, 2], [7, 2], [8, 2]][6 + b];
  stat = OF16(stat * n);
  return Math.floor(stat / d);
}

// Mirrors @smogon/calc `isQPActive` / `getQPBoostedStat` (util.js:512-549):
// Quark Drive (Electric terrain or Booster Energy) and Protosynthesis (Sun or
// Booster Energy) boost the mon's highest stat x1.5. The belief state reports
// `boostedStat` ('auto' or a stat name); absent => the QP effect is off, exactly
// as in the library (which keys off the same option).
function qpActive(p, weather, terrain) {
  if (!p.boostedStat) return false;
  return ((p.hasAbility('Protosynthesis') && (weather.includes('Sun') || p.hasItem('Booster Energy'))) ||
          (p.hasAbility('Quark Drive') && (terrain === 'Electric' || p.hasItem('Booster Energy'))) ||
          p.boostedStat !== 'auto');
}
function qpBoostedStat(p, gen) {
  if (p.boostedStat && p.boostedStat !== 'auto') return p.boostedStat;
  let best = 'atk';
  for (const s of ['def', 'spa', 'spd', 'spe']) {
    if (modifiedStat(p.rawStats[s], (p.boosts && p.boosts[s]) || 0, gen) >
        modifiedStat(p.rawStats[best], (p.boosts && p.boosts[best]) || 0, gen)) best = s;
  }
  return best;
}

// Effective action speed — see the block comment above for the full model.
function effectiveSpeed(gen, o, tailwind, field) {
  const p = buildPokemon(gen, o);
  const f = field || {};
  const weather = f.weather || '';
  const terrain = f.terrain || '';
  let speed = modifiedStat(p.rawStats.spe, (p.boosts && p.boosts.spe) || 0, gen);
  const mods = [];
  if (tailwind) mods.push(8192);
  if ((p.hasAbility('Unburden') && p.abilityOn) ||
      (p.hasAbility('Chlorophyll') && weather.includes('Sun')) ||
      (p.hasAbility('Sand Rush') && weather === 'Sand') ||
      (p.hasAbility('Swift Swim') && weather.includes('Rain')) ||
      (p.hasAbility('Slush Rush') && (weather === 'Hail' || weather === 'Snow')) ||
      (p.hasAbility('Surge Surfer') && terrain === 'Electric')) {
    mods.push(8192);
  } else if (p.hasAbility('Quick Feet') && p.status) {
    mods.push(6144);
  } else if (p.hasAbility('Slow Start') && p.abilityOn) {
    mods.push(2048);
  } else if (qpActive(p, weather, terrain) && qpBoostedStat(p, gen) === 'spe') {
    mods.push(6144);
  }
  if (!(p.hasAbility('Unburden') && p.abilityOn)) {
    if (p.hasItem('Choice Scarf')) {
      mods.push(6144);
    } else if (p.hasItem('Iron Ball', ...EV_ITEMS)) {
      mods.push(2048);
    } else if (p.hasItem('Quick Powder') && p.named && p.named('Ditto')) {
      mods.push(8192);
    }
  }
  // library chainMods (util.js:487) + pokeRound — exact for these dyadic mods.
  let M = 4096;
  for (const m of mods) if (m !== 4096) M = (M * m + 2048) >> 12;
  speed = pokeRound((speed * M) / 4096);
  if (p.hasStatus('par') && !p.hasAbility('Quick Feet')) {
    speed = Math.floor(speed * (PAR_SPEED_PCT[String(gen)] || 50) / 100);
  }
  return Math.max(0, Math.min(Number(gen) <= 2 ? 999 : 10000, speed));
}

// ── Special-damage moves (non-basePower formulas) ──────────────────────
// @smogon/calc v0.12.0 returns 0 for these (they're absent from its mechanics).
// We COMPLETE the calc with the official Smogon rules (Bulbapedia spec), fork-
// verified integer-exact (scratch/sweep_t1.js). @smogon/calc stays the source of
// truth for every other move; this only fills its documented gaps (D37). Rounding
// mirrors the fork: damage is Math.floored then clamped to >= 1 (battle.js:1835).
//
// lastHit = the damage the USER last took (belief-state state). Required by the
// reflectors; without it the move fails (0 damage), matching the fork's thisTurn gate.
// Gen-aware: these moves still respect TYPE IMMUNITY exactly like any other move
// (fork `runImmunity` precedes the formula dispatch): e.g. Super Fang (Normal) vs
// Ghost (gen 2+), Night Shade (Ghost) vs Normal, Fissure (Ground) vs Flying/
// Levitate, Psywave (Psychic) vs Dark. The fork does NOT apply 0.5x/2x
// multipliers or STAB to formula moves (fork-verified: Night Shade vs Ghost =
// level, not 2x level; Super Fang vs Steel/Rock = plain half HP). In gen 1 some
// formula moves carry ignoreImmunity (fork data: Super Fang / Night Shade /
// Seismic Toss hit Ghost types there). Strong Jaw does NOT boost them (no base
// power). All immunity facts come from special_move_immunity.json (fork-exact,
// see below).
const SPECIAL_DAMAGE_MOVES = new Set([
  'superfang', 'ruination', 'naturesmadness', 'guardianofalola', 'finalgambit',
  'endeavor', 'psywave', 'counter', 'mirrorcoat', 'metalburst', 'comeuppance',
  'nightshade', 'seismictoss', 'fissure', 'guillotine', 'horndrill', 'sheercold',
]);

// Type-immunity facts for these moves, precomputed PER ENGINE KEY from the
// fork's own dex data (type chart + move type + ignoreImmunity + ohko) — the
// fork is the source of truth for raw game data and the battle environment the
// calc must match. Regenerate with:  node scratch/gen_special_move_immunity.js
// (schema: { "<engine key>": { "<move id>": { "immune": [defender types],
//   "ground"?: true, "ohkoIce"?: true } } }).
//
// Why a table instead of recomputing from the chart at call time: (a) it is a
// static lookup — the AI calls this deep in its search tree and the calc must
// stay quick; (b) it captures fork quirks a hand-rolled chart walk misses (gen 1
// ignoreImmunity, per-gen move types); (c) it keeps the calc fork-exact by
// construction, since the fork's runImmunity is a pure function of (gen, move,
// defender types/ability/item).
const SPECIAL_MOVE_IMMUNITY = require('./special_move_immunity.json');

// ── T6: recoil / drain (attacker-side self-damage and healing) ─────────────
// @smogon/calc returns only the DAMAGE DEALT (defender side) and does not model
// move recoil, drain healing, or Life Orb self-damage. The AI needs those for
// move evaluation (e.g. "Brave Bird KOs but I lose 33% — is it worth it?"), so
// we COMPLETE the calc with the fork-exact attacker-side effects. The fork is
// the source of truth for the raw move/item data (per-engine values) and the
// application semantics (verified in dist code):
//   recoil (applyRecoilDamage; called ONLY when the move DEALT damage —
//   miss/immune => none). TWO formula variants exist in the fork: gens 5-9
//   and the champions engine use the base version (battle-actions.js:1216,
//   Math.round); gens 1-4 inherit the gen4 override (mods/gen4/scripts.js:167,
//   Math.floor — inheritance chain gen1->gen2->gen3->gen4). Live-verified:
//   gen1 Struggle dealt 29, recoil [1,2] -> floor(14.5) = 14 (not round = 15):
//     struggleRecoil        -> g1-4 clamp(floor(baseMaxhp/4),1)
//                              g5-9/champ clamp(round(baseMaxhp/4),1)
//     mindBlown/chloroblast -> round(maxhp/2)  (same in both variants)
//     recoil:[n,d]          -> g1-4 clamp(floor(damageDealt*n/d),1)
//                              g5-9/champ clamp(round(damageDealt*n/d),1)
//   drain (battle.js:1793-1800, inside battle.damage() so only on damage dealt):
//     gen<=4 -> clamp(floor(damageDealt*n/d), 1) ; gen>4 -> round(damageDealt*n/d)
//   Life Orb (items.js onAfterMoveSecondarySelf, uniform across this fork's
//   gens incl. gen4/gen6 mods): baseMaxhp/10 (10% of the user's max HP), gated
//   on `move.category !== "Status"` (and `!move.ohko` in the gen6 mod).
// Static per-engine table (regenerate: node scratch/gen_recoil_drain.js), same
// pattern as special_bp_moves.json — quick + API-reachable, no per-call fork.
const RECOIL_DRAIN = require('./recoil_drain_moves.json');
const OHKO_MOVES = new Set(['fissure', 'guillotine', 'horndrill', 'sheercold']);

const _metaCache = new Map();
// Fork-exact move metadata (the fork is the source of truth for raw game
// data), lazily loaded per (engine, move) and cached:
//   category  — the Life Orb "Status moves don't trigger it" guard;
//   critRatio — the fork's dex default is `Number(data.critRatio) || 1`, i.e.
//               EVERY base-power move rolls a base crit chance (official
//               behaviour: 1/24 in gens 7-9/champions, 1/16 in gens <= 6);
//   willCrit  — always-crit (Frost Breath, Storm Throw) / never-crit moves.
function moveMeta(gen, moveName) {
  const key = String(gen) + ':' + moveName;
  if (_metaCache.has(key)) return _metaCache.get(key);
  const meta = { category: 'Damage', critRatio: null, willCrit: null };
  try {
    const Sim = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));
    const fmt = ({ '0': 'bss', '1': 'gen1randombattle', '2': 'gen2randombattle', '3': 'gen3randombattle',
      '4': 'gen4randombattle', '5': 'gen5randombattle', '6': 'gen6randombattle', '7': 'gen7randombattle',
      '8': 'gen8randombattle', '9': 'gen9randombattle' })[String(gen)];
    const m = Sim.Dex.forFormat(fmt).moves.get(moveName);
    if (m) {
      if (m.category) meta.category = m.category;
      meta.critRatio = Number(m.critRatio) || 1;
      meta.willCrit = m.willCrit === true ? 'always' : (m.willCrit === false ? 'never' : null);
    }
  } catch (e) { /* unknown move -> defaults (damaging, unknown crit data) */ }
  _metaCache.set(key, meta);
  return meta;
}

// Fork critical-hit rate tables (battle-actions.js getDamage): rate = 1/critMult[ratio],
// ratio clamped to the table. gen <= 5: [0,16,8,4,3,2] ; gen 6: [0,16,8,2,1] ;
// gens 7-9 AND the champions engine (gen 9 + mod): [0,24,8,2,1]. Verified live
// (champions Brave Bird crits at 1.5x, base roll consistent with 1/24).
const CRIT_MULT = {
  '0': [0, 24, 8, 2, 1], '1': [0, 16, 8, 4, 3, 2], '2': [0, 16, 8, 4, 3, 2],
  '3': [0, 16, 8, 4, 3, 2], '4': [0, 16, 8, 4, 3, 2], '5': [0, 16, 8, 4, 3, 2],
  '6': [0, 16, 8, 2, 1], '7': [0, 24, 8, 2, 1], '8': [0, 24, 8, 2, 1], '9': [0, 24, 8, 2, 1],
};
// BASE critical-hit chance for a base-power move in this engine (fork-exact),
// or null when the fork would never roll a crit for it. Battle-state modifiers
// (Super Luck, Scope Lens, Focus Energy, Laser Focus, Merciless, Lucky Chant,
// Battle/Shell Armor) are belief-state facts the AI applies on top — the calc
// reports the fork's base rate only, same pattern as the T5 state flags.
function baseCritChance(gen, moveName) {
  const meta = moveMeta(gen, moveName);
  if (meta.willCrit === 'never') return 0;
  const mult = CRIT_MULT[String(gen)];
  if (!mult || meta.critRatio === null) return null;
  const r = Math.max(0, Math.min(meta.critRatio, mult.length - 1));
  return mult[r] === 0 ? null : 1 / mult[r];
}

// Attacker-side blocks for a damage response. The fork applies recoil/drain to
// the damage ACTUALLY DEALT, and it deals damage in two ways that change the
// span of outcomes (both fork-verified):
//   1. `battle.damage` clamps dealt damage to the target's REMAINING HP (an
//      overkill hit deals less), and recoil/drain are computed on that clamped
//      value (live: champions Brave Bird computed 237 vs 154-HP Blastoise ->
//      dealt 154 -> recoil round(154*33/100) = 51, not round(237*33/100) = 78);
//   2. every base-power move rolls a base crit chance (see baseCritChance) and
//      a crit deals more — so the MAX recoil/heal spans the crit outcome.
// The response therefore brackets [f(min(normalMin, remHP)), f(min(critMax, remHP))].
// `heal` is the healing amount (net gain is additionally capped by the user's
// remaining HP — the AI's belief state knows current HP and caps it).
function selfEffectBlocks(gen, moveId, moveName, attMaxHP, dmgMin, dmgMax, critMax, defRemHP, attackerItem) {
  const out = {};
  const row = (RECOIL_DRAIN[String(gen)] || {})[moveId];
  const genNum = Number(gen);
  const lo = Math.max(0, Math.min(dmgMin, defRemHP));
  const hi = Math.max(0, Math.min(critMax || dmgMax, defRemHP));
  if (row) {
    if (row.struggleRecoil) {
      // g1-4: FLOOR (gen4 override) ; g5-9/champions: ROUND (base)
      const v = (genNum >= 1 && genNum <= 4)
        ? Math.max(1, Math.floor(attMaxHP / 4))
        : Math.max(1, Math.round(attMaxHP / 4));
      out.recoil = { min: v, max: v };
    } else if (row.mindBlownRecoil || row.chloroblastRecoil) {
      const v = Math.round(attMaxHP / 2);
      out.recoil = { min: v, max: v };
    } else if (row.recoil && hi > 0) {
      const [n, d] = row.recoil;
      const f = (dmg) => (dmg > 0
        ? ((genNum >= 1 && genNum <= 4) ? Math.max(1, Math.floor(dmg * n / d)) : Math.max(1, Math.round(dmg * n / d)))
        : 0);
      out.recoil = { min: f(lo), max: f(hi) };
    }
    if (row.drain && hi > 0) {
      const [n, d] = row.drain;
      const f = (dmg) => (dmg > 0
        ? ((genNum >= 1 && genNum <= 4) ? Math.max(1, Math.floor(dmg * n / d)) : Math.round(dmg * n / d))
        : 0);
      out.heal = { min: f(lo), max: f(hi) };
    }
  }
  if (attackerItem && normItem(attackerItem) === 'Life Orb' && hi > 0) {
    const ohko = genNum === 6 && OHKO_MOVES.has(moveId); // gen6 mod guard `!move.ohko`
    if (moveMeta(gen, moveName).category !== 'Status' && !ohko) {
      const v = Math.floor(attMaxHP / 10);
      if (v > 0) out.lifeOrb = { min: v, max: v };
    }
  }
  return out;
}

// Effective defender types for the immunity checks below. The fork's
// runImmunity path (dex.getImmunity, dist/sim/dex.js:226) uses
// `target.getTypes()`, which returns the TERASTALLIZED type (replacing the
// base types) when the mon is terastallized and not Stellar
// (dist/sim/pokemon.js:1583) — and @smogon/calc's effectiveness overrides with
// `defender.teraType` the same way (dist/mechanics/gen789.js:292-294).
// Terastallization only exists in this fork's gen 9 and the champions engine,
// so the override applies there alone (a teraType passed for gen 1-8 is
// ignored, as in the library's gen 1-6 mechanics).
function effectiveTypes(gen, mon) {
  const base = (mon.species && mon.species.types) || [];
  const t = mon.teraType;
  if ((Number(gen) === 9 || Number(gen) === 0) && t && String(t) !== 'Stellar') return [String(t)];
  return base;
}

// True when the fork's runImmunity (pokemon.js:1664) would block this special
// move against this defender. Chart immunities come from the table; the two
// request-data-dependent rules are applied here, mirroring the fork:
//   * Ground moves: immune vs a grounded check — the defender is AIRBORNE (so
//     immune) when Flying-typed, or has Levitate/Eelevate, or holds Air
//     Balloon; Iron Ball grounds it (fork isGrounded, pokemon.js:1592).
//     Volatile-based grounding (Gravity/Ingrain/Smackdown/Magnet Rise/
//     Telekinesis) is battle state the belief state would have to report; the
//     static calc assumes they are not active.
//   * Sheer Cold: its "ohko: Ice" rule makes Ice types immune (the chart only
//     resists Ice 0.5x) — handled in the move's own case below.
// Tera-aware: a terastallized defender's Tera type stands in for its base
// types (see effectiveTypes) — matching BOTH the fork's getTypes-based
// immunity and the library's own effectiveness pipeline.
function specialMoveImmune(gen, moveId, defender) {
  const entry = (SPECIAL_MOVE_IMMUNITY[String(gen)] || {})[moveId];
  if (!entry) return false;
  const defTids = effectiveTypes(gen, defender).map(toID);
  if (entry.ground) {
    // Fork: Ground moves bypass the type chart entirely — only the grounded
    // check applies (isGrounded), and its FIRST rule is Iron Ball -> grounded
    // (which therefore overrides Flying type AND Levitate).
    const item = String(defender.item || '').toLowerCase();
    if (item === 'iron ball' || item === 'ironball') return false;
    if (defTids.includes('flying')) return true;
    const ability = String(defender.ability || '').toLowerCase();
    if (ability === 'levitate' || ability === 'eelevate') return true;
    if (item === 'air balloon' || item === 'airballoon') return true;
    return false;
  }
  const immuneTids = (entry.immune || []).map(toID);
  return defTids.some((t) => immuneTids.includes(t));
}

function specialMoveDamage(gen, moveId, attacker, defender, lastHit) {
  // -> { min, max, immune, ohko } or null (not a special move we handle).
  const defHP = defender.curHP ? defender.curHP() : defender.maxHP();
  const defMaxHP = defender.maxHP ? defender.maxHP() : (defHP || 1);
  const attHP = attacker.curHP ? attacker.curHP() : attacker.maxHP();
  // Tera-aware Ice check for Sheer Cold (see effectiveTypes): a terastallized
  // Ice defender is immune even if its base types are not Ice.
  const hasIce = effectiveTypes(gen, defender).some((t) => toID(t) === 'ice');
  const attLevel = attacker.level || 100;

  // Type-immunity gate: the fork's runImmunity applies to EVERY move, formula
  // moves included (see specialMoveImmune). Multipliers (0.5x/2x) and STAB are
  // deliberately NOT applied (fork-verified immunity-only behaviour for
  // non-basePower moves).
  if (specialMoveImmune(gen, moveId, defender)) {
    return { min: 0, max: 0, immune: true };
  }

  switch (moveId) {
    case 'superfang':
      return { min: Math.max(1, Math.floor(defHP / 2)), max: Math.max(1, Math.floor(defHP / 2)), immune: false };
    case 'ruination':
    case 'naturesmadness':
      return { min: Math.max(1, Math.floor(defHP / 2)), max: Math.max(1, Math.floor(defHP / 2)), immune: false };
    case 'guardianofalola':
      return { min: Math.max(1, Math.floor(defHP * 3 / 4)), max: Math.max(1, Math.floor(defHP * 3 / 4)), immune: false };
    case 'finalgambit':
      return { min: attHP, max: attHP, immune: false };
    case 'endeavor': {
      const d = Math.max(0, defHP - attHP);
      return { min: d, max: d, immune: d === 0 };
    }
    case 'psywave': // Past (gen 2-5): random(50,151) * level/100
      return { min: Math.max(1, Math.floor(50 * attLevel / 100)), max: Math.floor(150 * attLevel / 100), immune: false };
    case 'counter':
    case 'mirrorcoat':
      if (lastHit == null || lastHit <= 0) return { min: 0, max: 0, immune: true };
      return { min: 2 * Math.round(lastHit), max: 2 * Math.round(lastHit), immune: false };
    case 'metalburst':
    case 'comeuppance':
      if (lastHit == null || lastHit <= 0) return { min: 0, max: 0, immune: true };
      return { min: Math.max(1, Math.floor(1.5 * lastHit)), max: Math.max(1, Math.floor(1.5 * lastHit)), immune: false };
    case 'nightshade':
    case 'seismictoss':
      // T2: damage = attacker's level (fork: move.damage === "level" -> source.level).
      // Unified for both (the calc covered Seismic Toss but not Night Shade).
      return { min: attLevel, max: attLevel, immune: false };
    case 'fissure':
    case 'guillotine':
    case 'horndrill':
      // T3 OHKO: guaranteed KO on hit (fork: move.ohko -> target.maxhp). Accuracy is
      // in the move data; the calc reports the KO so the AI knows it faints the target.
      return { min: defMaxHP, max: defMaxHP, immune: false, ohko: true };
    case 'sheercold':
      // OHKO "Ice": immune vs Ice types (fork hitStepAccuracy [ohko]), else a KO.
      if (hasIce) return { min: 0, max: 0, immune: true };
      return { min: defMaxHP, max: defMaxHP, immune: false, ohko: true };
  }
  return null;
}

// ── T5: state-dependent base-power moves (bpCallback) ──────────────────────────
// @smogon/calc's gen789 mechanics has NO formula for these (they fall to
// `default: basePower = move.bp`, i.e. the STATIC base power from the data). For
// the moves whose data base power is 0 (beatup, spitup) the library therefore
// returns 0 damage — a hard belief-state bug. For the others it returns the
// correct no-state baseline but cannot see the transient battle state that the
// fork uses to re-scale them (hit-this-turn, last-move-failed, pledge combo,
// fainted allies, stockpile layers).
//
// We COMPLETE the calc the same way as the reflectors (D37): read the fork's
// per-engine-key base power (special_bp_moves.json — the fork is the source of
// truth for raw move data; champions bans the pledges so they are null there),
// compute the effective base power from the belief-state flags (req.move.*), and
// hand it to the library via `overrides.basePower`. The library then applies
// STAB / typing / boosts / immunities / crits exactly as for any normal move, so
// this stays the ONE unified damage pipeline (no second formula to maintain).
//
// State flags (all optional; absent = the no-state baseline the fork would use):
//   move.hitThisTurn      bool  avalanche: user was hit by the target this turn (x2)
//   move.lastMoveFailed   bool  stompingtantrum/temperflare: user's previous move failed (x2)
//   move.pledgeCombo      bool  fire/grass/waterpledge: a compatible pledge already hit this turn (BP 150)
//   move.allyFainted      int   lastrespects: number of the user's fainted allies (+50 BP each)
//   move.stockpileLayers  int   spitup: Stockpile layers (+100 BP each; 0 => the move fails)
//   move.beatupAllies     array beatup: participating allies (incl. user); each deals
//                         5 + floor(baseAtk/10) BP using its OWN stats
const SPECIAL_BP_MOVES = require('./special_bp_moves.json');
const STATE_BP_MOVES = new Set([
  'avalanche', 'beatup', 'firepledge', 'grasspledge', 'waterpledge',
  'lastrespects', 'spitup', 'stompingtantrum', 'temperflare',
]);

// Effective base power for a single-hit state-BP move, or null if the move is
// unusable/banned in this engine (the caller reports 0/immune — the normal
// path crashes on the banned/inherit move data). Fork formulas: avalanche/ST/TF = basePower * 2 (relative); pledges =
// absolute 150 on combo; lastrespects = 50 + 50*fainted (absolute); spitup =
// 100*layers (absolute, 0 => fails). See scratch/probe_t5_formulas.js.
function effectiveBP(gen, moveId, st) {
  const row = SPECIAL_BP_MOVES[String(gen)] || {};
  const baseBP = row[moveId];
  if (baseBP === null || baseBP === undefined) return null;
  switch (moveId) {
    case 'avalanche':        return baseBP * (st.hitThisTurn ? 2 : 1);
    case 'stompingtantrum':
    case 'temperflare':      return baseBP * (st.lastMoveFailed ? 2 : 1);
    case 'firepledge':
    case 'grasspledge':
    case 'waterpledge':      return st.pledgeCombo ? 150 : baseBP;
    case 'lastrespects':     return 50 + 50 * (st.allyFainted || 0);
    case 'spitup':           return 100 * (st.stockpileLayers || 0);
  }
  return null;
}

// Damage range for a state-BP move (normal + crit), or null if not applicable.
// beatup is multi-hit with a per-ally base power (5 + floor(baseAtk/10)); the
// total is the sum of each ally's own-stats hit (fork onModifyMove: allies are
// the user + non-fainted, non-statused party).
function stateMoveDamage(gen, moveId, attacker, defender, st, field) {
  if (moveId === 'beatup') {
    const allies = st.beatupAllies;
    if (!Array.isArray(allies) || allies.length === 0) return { min: 0, max: 0, immune: true, crit: { min: 0, max: 0 } };
    let nMin = 0, nMax = 0, cMin = 0, cMax = 0;
    for (const ally of allies) {
      const att = buildPokemon(gen, ally);
      const baseAtk = (att.species && att.species.baseStats && att.species.baseStats.atk) || (ally.baseAtk || 0);
      const bp = 5 + Math.floor(baseAtk / 10);
      for (const crit of [false, true]) {
        try {
          const mv = new Move(gen, 'beatup', crit ? { overrides: { basePower: bp }, isCrit: true } : { overrides: { basePower: bp } });
          const [mn, mx] = calculate(gen, att, defender, mv, field).range();
          if (crit) { cMin += mn; cMax += mx; } else { nMin += mn; nMax += mx; }
        } catch (e) { /* immune or unsupported -> contributes 0 */ }
      }
    }
    return { min: nMin, max: nMax, immune: nMax === 0, crit: { min: cMin, max: cMax } };
  }

  const bp = effectiveBP(gen, moveId, st);
  // Unusable/banned in this engine (e.g. pledges in champions: isNonstandard),
  // or no state to act on (spitup with 0 layers) -> report 0/immune. MUST NOT
  // return null here: the normal rangeFor path crashes on the banned/inherit
  // move data (Cannot read properties of undefined (reading 'hp')).
  if (bp === null || bp <= 0) return { min: 0, max: 0, immune: true, crit: { min: 0, max: 0 } };
  const range = (crit) => {
    try {
      const mv = new Move(gen, moveId, crit ? { overrides: { basePower: bp }, isCrit: true } : { overrides: { basePower: bp } });
      const [mn, mx] = calculate(gen, attacker, defender, mv, field).range();
      return { min: mn, max: mx };
    } catch (e) { return { min: 0, max: 0 }; }
  };
  const n = range(false), c = range(true);
  return { min: n.min, max: n.max, immune: n.max === 0, crit: { min: c.min, max: c.max } };
}

// % of defender max HP, rounded to 2 decimals (shared by all response paths).
const pctOf = (v, maxhp) => (maxhp ? Math.round((v / maxhp) * 10000) / 100 : 0);

function responseFromRange(min, max, immune, maxhp, desc, extra) {
  const expected = (min + max) / 2;
  const block = { min, max, expected, min_pct: pctOf(min, maxhp), max_pct: pctOf(max, maxhp), expected_pct: pctOf(expected, maxhp) };
  // Fixed/state/OHKO damage: a critical hit does not change the value (no base power).
  return Object.assign({ crit: { min, max, expected } }, block, { immune, maxhp, desc: desc || (immune ? 'immune' : '') }, extra || {});
}

function handleDamage(req) {
  const gen = req.gen ?? 9;
  const attacker = buildPokemon(gen, req.attacker);
  const defender = buildPokemon(gen, req.defender);
  const moveName = (req.move && req.move.name) || req.move;
  if (!moveName) throw new Error('move name required');
  const field = buildField(req.field);
  const maxHP = defender.maxHP();
  const attMaxHP = attacker.maxHP();
  const attItem = (req.attacker && req.attacker.item) || null;
  const moveId = String(moveName).toLowerCase().replace(/[^a-z0-9]/g, '');
  const lastHit = (req.move && typeof req.move === 'object') ? req.move.lastHitDamage : null;
  // T6: attacker-side self-damage/healing blocks (recoil, drain, Life Orb).
  // defRemHP = the defender's REMAINING HP (belief-state `hp` if given, else
  // full HP): the fork clamps dealt damage to it before computing recoil/drain.
  const defRemHP = defender.curHP ? defender.curHP() : maxHP;
  const selfFx = (dmgMin, dmgMax, critMax) =>
    selfEffectBlocks(gen, moveId, moveName, attMaxHP, dmgMin, dmgMax, critMax, defRemHP, attItem);

  // Special-damage moves: @smogon/calc has no formula for these, so we complete the
  // calc with the official (fork-verified) formula. `lastHitDamage` (belief-state
  // state) feeds the reflectors. See SPECIAL_DAMAGE_MOVES / specialMoveDamage.
  // These moves CANNOT crit in the fork (the formula dispatch returns before the
  // crit roll), so there is no crit outcome to span and no critChance field.
  if (SPECIAL_DAMAGE_MOVES.has(moveId)) {
    const r = specialMoveDamage(gen, moveId, attacker, defender, lastHit);
    if (r !== null) {
      const extra = Object.assign(r.ohko ? { ohko: true } : {}, selfFx(r.min, r.max, r.max));
      const desc = r.ohko ? 'OHKO — guaranteed KO if it hits (accuracy in move data)'
        : (r.immune ? 'immune' : '');
      return responseFromRange(r.min, r.max, !!r.immune, maxHP, desc, extra);
    }
  }

  // T5 state-BP moves (bpCallback): the library has no formula for these, so we
  // compute the effective base power from the belief-state flags (fork formulas)
  // and hand it to the library, which applies STAB/typing/boosts/immunity/crits
  // normally. See effectiveBP / stateMoveDamage / special_bp_moves.json (D39).
  const st = (req.move && typeof req.move === 'object') ? req.move : {};
  if (STATE_BP_MOVES.has(moveId)) {
    const r = stateMoveDamage(gen, moveId, attacker, defender, st, field);
    if (r !== null) {
      const cexp = ((r.crit && (r.crit.min + r.crit.max)) / 2) || 0;
      const out = Object.assign({
        min: r.min, max: r.max, expected: (r.min + r.max) / 2,
        min_pct: pctOf(r.min, maxHP), max_pct: pctOf(r.max, maxHP), expected_pct: pctOf((r.min + r.max) / 2, maxHP),
        crit: { min: r.crit.min, max: r.crit.max, expected: cexp },
        immune: !!r.immune, maxhp: maxHP,
        desc: r.immune ? 'immune' : '',
      }, selfFx(r.min, r.max, r.crit && r.crit.max));
      // T5 moves are base-power moves in the fork -> they roll the base crit
      // chance like any other (avalanche crits, etc.).
      const cc = baseCritChance(gen, moveName);
      if (cc !== null && !r.immune) out.critChance = cc;
      return out;
    }
  }

  function rangeFor(crit) {
    const move = new Move(gen, moveName, crit ? { isCrit: true } : {});
    const r = calculate(gen, attacker, defender, move, field);
    const [mn, mx] = r.range();
    return { min: mn, max: mx, expected: (mn + mx) / 2, desc: r.desc() };
  }

  let normal, crit;
  try { normal = rangeFor(false); } catch (e) { normal = { min: 0, max: 0, expected: 0, desc: '' }; }
  try { crit = rangeFor(true); } catch (e) { crit = { min: 0, max: 0, expected: 0 }; }

  const immune = normal.max === 0;
  const out = Object.assign({
    min: normal.min,
    max: normal.max,
    expected: normal.expected,
    min_pct: pctOf(normal.min, maxHP),
    max_pct: pctOf(normal.max, maxHP),
    expected_pct: pctOf(normal.expected, maxHP),
    crit,
    immune,
    maxhp: maxHP,
    desc: normal.desc || (immune ? 'immune' : ''),
  }, selfFx(normal.min, normal.max, crit.max));
  // Base-power moves roll the fork's base crit chance (1/24 in gens 7-9 and
  // champions, 1/16 in gens <= 6; higher-crit moves like Slash = 1/8). The AI
  // needs it to weight the crit outcome in expected-value math; battle-state
  // modifiers (Super Luck, Scope Lens, Focus Energy, Laser Focus, Merciless,
  // Lucky Chant, Battle/Shell Armor) are applied by the belief state on top.
  const cc = baseCritChance(gen, moveName);
  if (cc !== null && !immune) out.critChance = cc;
  return out;
}

function main() {
  let raw = '';
  process.stdin.setEncoding('utf8');
  process.stdin.on('data', (c) => (raw += c));
  process.stdin.on('end', () => {
    let req;
    try {
      req = JSON.parse(raw || '{}');
    } catch (e) {
      process.stdout.write(JSON.stringify({ error: 'invalid JSON request: ' + e.message }));
      process.exit(1);
      return;
    }
    // ONE shared per-request dispatcher (no duplicated calc logic): damage or
    // speed. SINGLE mode = one object in -> one object out (unchanged contract;
    // a bad request -> {error} + exit 1, which the test suite relies on). BATCH
    // mode = a JSON ARRAY of requests -> an array of results IN ORDER, with
    // per-item isolation (one bad item -> {error} for that item; the rest still
    // return). Batch is the forward-eval path (my moves x enemy mons in ONE
    // spawn, instead of N spawns).
    function dispatchOne(r) {
      if (r.type === 'damage') return handleDamage(r);
      if (r.type === 'speed') {
        const gen = r.gen ?? 9;
        return { base: buildPokemon(gen, r.pokemon).stats.spe, speed: effectiveSpeed(gen, r.pokemon, !!r.tailwind, r.field) };
      }
      throw new Error('unknown request type: ' + (r && r.type));
    }

    let res;
    if (Array.isArray(req)) {
      res = req.map((r) => {
        try { return dispatchOne(r); }
        catch (e) { return { error: e.message }; }
      });
    } else {
      try {
        res = dispatchOne(req);
      } catch (e) {
        process.stdout.write(JSON.stringify({ error: e.message }));
        process.exit(1);
        return;
      }
    }
    process.stdout.write(JSON.stringify(res));
  });
}

main();
