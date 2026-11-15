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
//   { type:"speed", gen, pokemon:{...}, tailwind? }
//     -> { base, speed }
//
// attacker/defender/pokemon option keys follow @smogon/calc: level, nature, evs,
// ivs, item, ability, status, boosts, teraType, moves. field keys: weather
// (capitalised: "Rain"/"Sun"/"Sand"/"Hail"), terrain, attackerSide, defenderSide.

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

// Modern-gen (3+) stat-stage table: +1..+6 => x1.5..x4 ; -1..-6 => /1.5../4.
// Mirrors @smogon/calc `getModifiedStat`.
function applyBoost(stat, boost) {
  if (!boost) return stat;
  const table = [1, 1.5, 2, 2.5, 3, 3.5, 4];
  const b = Math.max(-6, Math.min(6, boost));
  if (b >= 0) return Math.floor(stat * table[b]);
  return Math.floor(stat / table[-b]);
}

// Effective action speed, mirroring @smogon/calc `getFinalSpeed` for the common
// modifiers (boosts, Choice Scarf, Tailwind, paralysis). Gen 7+ paralysis = x0.5
// (matching both this library and this project's sim fork; see PLAN.md D30).
function effectiveSpeed(gen, o, tailwind) {
  const p = buildPokemon(gen, o);
  let s = p.stats.spe; // base stat (nature + EV), NOT including boosts
  s = applyBoost(s, (o.boosts || {}).spe || 0);
  if (tailwind) s = Math.floor(s * 2);
  if (normItem(o.item) === 'Choice Scarf') s = Math.floor(s * 1.5);
  if (o.status === 'par' && normAbility(o.ability) !== 'Quick Feet') s = Math.floor(s * 0.5);
  return Math.max(0, Math.min(10000, s));
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
function specialMoveImmune(gen, moveId, defender) {
  const entry = (SPECIAL_MOVE_IMMUNITY[String(gen)] || {})[moveId];
  if (!entry) return false;
  const defTypes = (defender.species && defender.species.types) || [];
  if (entry.ground) {
    // Fork: Ground moves bypass the type chart entirely — only the grounded
    // check applies (isGrounded), and its FIRST rule is Iron Ball -> grounded
    // (which therefore overrides Flying type AND Levitate).
    const item = String(defender.item || '').toLowerCase();
    if (item === 'iron ball' || item === 'ironball') return false;
    if (defTypes.includes('Flying')) return true;
    const ability = String(defender.ability || '').toLowerCase();
    if (ability === 'levitate' || ability === 'eelevate') return true;
    if (item === 'air balloon' || item === 'airballoon') return true;
    return false;
  }
  return !!(entry.immune && defTypes.some(t => entry.immune.includes(t)));
}

function specialMoveDamage(gen, moveId, attacker, defender, lastHit) {
  // -> { min, max, immune, ohko } or null (not a special move we handle).
  const defHP = defender.curHP ? defender.curHP() : defender.maxHP();
  const defMaxHP = defender.maxHP ? defender.maxHP() : (defHP || 1);
  const attHP = attacker.curHP ? attacker.curHP() : attacker.maxHP();
  const defTypes = (defender.species && defender.species.types) || [];
  const hasIce = defTypes.includes('Ice');
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

function responseFromRange(min, max, immune, maxhp, desc, extra) {
  const expected = (min + max) / 2;
  const pct = (v) => (maxhp ? Math.round((v / maxhp) * 10000) / 100 : 0);
  const block = { min, max, expected, min_pct: pct(min), max_pct: pct(max), expected_pct: pct(expected) };
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

  // Special-damage moves: @smogon/calc has no formula for these, so we complete the
  // calc with the official (fork-verified) formula. `lastHitDamage` (belief-state
  // state) feeds the reflectors. See SPECIAL_DAMAGE_MOVES / specialMoveDamage.
  const moveId = String(moveName).toLowerCase().replace(/[^a-z0-9]/g, '');
  const lastHit = (req.move && typeof req.move === 'object') ? req.move.lastHitDamage : null;
  if (SPECIAL_DAMAGE_MOVES.has(moveId)) {
    const r = specialMoveDamage(gen, moveId, attacker, defender, lastHit);
    if (r !== null) {
      const extra = r.ohko ? { ohko: true } : {};
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
      const pct = (v) => (maxHP ? Math.round((v / maxHP) * 10000) / 100 : 0);
      const cexp = ((r.crit && (r.crit.min + r.crit.max)) / 2) || 0;
      return {
        min: r.min, max: r.max, expected: (r.min + r.max) / 2,
        min_pct: pct(r.min), max_pct: pct(r.max), expected_pct: pct((r.min + r.max) / 2),
        crit: { min: r.crit.min, max: r.crit.max, expected: cexp },
        immune: !!r.immune, maxhp: maxHP,
        desc: r.immune ? 'immune' : '',
      };
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
  const pct = (v) => (maxHP ? Math.round((v / maxHP) * 10000) / 100 : 0);
  return {
    min: normal.min,
    max: normal.max,
    expected: normal.expected,
    min_pct: pct(normal.min),
    max_pct: pct(normal.max),
    expected_pct: pct(normal.expected),
    crit,
    immune,
    maxhp: maxHP,
    desc: normal.desc || (immune ? 'immune' : ''),
  };
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
    let res;
    try {
      if (req.type === 'damage') res = handleDamage(req);
      else if (req.type === 'speed') {
        const gen = req.gen ?? 9;
        res = { base: buildPokemon(gen, req.pokemon).stats.spe, speed: effectiveSpeed(gen, req.pokemon, !!req.tailwind) };
      } else {
        throw new Error('unknown request type: ' + req.type);
      }
    } catch (e) {
      process.stdout.write(JSON.stringify({ error: e.message }));
      process.exit(1);
      return;
    }
    process.stdout.write(JSON.stringify(res));
  });
}

main();
