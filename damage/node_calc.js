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
const { calculate, Pokemon, Move, Field, Side } = require('@smogon/calc');

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

function handleDamage(req) {
  const gen = req.gen ?? 9;
  const attacker = buildPokemon(gen, req.attacker);
  const defender = buildPokemon(gen, req.defender);
  const moveName = (req.move && req.move.name) || req.move;
  if (!moveName) throw new Error('move name required');
  const field = buildField(req.field);
  const maxHP = defender.maxHP();

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
