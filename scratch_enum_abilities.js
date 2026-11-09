'use strict';
// ABILITY NAME-CONTRACT AUDIT — ground truth = the official Smogon damage calc
// (@smogon/calc, the engine behind smogon.com's calculator).
//
// The set of "abilities that influence damage" is DEFINED by the official calc:
// exactly the abilities its damage code branches on (hasAbility(...)). Our
// wrapper passes ability IDS (from the AI / legality DB / sim fork), and the
// calc resolves them by DISPLAY NAME. So for every ability the official calc
// models, the fork's display name for that id MUST equal the name the calc
// expects — otherwise the effect is silently dropped (the Guts-style bug).
//
// This script cross-references:
//   * the official calc's expected names (parsed from hasAbility(...) in dist), and
//   * the fork's id -> display-name table (our normalisation source of truth).
// and flags every mismatch / missing mapping — i.e. every ability where
// normAbility(forkId) would NOT produce the name the calc looks for.
//
// Run: node scratch_enum_abilities.js

const path = require('path');
const fs = require('fs');

// ── 1. official calc: the abilities its damage code branches on ─────────
// Parse hasAbility(...) string args from the compiled mechanics. Apostrophe- and
// quote-aware (handles "Dragon's Maw", 'Mind's Eye', 'Dragon\'s Maw').
const MECH = path.join(__dirname, 'node_modules', '@smogon', 'calc', 'dist', 'mechanics');
const calcNames = new Set();
for (const f of fs.readdirSync(MECH)) {
  if (!f.endsWith('.js')) continue;
  const txt = fs.readFileSync(path.join(MECH, f), 'utf8');
  const re = /hasAbility\(([^)]*)\)/g;
  let m;
  while ((m = re.exec(txt))) {
    // pull every single- or double-quoted string in the argument list
    const strs = m[1].match(/"([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'/g) || [];
    for (const s of strs) {
      const inner = s.slice(1, -1).replace(/\\(.)/g, '$1').replace(/\\"/g, '"').replace(/\\'/g, "'");
      if (inner) calcNames.add(inner);
    }
  }
}
console.log(`Official calc models ${calcNames.size} abilities (hasAbility set).`);

// ── 2. fork ground truth: id -> display name ────────────────────────────
const _ab = require(path.join(__dirname, 'pokemon-showdown', 'dist', 'data', 'abilities'));
const Abilities = _ab.Abilities || _ab;
const nameToIds = {}; // lowercased name -> [ids]
const idToName = {};
for (const id of Object.keys(Abilities)) {
  const a = Abilities[id];
  if (!a || !a.name) continue;
  idToName[id] = a.name;
  (nameToIds[a.name.toLowerCase()] ||= []).push(id);
}

const norm = (s) => s.toLowerCase();
const ciFind = (name) => nameToIds[norm(name)] || null;

// ── 3. cross-reference ──────────────────────────────────────────────────
const ok = [], broken = [];
for (const name of [...calcNames].sort()) {
  const ids = ciFind(name);
  if (ids && ids.length) ok.push({ name, ids });
  else broken.push(name);
}

console.log(`\n=== CONTRACT OK: ${ok.length}/${calcNames.size} official abilities map cleanly to a fork id ===`);
for (const o of ok) console.log(`  ${o.name}  -> ${o.ids.join(', ')}`);

console.log(`\n=== BROKEN / MISSING: ${broken.length} official ability name(s) have NO fork id with that exact display name ===`);
for (const name of broken.sort()) {
  // find the closest fork name to diagnose (first 6 chars)
  const frag = norm(name).replace(/[^a-z]/g, '').slice(0, 5);
  const close = Object.values(idToName).filter((n) => norm(n).replace(/[^a-z]/g, '').startsWith(frag)).slice(0, 4);
  console.log(`  ${name}${close.length ? `   [fork close-matches: ${close.join(' | ')}]` : '   [no close fork match]'}`);
}
