'use strict';
// ─────────────────────────────────────────────────────────────────────────────
// D48 evidence — the residual-vs-move log protocol the belief adapter relies on.
//
// belief/log_events.py classifies a `-damage`/`-heal` line as a RESIDUAL event
// (not a pure move hit) iff its 3rd payload is a BRACKETED context tag
// (belief/log_events.py:  _CONTEXT_TAG = ^\[(from|silent|of|zeffect)\]).
// This probe runs the FORK'S OWN logging code paths (battle.damage → spreadDamage,
// battle.heal → heal) against a live gen-9 BSS battle and asserts that invariant:
//   * a pure MOVE HIT  logs exactly TWO payloads:  |-damage|slot|hp/max
//   * every RESIDUAL line (burn/poison/toxic/recoil/confusion/powder/drain/
//     weather/…) carries a BRACKETED 3rd payload (^[from|silent|of|zeffect]).
//
// Fork source for the protocol (static ground truth, cited for the record):
//   * spreadDamage log switch:  sim/battle.ts:2159-2178  (default → `if
//     effect.effectType === 'Move' || !name` logs NO 3rd payload; else bracketed)
//   * heal log switch:          sim/battle.ts:2293-2314  (leechseed/rest → [silent],
//     drain → [from] drain, zpower → [zeffect], move → no bracket, else [from] name)
//   * recoil/drain fallback:    sim/dex-conditions.ts:700-707  (name 'Recoil'/'Drain')
//   * fullname = data.fullname || name:  sim/dex-data.ts:125  → `!name` needs an
//     effect whose name AND fullname are empty; every real damage source has a name.
//
// Run:  node scratch/probe_residual.js
// ─────────────────────────────────────────────────────────────────────────────
const path = require('path');
const ROOT = path.join(__dirname, '..');
const Sim = require(path.join(ROOT, 'pokemon-showdown/dist/sim'));

const FMT = Sim.Dex.formats.get('gen9bss3v3', true);
if (!FMT) throw new Error('format gen9bss3v3 not found');

const team = (sp) => ({ species: sp, level: 50, nature: 'Docile',
  evs: { hp: 252, atk: 0, def: 252, spa: 0, spd: 0, spe: 0 },
  ability: 'limiter', moves: ['tackle'] });

const battle = new Sim.Battle({
  format: FMT, debug: true, strictChoices: false, seed: 'gen5,1234123412341234',
  p1: { team: [ team('garchomp') ] },
  p2: { team: [ team('blastoise') ] },
});
if (battle.requestState === 'teampreview') battle.makeChoices('team 1', 'team 1');

const src = battle.p1.active[0];
const tgt = battle.p2.active[0];

const ctxRe = /^\[(from|silent|of|zeffect)\]/;
const cases = [];
let failures = 0;

function check(label, kind, effect, expectResidual) {
  // kind: 'damage' | 'heal'
  tgt.hp = kind === 'heal' ? tgt.maxhp - 40 : tgt.maxhp;
  const before = battle.log.length;
  if (kind === 'damage') battle.damage(20, tgt, src, effect);
  else battle.heal(40, tgt, src, effect);
  const lines = battle.log.slice(before)
    .filter(l => typeof l === 'string' && /^\|-(damage|heal)\|/.test(l));

  const eff = (typeof effect === 'string')
    ? { id: effect, note: '(resolved by fork)' }
    : { id: effect.id, name: effect.name, effectType: effect.effectType };
  let ok = true;
  for (const l of lines) {
    const parts = l.split('|');          // ['',' -damage', slot, hp/max, ctx?, ctx4?]
    const ctx = parts.length >= 5 ? parts[4] : null;
    const residual = !!ctx && ctxRe.test(ctx);
    if (residual !== expectResidual) { ok = false; failures++; }
    const tag = residual ? 'RESIDUAL' : 'MOVE-HIT';
    const mark = (residual === expectResidual) ? '  ✓' : '  ✗ MISMATCH';
    console.log(`  ${mark} ${tag.padEnd(9)} ctx=${ctx ?? '(none)'}  ${l}`);
  }
  if (lines.length === 0) { ok = false; failures++; console.log(`  ✗ no -damage/-heal line produced`); }
  cases.push({ label, ...eff, expectResidual, ok, n: lines.length });
  const head = `${label}  [id=${eff.id}${eff.name ? ', name=' + eff.name : ''}${eff.effectType ? ', type=' + eff.effectType : ''}]`;
  console.log(`# ${head}  =>  expected ${expectResidual ? 'RESIDUAL (bracketed 3rd payload)' : 'MOVE-HIT (2 payloads)'}`);
}

try {
  console.log('D48 residual-vs-move log protocol — live fork probe (gen9bss3v3)\n');
  check('CONTROL move hit (Tackle)',   'damage', battle.dex.moves.get('tackle'), false);
  check('burn tick  (brn)',           'damage', battle.dex.conditions.get('brn'), true);
  check('poison tick (psn)',          'damage', battle.dex.conditions.get('psn'), true);
  check('toxic tick  (tox→psn)',      'damage', battle.dex.conditions.get('tox'), true);
  check('recoil     (recoil)',        'damage', 'recoil', true);
  check('confusion  (confused)',      'damage', battle.dex.conditions.get('confused'), true);
  check('powder     (powder)',        'damage', battle.dex.conditions.get('powder'), true);
  check('weather dmg (Sandstorm)',    'damage', battle.dex.conditions.get('sandstorm'), true);
  check('drain heal (drain)',         'heal',   'drain', true);
  check('leechseed heal (leechseed)', 'heal',   battle.dex.conditions.get('leechseed'), true);
} finally {
  console.log('\n────────────────────────────────────────────────────────────');
  for (const c of cases) console.log(`  ${c.ok ? 'PASS' : 'FAIL'}  ${c.label}`);
  console.log('────────────────────────────────────────────────────────────');
  console.log(failures === 0
    ? `ALL ${cases.length} CASES PASS — move hits are 2-payload; every residual line is bracketed.`
    : `${failures} FAILURE(S) — a residual line was NOT bracketed, or a move hit WAS.`);
  battle.destroy();
  process.exit(failures === 0 ? 0 : 1);
}
