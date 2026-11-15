'use strict';
// Step 3.1 smoke keeper — run ONE live seeded fork battle in the BSS format
// and print the raw log, line by line. The pytest gate (belief/tests/
// test_step3_1.py) feeds those lines through belief/log_events.py and asserts
// the event stream is sane. Battle is designed to exercise: switches in of
// two mons per side (species + ability reveals), several move uses, damage
// (incl. a faint/KO), a status (Scald burn attempt), and a boost (Swords
// Dance). Seed is fixed → reproducible.

const path = require('path');
const Sim = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));

const FMT = Sim.Dex.formats.get('gen9bss3v3', true);
if (!FMT) throw new Error('format gen9bss3v3 not found');

const LEVEL = 50;
const team = (sp, moves, item) => {
  const t = { species: sp, level: LEVEL, nature: 'Jolly', evs: { atk: 252, spa: 252, spe: 252 }, moves };
  if (item) t.item = item;
  return t;
};

const opts = {
  format: FMT,
  debug: true,
  strictChoices: false,
  seed: 'gen5,1234123412341234',
  p1: { team: [
    team('garchomp', ['earthquake', 'swordsdance', 'roost']),
    team('blastoise', ['scald', 'protect']),
    team('tyranitar', ['stoneedge', 'crunch']),
  ] },
  p2: { team: [
    // mismagius LEADS so turn 1's Earthquake KOs it (ghost 2x weak, frail):
    // the smoke log then contains a real faint + a REAL second p2 switch-in
    // (charizard). (With charizard leading the battle never switches and the
    // smoke test's ">= 2 p2 switch-ins" only ever passed via the dup-line bug.)
    team('mismagius', ['thunderbolt', 'shadowball']),
    team('charizard', ['flamethrower', 'slash', 'wingattack']),
    team('gengar', ['shadowball', 'sludgebomb']),
  ] },
};

const battle = new Sim.Battle(opts);
try {
  if (battle.requestState === 'teampreview') battle.makeChoices('team 1, 2, 3', 'team 1, 2, 3');
  // NOTE: hit-all/self moves take NO target spec ('move earthquake', not ' 1').
  // Turn 1 scripted (both choices known-legal); the rest auto — the point is
  // a real multi-turn battle log, not a specific line (3.7 fuzz drives
  // scripted scenarios properly).
  battle.makeChoices('move earthquake', 'move thunderbolt 1');  // p2 lead is mismagius
  for (let i = 0; i < 3; i++) {
    if (battle.ended) break;
    battle.makeChoices();  // auto-choose on both sides
  }
} finally {
  console.log('=== BATTLE LOG START ===');
  for (const line of battle.log) {
    if (typeof line === 'string') console.log(line);
  }
  console.log('=== BATTLE LOG END ===');
  battle.destroy();
}
