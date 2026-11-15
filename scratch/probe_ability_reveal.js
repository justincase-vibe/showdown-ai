'use strict';
// Probe: does the fork reveal a mon's ability in the log at switch-in, and if
// so under what conditions? We run 3 tiny seeded BSS battles and print every
// |-ability| line (and the weather line) so we can see WHEN an ability becomes
// observable:
//   A) passive ability (Charizard/Blaze)      -> expect NO reveal at switch-in
//   B) weather ability (Starmie/Drizzle)       -> reveal via rain onStart?
//   C) ability-shed (Mold Breaker via Garchomp)-> reveal when it takes effect?
const path = require('path');
const Sim = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));
const FMT = Sim.Dex.formats.get('gen9bss3v3', true);
const L = 50;
const mon = (sp, moves, item, ability) => {
  const t = { species: sp, level: L, nature: 'Jolly', evs: { atk: 252, spa: 252, spe: 252 }, moves };
  if (item) t.item = item;
  if (ability) t.ability = ability;
  return t;
};
function run(label, p1, p2, moves) {
  const b = new Sim.Battle({
    format: FMT, debug: true, strictChoices: false, seed: 'gen5,1111222233334444',
    p1: { team: p1 }, p2: { team: p2 },
  });
  try {
    if (b.requestState === 'teampreview') b.makeChoices('team 1, 2, 3', 'team 1, 2, 3');
    for (const [a, c] of moves) {
      if (b.ended) break;
      b.makeChoices(a, c);
    }
  } catch (e) { /* ignore choice errors; we only want the log */ }
  const lines = b.log.filter((l) => typeof l === 'string');
  console.log(`\n===== ${label} =====`);
  for (const l of lines) {
    if (l.includes('-ability') || l.startsWith('|weather') || l.includes('Drizzle') || l.includes('Mold Breaker')) {
      console.log(l);
    }
  }
  b.destroy();
}

// A: passive ability — Charizard (Blaze) vs Garchomp (Rough Skin/Sand Veil)
run('A passive (Blaze / Sand Veil) — expect NO -ability at switch-in',
  [mon('charizard', ['flamethrower', 'slash'], null, 'blaze'),
   mon('garchomp', ['earthquake', 'roost'], null, 'sandveil')],
  [mon('gengar', ['shadowball', 'sludgebomb']),
   mon('garchomp', ['earthquake', 'roost'], null, 'roughskin')],
  [['move flamethrower 1', 'move shadowball 1']]);

// B: weather ability — Starmie (Drizzle) should start rain and reveal Drizzle
run('B weather (Drizzle) — expect -ability reveal via rain',
  [mon('starmie', ['scald', 'rapidspin'], null, 'drizzle')],
  [mon('charizard', ['flamethrower', 'slash'], null, 'blaze')],
  [['move scald 1', 'move flamethrower 1']]);

// C: Mold Breaker on switch + a Mold-Breaker move (Earthquake vs Levitate)
run('C Mold Breaker (Charizard/Levitate target) — reveal on effect?',
  [mon('garchomp', ['earthquake', 'roost'], null, 'roughskin')],
  [mon('charizard', ['fly', 'flamethrower'], null, 'blaze')],
  [['move earthquake', 'move flamethrower 1']]);
