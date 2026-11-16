'use strict';
// Step 3.7 fuzz keeper — run N seeded fork battles in the BSS format and print,
// per battle, a team-spec JSON header (FUZZ_HEADER:) + the raw log. The pytest
// gate (belief/tests/test_step3_7.py) feeds each log through belief/log_events.py
// + BeliefState.update() every event and asserts the belief invariants hold,
// counts Unknown events (the observable bug-catcher), and writes the per-turn
// report artifact.
//
// Why diverse teams + full-length battles: the point is to exercise the belief
// UPDATE machinery over REAL logs — real switch-ins (new priors), KOs + re-switches
// (slot re-init), statuses, boosts, Choice/Tera — not to force every mechanic
// (those are unit-tested in 3.4/3.6). Auto-choices on both sides over a full battle
// guarantee KOs + switches; different pairings/species give different priors and
// evidence. The teams are the SINGLE source of truth: the header hands the exact
// own-team (p1) specs to the gate so it builds BeliefState.own_team without
// duplicating the team definition.

const path = require('path');
const Sim = require(path.join(__dirname, '..', 'pokemon-showdown', 'dist', 'sim'));

const FMT = Sim.Dex.formats.get('gen9bss3v3', true);
if (!FMT) throw new Error('format gen9bss3v3 not found');
const LEVEL = 50;
const BASE_EVS = { atk: 84, spa: 84, def: 84, spd: 120, spe: 84 };

function team(species, moves, opts = {}) {
  const { nature = 'Jolly', evs = BASE_EVS, item = '', ability = '' } = opts;
  const t = { species, level: LEVEL, nature, evs: { ...evs }, moves: [...moves] };
  if (item) t.item = item;
  if (ability) t.ability = ability;
  return t;
}
// a belief-state own_team spec (the fields mon_options() reads): name/nature/evs/item/ability
function specOf(t) {
  return { name: t.species, level: t.level, nature: t.nature, evs: t.evs,
           item: t.item || '', ability: t.ability || '', moves: t.moves };
}

// ── p1 (own side) team archetypes — species/abilities are all fork-legal ──
const P1 = {
  A: [
    team('garchomp', ['earthquake', 'swordsdance', 'roost', 'dragonclaw'], { ability: 'roughskin', item: 'lifeorb' }),
    team('blastoise', ['scald', 'protect', 'surf'], { ability: 'torrent' }),
    team('tyranitar', ['stoneedge', 'crunch', 'stealthrock'], { ability: 'sandstream' }),
  ],
  B: [
    team('heatran', ['lavaplunge', 'flamecharge', 'lavaplug'], { ability: 'flashfire', item: 'choiceband' }),
    team('dragapult', ['dracometeor', 'dragonascent', 'uturn'], { ability: 'clearbody' }),
    team('chandelure', ['shadowball', 'energyball', 'healpulse'], { ability: 'wiseeyes' }),
  ],
  C: [
    team('goodra', ['sludgewave', 'icebeam', 'dragonpulse'], { ability: 'sapsipper' }),
    team('ferrothorn', ['spikes', 'leechseed', 'knockoff'], { ability: 'ironbarbs' }),
    team('zapdos', ['thunderbolt', 'heatwave', 'roost'], { ability: 'pressure' }),
  ],
};
// ── p2 (opponent) team archetypes ──
const P2 = {
  X: [
    team('charizard', ['flamethrower', 'solarbeam', 'dragonclaw'], { ability: 'flamebody' }),
    team('gengar', ['shadowball', 'sludgebomb', 'darkpulse'], { ability: 'levitate' }),
    team('machamp', ['dynamicpunch', 'stoneedge', 'drainpunch'], { ability: 'guts' }),
  ],
  Y: [
    team('corviknight', ['bravebird', 'roost', 'bodypress'], { ability: 'pressure' }),
    team('tapulele', ['psychic', 'moonblast', 'roost'], { ability: 'electric' }),
    team('togekiss', ['dazzlinggleam', 'wish', 'protect'], { ability: 'serenegrace' }),
  ],
  Z: [
    team('garchomp', ['earthquake', 'dragonclaw', 'swordsdance'], { ability: 'roughskin' }),
    team('blastoise', ['scald', 'surf', 'protect'], { ability: 'torrent' }),
    team('togekiss', ['dazzlinggleam', 'wish', 'protect'], { ability: 'serenegrace' }),
  ],
};

// 12 battles: 6 pairings x 2 seeds (diverse teams AND flows).
const PAIRINGS = [['A', 'X'], ['A', 'Y'], ['A', 'Z'], ['B', 'X'], ['B', 'Y'], ['B', 'Z']];
const SEEDS = ['gen5,1010101010101010', 'gen5,9876543219876543'];

function runBattle(seed, p1, p2) {
  const opts = { format: FMT, debug: true, strictChoices: false, seed,
                 p1: { team: p1 }, p2: { team: p2 } };
  const battle = new Sim.Battle(opts);
  const MAXTURNS = 30;
  try {
    if (battle.requestState === 'teampreview') {
      battle.makeChoices('team 1, 2, 3', 'team 1, 2, 3');
    }
    for (let i = 0; i < MAXTURNS && !battle.ended; i++) {
      battle.makeChoices(); // auto-choose on both sides
    }
  } catch (e) {
    // A malformed choice / edge line is a signal the fuzz should surface, not a crash.
    console.error(`  [fuzz] battle stopped early: ${e.message}`);
  } finally {
    const lines = battle.log.filter((l) => typeof l === 'string');
    battle.destroy();
    return lines;
  }
}

const battles = [];
for (const [a, b] of PAIRINGS) {
  for (const seed of SEEDS) battles.push({ seed, p1: P1[a], p2: P2[b] });
}

battles.forEach((b, i) => {
  console.log(`=== FUZZ BATTLE ${i} START ===`);
  console.log('FUZZ_HEADER: ' + JSON.stringify({ seed: b.seed, p1: b.p1.map(specOf), p2: b.p2.map(specOf) }));
  for (const l of runBattle(b.seed, b.p1, b.p2)) console.log(l);
  console.log(`=== FUZZ BATTLE ${i} END ===`);
});
console.log(`=== FUZZ DONE: ${battles.length} battles ===`);
