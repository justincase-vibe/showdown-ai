# Showdown AI — Plan & Roadmap (10-step build)

> **Single source of truth** for *what we're building, the decisions we've made, and where
> we are*. `README.md` is the intro/how-to-run; the verbatim 10-step spec from the user
> lives in [build_plan_10step.txt](build_plan_10step.txt); module-level detail for the
> legality DB lives in [legality/PLAN.md](legality/PLAN.md).
> Update the status table in §3 the moment a step lands — never lose the thread.

**Goal:** the greatest Pokemon AI ever — a bot that plays **mixed strategy** (sometimes
switches, sets hazards/screens/boosts, anticipates danger) instead of always clicking its
strongest move. The old agent (~10k battles of PPO) only used its strongest moves and
never switched; the environment wrapper, battle loop and PPO training loop from the old
project exist and are reused where the plan says to.

**Project setup (2026-11-07, D25–D28):** the old project `showdown AI` (one folder up)
is kept **read-only as a reference**. All work happens in this folder,
`showdown AI-imitation`. Its design (Gen 6–9 × OU/UU/RU, 892-dim flat observation, PPO
from scratch, draft head) is **superseded** by the 10-step plan below.

---

## 1. Rules for the coder (user-mandated)

1. **One step at a time.** Do not start step N+1 until step N's tests pass.
2. **Every step has runnable tests** (pytest), written alongside the code.
3. **Small, independent modules** — each testable alone.
4. **No hidden-info leakage:** never put into the observation anything the player could
   not see at that moment (opponent's unrevealed moves/items/abilities). Use **belief
   sampling** (step 3) instead.
5. **Action masking** for illegal actions — never penalties.
6. **Log everything useful:** policy entropy, switch rate, setup-move rate, win rate per
   opponent type.

## 2. Target format (DECIDED, D27; scope widened 2026-11-08)

- **Primary target: Generation 9, `[Gen 9] BSS 3v3`** (id `gen9bss3v3`) —
  fixed-team 3v3 singles under Battle Spot Singles flat rules (bring 6, pick 3, level 100).
  Defined in this project's `pokemon-showdown/config/formats.ts` — **custom to this
  server build, not a stock Smogon format** (that's why the plan says "confirm the exact
  format ID": done, it's `gen9bss3v3`).
- **Format-agnostic requirement (user, 2026-11-08):** BSS is the *first/primary* target,
  not the only one — the damage/speed calc (and the AI stack built on it) must work for
  **all formats in the legality DB** (50 formats, gens 1–9 incl. the 2 Champions formats,
  see D32). Callers key everything by **format**, never by a hard-coded gen.
- Data coverage: `gen9ou`, `gen9uu`, `gen9ru` (Smogon set sources) + `gen9bss3v3`.
- BSS rules verified in server data: `pickedTeamSize=3`, `maxLevel=100`, `maxTeam` unset
  (assert `in (None, 6)`), see `test_bss_format_rules`.
- **Team preview** (picking which 3 of 6) is a separate hidden-information problem →
  start with a fixed pick / fixed teams; learned selection later.

## 3. Ten-step roadmap (STATUS — update per step)

| # | Step | Deliverable | Gate (tests) | Status |
|---|------|-------------|--------------|--------|
| 1 | Move/species database | lookup layer over Showdown data + Smogon common sets (Gen 9) | known lookups correct; learnset integrity (no missing keys); set probabilities sum to 1 | ✅ **2026-11-07** — `legality/tests/test_step1.py` **21/21 green** |
| 2 | Damage calculator wrapper | `calc_damage(att, def, move, field, fmt) → (min%, max%, expected%)` via @smogon/calc (Node) | 15–20 hand-checked matchups within rounding; screens/boosts/immunities; speed helper | ✅ **2026-11-08** — `damage/tests/test_step2.py` **34/34 green** (17 hand-checked matchups + 3 immunities + STAB/eff/burn/boost/weather/screen/item/ability/crit/speed); **+ format-key addendum (D32):** `fmt=` param on all 3 APIs, `legality.formats.engine_key` (`champions`→0, `genN`→N), `damage/tests/test_format_key.py` **7/7 green** — full gate **62/62** |
| 3 | Opponent belief sampler | belief over hidden info; `update(event)`, `sample(n)`, `move_probabilities(mon)` | revealed move in every sample; illegal sets = 0; probs sum to 1 after each update; Choice-lock shift | ⬜ |
| 4 | Observation encoder | structured per-mon + damage block + field + legal mask (replaces flat 892-dim) | fixed shapes; no leakage (unrevealed move absent); deterministic; bench swap = slot permutation only | ⬜ |
| 5 | Network | set encoder + GRU core + masked policy/value heads (hidden 128–256) | illegal actions = exactly 0 prob; permutation invariance; overfit-32 sanity; GRU reset per battle | ⬜ |
| 6 | Behavior cloning | replay parser → BC from human replays (cross-entropy, **no reward**) | switch rate within a few pts of human (~15–25%); per-action-type accuracy; beats random, competitive with max-damage | ⬜ |
| 7 | PPO fine-tune + league | reuse PPO loop; win/loss + annealed shaping; KL→BC annealed; entropy; league bots | switch/setup rates stay non-zero (alarm <3%); >60% vs max-damage and rising; ≥400 battles per comparison | ⬜ |
| 8 | Payoff matrix builder | depth-1 matrix: own legal actions × belief-weighted opponent actions → value head | OHKO case → switch row wins; harmless-slow case → setup row wins; dims match; no NaNs | ⬜ |
| 9 | LP / Nash solver | `linprog` zero-sum solve → mixed strategy over own actions | RPS = [⅓,⅓,⅓]; matching pennies = [½,½]; dominant row → 1; valid simplex | ⬜ |
| 10 | Integration & eval | player class: obs → net → (optional 8–9) → sample | A vs B ≥400 battles each, 95% CIs, switch/setup rates; B not worse than A (else debug value head/matrix) | ⬜ |

**Step spec text (Build + TESTS per step):** see [build_plan_10step.txt](build_plan_10step.txt)
— the user's plan, kept verbatim.

**Gate rule:** a step starts only when the previous step's tests are green and logged
here (✅ + date + evidence). Failed gate = fix + re-run, never "proceed with caveats."

## 4. Decision log

Continues the old project's log (D1–D19 in the old `showdown AI/legality/PLAN.md`, still
valid as the legality module's foundation). D20–D23 were assigned during the Step 1
session (already cited in code comments, e.g. `schema.sql`); D24/D29 land with the
Step 1 wrap-up.

| # | Decision | Rationale / note | Date |
|---|----------|------------------|------|
| D20 | **Schema v3:** `strategy.ability_id` added | Smogon sets always list an ability; the AI (set→action mapping, belief) needs it | 2026-11-07 |
| D21 | **Type chart:** encoding `{0:1x, 1:2x, 2:0.5x, 3:0x-immune}`; matrix stored `[defending][attacking]` | verified against `sim/dex.ts getEffectiveness/getImmunity` + `dist/data/typechart.js`; NOT log2 | 2026-11-07 |
| D22 | **Type chart = 18 real types only** (324 rows; `stellar` pseudo-row excluded by design) | pseudo-types are not playable types | 2026-11-07 |
| D23 | **Set ingester skips sets whose species is not legal in the format** (validator verdict) | unplayable sets would pollute step 3's belief space; gen9: 489 raw → 477 ingested (2,033 raw across all 12 Smogon formats) | 2026-11-07 |
| D24 | `get_type_chart()` returns **display-name keys** ("Fire") to match `move.type` / `form_type.type` in the DB — damage math can index directly | table stores lowercase ids; display name = id capitalized (true for all 18 types) | 2026-11-07 |
| D25 | **New project folder** `showdown AI-imitation` (full copy); old `showdown AI` kept **read-only reference** | user setup; compare new vs old; old project's artifacts (env, PPO loop, server fork, team pools) remain reusable | 2026-11-07 |
| D26 | **10-step build plan adopted** as the roadmap; one step at a time; every step pytest-gated | user plan (build_plan_10step.txt) supersedes the old Gen 6–9 plan | 2026-11-07 |
| D27 | **Gen 9 + `[Gen 9] BSS 3v3`** (`gen9bss3v3`) is the AI target format; data also covers `gen9ou/uu/ru` | BSS format added to `pokemon-showdown/config/formats.ts` (this project's server copy only) + `npm run build`; rules verified in tests | 2026-11-07 |
| D28 | **Keep JSON/TS data pipeline** — Node `Dex` extracts Showdown data → JSON payload → SQLite; no format migration | user: "JSON is fine, we don't need to change the file types" | 2026-11-07 |
| D29 | **This fork's data = ground truth, always.** Type chart verified to match the official chart (20 cells checked against `dist/data/typechart.js`). Species typing DOES deviate: `ninetalesalola` = **Ice/Fairy** (official: Fire/Fairy). Earlier note claiming "chart deviates (Fire→Fairy 1x, Rock→Bug 1x, …)" was an **axis mix-up** — Rock→Bug is 2× (super-effective); the 1× cell was Bug→Rock | tests assert fork-verified values; never re-derive from memory of the official chart | 2026-11-07 |
| D30 | **Step 2 engine = @smogon/calc v0.12.0** (the official Smogon calc library) wrapped as a Node CLI (`damage/node_calc.js`) reading JSON from stdin; Python `damage/calc.py` spawns it and exposes `calc_damage(...)` + `compare_speed(...)`. Damage, critical hits (separate value), and immunity (→0, the library *throws* on 0-damage so we catch it) all come from the library — no formula re-implementation. Installed at project root (`package.json`); `node_modules/` stays gitignored | plan says "prefer wrapping"; the library is literally the Smogon calculator, so "match the official calc within rounding" is satisfied by construction and cross-checked against an independent formula | 2026-11-08 |
| D31 | **Speed helper & natures follow @smogon/calc / this fork (ground truth), not memory.** Natures: `Jolly=+Spe/−SpA`, `Adamant=+Atk/−SpA`, `Bold=+Def/−Atk`, `Modest=+SpA/−Atk`, `Naive=+Spe/−SpD`, `Serious=neutral` (fork `data/natures.ts` == @smogon/calc `NATURES`). **Gen 7+ paralysis = ×0.5** (not the classic ×0.25) — matches BOTH @smogon/calc `getFinalSpeed` and this fork's `par` handler, so the bot's speed reasoning stays consistent with the sim it fights on. Crit bypasses Reflect in the library model. Speed order: base → boosts → Tailwind(×2) → Scarf(×1.5) → par(×0.5) | my own nature/paralysis memory was wrong twice (Jolly, par ×0.25); always source these from the library/fork, never from memory | 2026-11-08 |
| D32 | **Format-awareness investigated (within-gen mods DO exist).** `[Gen 9 Champions] *` formats (`mod: champions`, fork `data/mods/champions/`) change ~25 move base powers vs vanilla gen 9 (Slash 70→**80**, Anchor Shot 85→90, Beak Blast 100→120, Bloodmoon 120→130, First Impression 90→100, Bonerush 25→30, …), plus types (Growth→Grass, Snaptrap→Steel), accuracy (Crabhammer→95), PP (Beak Blast→5), 201 moves banned, and status mechanics (par = ×0.5 **and** 1/8 move-skip; sleep/freeze durations). **@smogon/calc v0.12.0 implements champions natively as engine key `gen=0`** (`MECHANICS[0]` in `dist/calc.js`); verified library champions Slash BP=80 == fork `mods/champions/moves.ts` (gen-9 Slash=70). Our DB already contains 2 champions formats (`gen9championsou`, `gen9championsvgc2026regmc`; `gen=9`, `mod='champions'`). **Therefore the engine key for a format = its `mod` (`genN`→N, `champions`→0), NOT its `gen` column** — a naive `format.gen` lookup would silently mis-calc the champions rows. User: calc must serve **all** formats (BSS primary). `Move(gen, name, {overrides:{…}})` deep-merges into the formula if finer per-format tweaks are ever needed. **Implemented (user-approved 2026-11-08):** `legality.formats.engine_key(fmt)` (registry `mod` → key; single source of truth) + `fmt=` param on `calc_damage`/`effective_speed`/`compare_speed` (fmt wins over `gen`; unknown format → clear error). Tests: BSS ≡ gen-9 (242–288 unchanged), champions EQ = **120–144** (level-independent at L50 and L100 — the champions engine is a *different damage model*, not just BP tweaks; the fork's own `mods/champions/scripts.ts` implements its `modifyDamage`, so the library model tracks the sim), champions Slash ≠ gen-9 Slash. **62/62 green.** Follow-up (optional, before champions formats are used in training): cross-check a champions matchup against a live fork battle | investigation requested by user ("champions is gen 9 technically? different base powers?"); verified against fork ground truth + library dist, not memory | 2026-11-08 |

## 5. Verified facts about this server build (evidence — don't re-derive)

- **Type chart source of truth:** `pokemon-showdown/dist/data/typechart.js` →
  `TypeChart[defender_id].damageTaken[Attacker_name]`, encoding
  `0=1x, 1=2x, 2=0.5x, 3=immune`. **Beware the axis:** `Rock→Bug` lives in
  `bug.damageTaken.Rock` (=1 → 2×), not `rock.damageTaken.Bug` (=0 → 1×, that's Bug→Rock).
  Verified chart cells (attacker→defender): Water→Fire 2, Fire→Grass 2, Ice→Dragon 2,
  Poison→Fairy 2, Steel→Fairy 2, **Rock→Bug 2**, Electric→Ground 0, Dragon→Fairy 0,
  Ghost→Normal 0, Psychic→Dark 0, Poison→Steel 0, Ground→Flying 0, Fighting→Ghost 0,
  Normal→Ghost 0, Grass→Grass 0.5, Ground→Bug 0.5, Fairy→Fire 0.5, Fire→Fairy 1, Bug→Bug 1,
  Bug→Rock 1 — all match the DB (`type_chart`, 324 rows, gen 9).
- **Species typing deviates from official:** `ninetalesalola` = Ice/Fairy (checked in
  `dist/data/pokedex.js`). Treat the fork's dex as ground truth for species too.
- **Dex API quirks (hard-won):** `dex.species.all()/moves.all()/…` return **effect
  objects** — use `.id`; `dex.data.Types` is undefined — enumerate the 18 real types
  explicitly; `dex.types.get('fire')` works but `damageTaken` inner keys are **display
  names** while top-level keys are **ids**; Showdown's `gen` field = gen the entity was
  **introduced** (not removed).
- **BSS format:** `pickedTeamSize=3`, `maxLevel=100`, `maxTeam` unset in this fork
  (assert `in (None, 6)` — do not assume 6).
- **Ingest pitfall (fixed):** payload rows containing only PK columns produced an empty
  `DO UPDATE SET` ("incomplete input") — `api.ingest` now falls back to `DO NOTHING`.
- Legacy `Hidden Power <type>` (16 variants) all carry server id `hiddenpower` →
  extractor skips name/id mismatches; the set ingester maps variant names → `hiddenpower`.

## 6. Step 1 deliverables (done — evidence)

- **DB:** `legality/legality.db`, schema v3, `PRAGMA integrity_check` ok.
  Counts: pokemon 1120 · form 1517 · move 938 · ability 321 · item 587 ·
  form_learnset 65872 · type_chart 324 · format 50 (49 registry + `gen9bss3v3`) ·
  format_form_move 201350 · **strategy 477** + strategy_move 1893.
- **API:** `legality/lookup.py` — `get_move` (power/type/category/acc/priority/
  boosts + `is_hazard/is_screen/is_tailwind` flags), `get_species` (types/stats/
  abilities/tier), `legal_moves`, `get_common_sets` (probability = 1/n per species),
  `get_type_chart` (display-name keys).
- **Tests:** `legality/tests/test_step1.py` — **21/21 green (2026-11-07)**, covering the
  plan's Step 1 checklist (hazard/screen/boost flags, Hidden Power collapse, species
  lookups, full 18×18 chart + known matchups, learnset integrity, set legality,
  probabilities sum to 1, BSS format rules).
- **Rebuild recipe** (only if the extractor or data changes — no rebuild needed now):
  ```
  node legality/extract_gen9.js --gen 9 --formats gen9ou,gen9uu,gen9ru,gen9bss3v3 --out legality/data/extract_gen9.json
  python -m legality.build bootstrap --force
  python -c "from legality import api; api.ingest_file('legality/data/extract_gen9.json', source='showdown')"
  python -m legality.ingest_movesets
  python -m pytest legality/tests -v
  ```

## 6a. Step 2 deliverables (done — evidence)

- **Engine:** `damage/node_calc.js` (Node CLI, JSON in/out) over **@smogon/calc v0.12.0**;
  `damage/calc.py` is the Python API (`calc_damage`, `effective_speed`, `compare_speed`);
  `damage/__init__.py` mirrors the `legality` package layout.
- **Public API:**
  - `calc.calc_damage(attacker, defender, move, field=None, gen=9)` →
    `{min,max,expected, min_pct,max_pct,expected_pct, crit:{min,max,expected}, immune, maxhp, desc}`.
  - `calc.compare_speed(a, b, gen=9, a_tailwind, b_tailwind)` → `{a, b, faster}`.
- **Verified behavior (34/34 tests, `damage/tests/test_step2.py`):** 17 hand-checked
  matchups (each recomputed from the standard formula with the library's own base stats,
  then asserted) + 3 immunities (→0); STAB, super-effective/neutral/resisted, burn (÷2),
  offence +2/−2 and defence +2 boosts, Rain ×1.5 (Water) / Sun ×1.5 (Fire), Reflect ÷2
  (physical) / Light Screen ÷2 (special), Life Orb ×1.3, Choice Band ×1.5 Atk,
  Adaptability (×2 STAB), critical hits (separate value; ignore a −2 Atk drop; bypass
  Reflect), and the speed helper (base/boosts/Scarf/par/Tailwind orderings, incl. the
  par- and boost-flips). Step 1 stays green (21/21) — **55/55 combined**.
- **Gate run:**
  ```
  cd "C:\\Users\\JustinCase\\Documents\\Projects\\showdown AI-imitation"
  python -m pytest damage/tests -v     # Step 2: 34 passed
  python -m pytest legality/tests damage/tests -q   # 55 passed
  ```
- **Note:** @smogon/calc **throws** (`damage===0`) instead of returning 0 for complete
  immunities — `node_calc.js` catches that and reports `{immune:true, min:max=0}`.

## 7. Superseded (old project, read-only reference)

The old plan — Gen 6–9 × {OU,UU,RU}, 892-dim flat observation, unified 963-action space,
PPO-from-scratch, drafting value head — lives in the read-only `showdown AI` folder
(`PLAN.md`, `README.md` there). Useful carry-overs into steps 6–10: the poke-env
wrapper + battle loop, the masked-PPO training loop, measured throughput
(~10–12 battles/min at 16 concurrent; 4 worker processes), and server config
(throttling ON, `simulator: 16`). The new plan replaces the *design*, not the *harness*.
