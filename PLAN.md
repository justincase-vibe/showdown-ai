# Showdown AI — Plan & Roadmap (10-step build)

> **Single source of truth** for *what we're building, the decisions we've made, and where
> we are*. `README.md` is the intro/how-to-run; the verbatim 10-step spec from the user
> lives in [build_plan_10step.txt](build_plan_10step.txt); module-level detail for the
> legality DB lives in [legality/PLAN.md](legality/PLAN.md).
> Update the status table in §3 the moment a step lands — never lose the thread.

> **Sources of truth — what's authoritative for what (keep these distinct, D37):**
> - **Damage model (damage numbers, crit, immunity, %HP): `@smogon/calc` — the official
>   Smogon damage calculator — IS the source of truth.** This is the canonical model the
>   AI reasons about and what we ship. When we need a damage number, the calc decides; we
>   make it *complete and correct* per the official Smogon rules (Bulbapedia/Smogon wiki =
>   the spec) and never re-derive damage from memory.
> - **The fork (`pokemon-showdown` sim) is the battle ENVIRONMENT** — where the AI plays
>   and where damage is *observed* — and our **validation oracle**: we run it to confirm
>   the calc matches in-game reality (both implement the official rules, so they must
>   agree; the calc is the reference). The fork is **not** the source of a damage formula.
> - **Raw game DATA (species typing/stats, move/ability/item tables, type chart): this
>   fork's `Dex` is the source of truth** (this server build's data; e.g. `ninetalesalola`
>   = Ice/Fairy). This is a different domain from damage formulas (see D29).
>
> In short: **damage → `@smogon/calc` · game data → the fork's `Dex` · the battle itself
> → the fork.** "Fork = ground truth" applies only to raw game data and to *validating*
> the calc — never to the damage model itself.

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
  fixed-team 3v3 singles under Battle Spot Singles flat rules (bring 6, pick 3,
  **level 50** — in this fork Flat Rules = "Adjust Level Down 50", see D33).
  Defined in this project's `pokemon-showdown/config/formats.ts` — **custom to this
  server build, not a stock Smogon format** (that's why the plan says "confirm the exact
  format ID": done, it's `gen9bss3v3`).
- **Format-agnostic requirement (user, 2026-11-08):** BSS is the *first/primary* target,
  not the only one — the damage/speed calc (and the AI stack built on it) must work for
  **all formats in the legality DB** (50 formats, gens 1–9 incl. the 2 Champions formats,
  see D32). Callers key everything by **format**, never by a hard-coded gen.
- Data coverage: `gen9ou`, `gen9uu`, `gen9ru` (Smogon set sources) + `gen9bss3v3`.
- BSS rules verified in server data: `pickedTeamSize=3`, `maxLevel=100` (the CAP),
  `adjustLevelDown=50` → **standard teams battle at level 50**, `maxTeam` unset (assert
  `in (None, 6)`), see `test_bss_format_rules` + D33.
- **Team preview** (picking which 3 of 6) is a separate hidden-information problem →
  start with a fixed pick / fixed teams; learned selection later.

## 3. Ten-step roadmap (STATUS — update per step)

| # | Step | Deliverable | Gate (tests) | Status |
|---|------|-------------|--------------|--------|
| 1 | Move/species database | lookup layer over Showdown data + Smogon common sets (Gen 9) + format level rules | known lookups correct; learnset integrity (no missing keys); set probabilities sum to 1; level rules match live fork | ✅ **2026-11-07** — `legality/tests/test_step1.py` **21/21 green**; **+2026-11-08 (D33 level fix):** schema v4, all 50 formats carry fork-verified level rules, `battle_level()` helper, live-fork match guard — suite **24/24 green** |
| 2 | Damage calculator wrapper | `calc_damage(att, def, move, field, fmt) → (min%, max%, expected%)` via @smogon/calc (Node) | 15–20 hand-checked matchups within rounding; screens/boosts/immunities; speed helper | ✅ **2026-11-08** — `damage/tests/test_step2.py` **34/34 green** (17 hand-checked matchups + 3 immunities + STAB/eff/burn/boost/weather/screen/item/ability/crit/speed); **+ format-key addendum (D32+D33):** `fmt=` param on all 3 APIs — engine key from `mod` (`champions`→0, `genN`→N) and each mon's level resolved by the format's level rules (BSS/VGC/champions L50, LC L5, ladder L100; illegal level → error), `damage/tests/test_format_key.py` **10/10 green** — full gate **68/68** | **+ cross-check vs live fork sim (D34, 2026-11-09):** 11 mechanics × 24 seeds all 11/11; **item/ability id-vs-name contract bug fixed** (`normItem`/`normAbility`) + 3 regression tests — full gate **71/71**; **+ 10 more matchups + Guts/veil/burn deep-dive (D35, 2026-11-09):** +10 hand-checked matchups (incl. dual-type Surging Strikes, Life Orb, +2 boost, Adaptability, priority Aqua Jet — 2 sim-confirmed) and +10 Guts/Aurora-Veil/burn-interaction tests — full gate **91/91**; **+ damage-ability audit + current-HP pass-through (D36, 2026-11-09):** 192/192 ability name-contract clean (no silent drops); the ONE real wrapper gap fixed — current HP was never passed, so **every** HP-gated effect was wrong (low-HP attacker effects never fired, full-HP-only defender effects always did); each mon now accepts `hp` (0.0-1.0 fraction) / `curHP` (absolute) — Defeatist, Overgrow/Blaze/Torrent/Swarm, Eruption/Water Spout/Flail/Reversal, Multiscale/Shadow Shield/Tera Shell, Brine; +21 engine-verified tests — full gate **112/112** | **+ special-damage (non-basePower) moves (D37, 2026-11-10):** `@smogon/calc` returned 0 for 11 formula-based moves (Super Fang, Ruination, Nature's Madness, Guardian of Alola, Final Gambit, Endeavor, Psywave, and the 4 reflectors Counter/Mirror Coat/Metal Burst/Comeuppance) — `node_calc.js` now completes them with the official fork-verified formulas; **Super Fang's gen-aware Ghost immunity** (immune vs Ghost in gen 2+, hits Ghost in gen 1); reflectors take a new `move.lastHitDamage` (belief-state prior hit; absent → fails 0, matching the fork's thisTurn gate). `damage/tests/test_special_damage_moves.py` +17 fork-pinned tests — full gate **134/134** | **+ T2–T4 special-damage moves + fork-exact type-immunity gate (D38, 2026-11-10):** T2 level moves (Nightshade/Seismic Toss = attacker level), T3 OHKO (Fissure/Guillotine/Horn Drill always-KO; Sheer Cold = KO or Ice-immune), T4 (Explosion/Self-Destruct/Misty Explosion via the library's base-power path; faint-support moves report 0); **type-immunity gate rebuilt** — `@smogon/calc` ignores immunities on formula moves, so `node_calc.js` now gates them via a fork-exact precomputed table `damage/special_move_immunity.json` (regen: `scratch/gen_special_move_immunity.js`) + fork-dispatch rules (Iron Ball overrides Flying/Levitate/Eelevate/Air Balloon for Ground moves; Sheer Cold Ice-immunity); a prior from-memory move-type map was found **wrong** (fork data was right — Natures Madness=Fairy, Final Gambit=Fighting, Seismic Toss=Fighting, Ruination=Dark, …). +12 tests — full gate **146/146** | **+ T5 state-dependent base-power moves + champions hardening (D39, 2026-11-15):** 9 BSS-legal `bpCallback` moves (avalanche/beatup/pledges/lastrespects/spitup/ST/TF) completed via a fork-exact per-engine base-BP table + belief-state flags → `Move(overrides:{basePower})` into the one library pipeline (fixes the `beatup`/`spitup` **0-damage bug**; ST/TF = prev-move-fail, lastrespects = absolute 50+50×fainted — both formula corrections); champions engine hardened: banned pledges report 0/immune (no more crash) and non-whitelist species (chansey etc.) now raise a clean legality error instead of an opaque library crash; +17 tests — full gate **163/163** |
| 3 | Opponent belief sampler | belief over hidden info; `update(event)`, `sample(n)`, `move_probabilities(mon)` | revealed move in every sample; illegal sets = 0; probs sum to 1 after each update; Choice-lock shift | ✅ **3.1–3.8 ✅ 2026-11-16** (belief 84/84; D43, D46, D47, D48) — 3.7 whole-battle fuzz bug-catcher **GREEN** (12 seeded BSS battles; the 🔴 was a real Step-1 "Past"-tag data bug, fixed per D47) · 3.8 perf bench (**update ~0.5–0.75s** Node-spawn-bounded, **sample(100) ~0.5ms** — acceptable for per-turn; a batch endpoint is the flagged deep-tree follow-up) · **D48 post-milestone bug-hunt:** residual-vs-move classification (burn ticks were collapsing the posterior) + `last_hit` reflector input, all fork-verified + regression-gated |
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

**Step 3 execution plan (locked 2026-11-15; Bayesian core per user approval):**
- **Architecture:** per-slot posterior over **full-set candidates** (correlations preserved
  by construction — e.g. a Wish set already contains Protect) with weights; every event is a
  **likelihood function** over candidates. **HARD evidence** (revealed move / ability /
  Choice-lock / Tera) = exactly 0/1. **SOFT evidence** (damage window, speed order) = (0,1]
  with an **ε floor** → the posterior can **never empty out** (off-DB sets survive). One
  **shared evidence/window module** is used by both the set posterior and the HP/EV inference
  (nothing written twice). Module: `belief/` (small, independent, Python; no DB, plain
  dataclasses).
- **3.1** belief state + event model + prior (Step-1 `get_common_sets`; bounded legal-move
  fallback for set-less species) + **mini log→event adapter**, smoke-tested against one live
  seeded fork battle → *gate: init tests (sums to 1, fallback, visibility dims) + event smoke*
  → ✅ **2026-11-16** — `belief/tests/test_step3_1.py` **29/29** (incl. the moveset-is-an-
  unordered-set invariant, user 2026-11-16); 3 adapter bugs fixed (move indices, switch-in dup
  dedupe, `movefail` doesn't exist in this fork — see §5 log-protocol facts).
- **3.2** evidence engine (per-evidence likelihoods; HARD/SOFT split; ε floor; never-empty
  guard) → *gate: per-evidence-function unit tests*
  → ✅ **2026-11-16** — `belief/evidence.py` (ONE shared module for set posterior 3.3–3.5 AND
  HP/EV inference 3.6 — nothing written twice). HARD likelihoods exact 0/1; **"unknown is not
  disconfirmed"** (a candidate with `item==""`/`tera is None` is consistent with a reveal of
  that dimension, not ruled out); **EPS floor (1e-9)** in `apply_evidence` guarantees the
  posterior **never empties** (off-DB set → all floor at EPS → uniform, still normalizable).
  `belief/tests/test_step3_2.py` **13/13**.
- **3.3** `update()`: revealed move / ability / **Choice-lock (same move twice ⇒ Choice item)** /
  Tera → *gate: the 4 spec tests (revealed in every sample; illegal = 0; sums to 1; Choice shift)*
  → ✅ **2026-11-16** — `SlotBelief.update(event)` + `BeliefState.update(event)` (routes by
  `event.slot`; own-side p1* and field/damage lines ignored — those are 3.4/3.6). Choice-lock
  fires once on the 2nd sighting of the SAME move on that mon. `add_opponent` re-initializes a
  slot's prior when a NEW mon takes it (3v3 faint+re-switch) and keeps it for the same mon.
  `belief/tests/test_step3_3.py` **9/9** (the 4 spec tests + routing/re-init integration).
- **3.4** damage + speed-order evidence via the Step 2 calc (**KO-clip rule D40** baked in)
  → *gate: in/out-of-window, KO-clip, speed-discrimination tests*
  → ✅ **2026-11-16** — SOFT evidence wired through `BeliefState.update` (**belief-slot-only**,
  own side exact — no N×M coupling) over the ONE shared `evidence.py` window module. Damage window
  = HARD window over the Step-2 calc's `[min,max]` **plus** its crit range, in BOTH directions
  (candidate-as-victim and candidate-as-attacker); **KO-clip (D40)** = a KO line is consistent iff
  the candidate *could* KO (`max>=maxhp or crit_max>=maxhp`); an always-KO config (`min>maxhp`)
  is inconsistent with any surviving hit; an immune candidate is ruled out by a real hit. Speed
  order = equal-priority moves, first mover is faster (a tie is consistent; different priority =
  no speed evidence; a switch/faint between the two moves breaks the order). Per-slot **fact sheet**
  (`Facts`: hp/maxhp, statuses, boosts, moves_used, tera, last_hit) + own-side facts + shared
  field (weather/terrain/Tailwind with turn-count expiry). **Bug found & fixed (3.4-only, untested
  until now):** `mon_options` required `spec["name"]` but belief candidates carry no `name`
  (the species is `SlotBelief.species`), so every candidate window/speed raised `KeyError`; the
  candidate callers now pass `name=species` (own-team specs still use their own `name`).
  `belief/tests/test_step3_4.py` **13/13**; belief suite **64/64**; **full gate 286/286**;
  keepers 40/40 + 29/29+20/20 + 14/14.
- **3.5** `sample(n)` + `move_probabilities()` + correlation preservation
  → ✅ **2026-11-16** — `SlotBelief.sample(n, seed)` = SEEDABLE multinomial draw of FULL sets
  (fresh `random.Random(seed)`, stateless across calls/slots, returns COPIES so a sample can't
  corrupt the belief); `move_probabilities()` = posterior MASS over the sets CONTAINING each
  move (membership, order-blind per the 3.3 moveset-is-a-set invariant; values in [0,1],
  sum ≈ posterior-expected move count, not 1; PP unobservable → ignored). Correlation is
  preserved BY CONSTRUCTION (full-set sampling) — the spec test pins P(Protect|Wish)=0.9.
  `belief/tests/test_step3_5.py` **10/10**; belief suite **74/74**; **full gate 296/296**.
- **3.6** max-HP / EV-bucket inference on the shared window module (Seismic Toss case;
  burned-but-hits-hard ⇒ Guts) → *gate: window, EV-narrowing, Guts tests*
  → ✅ **2026-11-16** — `belief/tests/test_step3_6.py` **5/5**: the 3.4 window machinery now
  PROVES the three things the user cares about, locked as regressions — **window** (Seismic
  Toss = flat attacker level = a degenerate [50,50] band @ BSS level 50, non-critting),
  **EV-narrowing** (a defender's SpD EVs / an attacker's Atk EVs shift the calc window → an
  observed hit discriminates the spread), **Guts** (burned Guts ×1.5 Atk vs burned non-Guts
  ×0.5 Atk — the 3.4 window already discriminates; no new code). **+ the one genuinely-new
  piece (D46): a max-HP-consistency EV evidence** — the switch-in line reveals max HP
  (verified 3.1); `maxhp_for()` reads the Step-2 calc's `maxhp` field (the defender's maxHP at
  IV 31 = the spread's maxHP *maximum* — probe-verified immune-safe + one-sided ordering) and
  `maxhp_likelihood` is ONE-SIDED / IV-safe (observed>candidate ⇒ ruled out; observed≤candidate ⇒
  consistent — never a false rule-out), applied on the p2 switch-in (idempotent on re-entry).
  **OPEN DECISION (user asleep 2026-11-16 — flagged for review, NOT yet user-approved):** the
  §3.6 plan said "residual-damage likelihoods (burn 1/16, poison) land here"; I reframed that
  into this max-HP-consistency evidence (a burn/poison hit is just another observation of the
  SAME max HP — the switch-in line is the direct one). See §3a + **D46** for the alternative
  (literal burn/poison line) — a local swap if the user prefers it. belief suite **79/79**,
  full gate **301/301**.
- **3.7** **whole-battle fuzz bug-catcher:** N seeded fork battles, real log → events →
  `update` every turn, invariants asserted + **human-readable per-turn belief report**
  (top candidates, top move per slot, entropy) → *gate: fuzz green + report artifact*
- **3.8** perf bench (update + 100 samples per mon, ms-scale) + D43 entry + commit
  → *gate: full pytest green*

### 3a. Open decision options — 3.4–3.8 (2026-11-16, AWAITING USER REVIEW)

> Written per user request (user asleep 2026-11-16): the **options** for each of the
> next sub-steps plus my recommendation in **bold**. **3.4 will implement the bold
> recommendation unless the user objects** (it is reversible — evidence is a pure
> likelihood, so a different shape is a local change). 3.5–3.8 wait on the user.

**3.4 — damage window + speed order (SOFT evidence via the Step-2 calc):**

1. **Window likelihood shape:** **HARD window** (observed delta ∈ `[min,max]` OR ∈
   `[crit.min,crit.max]` → 1, else 0→EPS floor) vs a soft falloff (e.g. Gaussian
   around the window). Rec: hard — the Step-2 calc is verified against the sim
   within rounding, so a hard window is decisive AND exactly testable; the crit
   window is included as also-consistent (sim crit base rate 1/24 in gens 7–9).
2. **Cross-slot coupling:** **update the belief (p2) slot ONLY** — the other mon is
   exactly known (own team) or, if it is also a belief slot (multi-battle only),
   its top candidate (mode) is used as the neutral stand-in. Never joint N×M
   marginalization over both slots' candidates (bounded cost, no N×M calc blowup).
3. **KO-clip (D40):** a KO line is consistent iff `max ≥ maxhp` (or `crit_max ≥
   maxhp`); a candidate with `min > maxhp` (always-KO) is ALSO consistent with a KO
   line but inconsistent with any surviving (non-KO) hit.
4. **Speed-order signal:** **the order of `|move|` announce lines within a turn =
   action order**; if the two moves have different priorities → no evidence; if equal
   → the first mover's effective speed ≥ the second's (a TIE is consistent — the sim
   resolves ties randomly, so either order is possible). Tailwind / boosts /
   paralysis / weather enter via the calc's `effective_speed`; Quick Claw ignored
   (rare, and this fork's log carries no observable Quick Claw signal).
5. **Fact sheet — the "make unknown gamestates known" registry:**
   **`SlotBelief.facts = {hp, maxhp, statuses, boosts, moves_used, tera, last_hit}`**
   **+ `BeliefState.field = {weather, terrain, tailwind_side/turns}`**; `_used_moves`
   migrates into `facts.moves_used` (single observation registry; 3.3 tests stay
   green). `last_hit` (the damage this mon last took) is the reflector input for 3.6.
6. **Residual damage (burn / poison / recoil / hazards): DEFERRED to 3.6** (HP/EV
   inference) — 3.4 consumes move-source damage + speed order only; an unattributable
   or non-move damage line still updates the HP fact, just produces no set evidence.
7. **Perf:** **dedupe identical calc calls** — the window depends on the mon SPEC
   (stats/item/ability/tera/status/boosts/hp) + move + field, not on the set, so
   candidates sharing stats cost one Node spawn each. Per-call Node spawn stays for
   3.4; a batch endpoint in `node_calc.js` only if the 3.8 bench is too slow.

**3.5 — sampling:** **seeded multinomial `sample(n)`**; `move_probabilities()` =
posterior mass over the sets containing the move (membership, order-blind —
3.3 invariant); PP ignored (unobservable in the log).

**3.6 — EV/nature/ability refinement:** max HP is **ALREADY revealed by the
switch-in line in this fork** (the `hp/maxhp` denominator — verified 3.1), so 3.6 =
EV/nature/ability refinement **on the same shared window module** (the Seismic
Toss flat-damage case stays as a window regression; "burned-but-hits-hard ⇒ Guts"
lands here). Residual-damage likelihoods (burn 1/16, poison, recoil) also land
here per 3.4 decision #6.

  **OPEN (implemented one option; user to choose, 2026-11-16):** the "residual
  damage (burn/poison)" signal is, mechanically, just another observation of the
  SAME max HP (a burn = max(1,⌊maxhp/16⌋), a poison = max(1,⌊maxhp/8⌋) — both
  fork-verified). Two ways to consume it:
  * **A (implemented, RECOMMENDED):** **max-HP-consistency evidence on the
    switch-in line** — the direct, clean observation of max HP. One-sided /
    IV-safe (`observed > candidate` ⇒ ruled out; `observed ≤ candidate` ⇒
    consistent), so it narrows the HP-EV spread with **no false rule-outs**.
  * **B (not implemented; a local swap inside `_apply_maxhp_evidence`):** consume
    the literal **burn/poison damage line** — derive the implied maxHP band from
    the damage via the fork-verified rate, then apply the same one-sided rule.
  Both are the same signal; A is the cleaner source (no rate-inversion rounding).
  **Deferred (documented, not implemented):** Toxic (stage-dependent rate), recoil
  (attacker-side — no opponent-set signal), hazard damage, and any two-sided
  maxHP band (IV-unsafe → false rule-outs).

**3.7 — whole-battle fuzz bug-catcher:** N≈10–20 **seeded fork battles**, real log →
events → `update` every turn, invariants asserted + **human-readable per-turn report**
(top candidates, top move per slot, entropy) saved as an artifact; `Unknown` event
count reported (the observable bug-catcher).

  **✅ RESOLVED 2026-11-16 (Option B, D47) — the 3.7 gate is GREEN.** The root cause below
  is a real Step-1 "Past"-tag data bug (NOT "a stale DB a rebuild fixes"); it was fixed per D47
  and 3.7 re-ran green (12 seeded BSS battles). The A/B/C/D decision was made (B) and shipped in
  the same clean commit. The 6 benign `prepare`/charge-up `Unknown` events are an adapter gap
  (a candidate future `MovePreparing` event), not a 3.7 failure.

  * The harness (`scratch/step3_7_fuzz_battles.js`, 12 seeded BSS battles, real multi-turn
    logs) and the gate (`belief/tests/test_step3_7.py`) both work. The gate fails on battle 0:
    `ValueError: no prior for 'machamp' in 'gen9bss3v3': no sets and no legal moves`.
  * **The prior handoff's diagnosis was WRONG** and has been corrected here. It said the DB
    was "stale" and that a rebuild would fix it. **A rebuild does NOT fix it** (re-running the
    extractor reproduces the exact same gap) — verified: the current fork's `getFullLearnset`
    has no gen-9 codes for these species, and the fork marks them nonstandard.
  * **REAL root cause (2 layers, same ~454 species — incl. machamp, abra, onix, staryu, aerodactyl, aggron…):**
    1. This fork's `data/learnsets.ts` is **missing gen-9 entries** for ~454 species — their
       learnset codes cap at **gen 8** (verified: machamp max-gen = 8; garchomp/charizard = 9).
    2. Consequently the fork's dex marks those species **`isNonstandard: "Past"`, `tier: "Illegal"`**
       (454 / 1517 in `Dex.forGen(9)`). The extractor's `checkSpecies` call therefore rejects
       them ("does not exist in Gen 9") → they are **absent from `format_form`** →
       `legal_moves()` returns `[]` → `build_prior` has no sets (BSS has none for ANY species)
       and no legal moves → raises.
  * **The fork is internally inconsistent here:** its own `TeamValidator.validateTeam`
    *accepts* a machamp/abra set (any moves, even non-learnset ones) in BSS and OU, while
    `checkSpecies` (as the extractor calls it) *rejects* them as "Past." So the "Past" tag is a
    data artifact, not a deliberate ban.
  * **What I already changed (in the working tree, NOT committed, NOT yet sufficient):** the
    extractor's learnset filter was `c[0] === "9"` (gen-9 codes only) — too strict. I widened it
    to the fork's actual rule `3 ≤ learnedGen ≤ dex.gen` (moves carry forward; mirrors
    `checkCanLearn`'s `noFutureGen` + `minSourceGen`). This is a genuine, verified-correct fix
    (0 orphan move ids; Step-1 24/24 still green; machamp learnset 0 → 107). **But it does NOT
    unblock 3.7**, because those species are still rejected by `checkSpecies` due to the "Past"
    tag (layer 2 above). It should ship with the real fix, not alone.

  **DECISION MADE (B) + SHIPPED (D47, 2026-11-16):** the user's 3.7/3.8 green-light
  ("3.8 is next… you can do this, boss") authorized the A/B/C/D call; **B** was implemented as
  the complete, verified Step-1 fix in one clean commit (Step-1 re-verified 24/24, 3.7 re-run
  GREEN). The options, kept for the record:
  * **A — Fix the fork's learnset data:** add the missing gen-9 learnset codes to the ~454
    species so they are normal gen-9 species (un-"Past"). Most "correct", but = editing the
    fork (our game-data source of truth) and I don't yet have a verified-correct gen-9
    learnset for all 454 ready.
  * **B — Extractor overrides the "Past" tag** for species that exist (have `num` + `exists`) —
    i.e. don't let `isNonstandard == "Past"` block a species the fork's own `validateTeam`
    accepts. Pragmatic, keeps the fork as-is, matches BSS "Flat Rules = all species allowed"
    intent, and is what the fork's *team validator* actually does. Localized, reversible,
    testable. Risk: un-bans any species that is genuinely Past-only in gen 9 (in standard gen 9
    there are none — real gen 1–8 species are all legal — so this is likely safe/desirable).
    **My lean (NOT user-approved): B.**
  * **C — Accept the limitation:** restrict teams + belief to the ~1063 non-Past species
    (drops machamp/abra/onix/staryu… — common competitive mons). Simplest, no fork change.
  * **D — Re-examine the fork choice:** a clean stock Smogon build may have correct gen-9 data
    and avoid this whole class of problem.

  **Status (2026-11-16, milestone reached):** belief 3.1–3.8 **81/81 green** · Step-1 (legality)
  **24/24 green** · 3.7 gate **GREEN** · 3.8 bench **GREEN** (update ~0.5–0.75s Node-spawn-
  bounded, sample(100) ~0.5ms) · full suite green. Committed (D43 + D47). **Step 3 complete.**

**3.8 — perf bench + D43 + commit:** ✅ **2026-11-16** — `belief/tests/test_step3_8.py` **1/1**
(measures the per-turn belief cost: switch-in (3.6 maxhp evidence) + a SOFT damage
`update` + `sample(100)`, asserts sane ceilings + the window dedupe cache, writes
`belief/artifacts/bench_3_8_report.txt`). Result: **update ~0.5–0.75s** (one Node spawn
≈ 250ms per unique mon spec; the defender's HP fraction changes each turn → a cache miss each
turn, so the Node spawn IS the realistic per-turn cost) · **sample(100) ~0.5ms** (the D45
`deepcopy` concern is a non-issue). **Verdict: acceptable for per-turn belief updates**
(well inside Showdown's turn budget); the Node spawn is the cost to watch for deep-tree search
(Step 5+) — a batch endpoint in `node_calc.js` is the flagged follow-up, NOT a Step 3 blocker.
**D43** (ability-reveal fact) + **D47** (the Step-1/Past-tag data fix) land in the decision log.

**Post-milestone bug-hunt (D48, 2026-11-16) — OPEN ITEMS for user review:**

> The user's standing instruction before sleep: "compile an open questions review for
> me later and continue on bug fixes". The bug-hunt (D48) is done + regression-tested
> (committed in the D48 commit, incl. the residual-log protocol probe
> `scratch/probe_residual.js` 10/10 + the `!name`-edge static proof); these are the
> decisions it surfaces.

1. **D46 Option B is now implementable (still RECOMMEND A):** D48's residual
   classification (`source_kind="residual"` on burn/poison/recoil/drain lines) is
   exactly the hook Option B needs to consume the literal burn/poison damage LINE as
   another max-HP observation (implied band = `damage ∈ [max(1,⌊maxhp/16⌋)]` for burn,
   `/8` for poison). Option A (max-HP consistency on the switch-in line, implemented)
   is still the cleaner source — no rate-inversion rounding, and the switch-in line is
   the direct observation. **Rec: keep A; B only if the user wants the extra signal
   when the switch-in line is ambiguous (e.g. slot-reuse edge).**
2. **Crit-window conservatism (accepted, recorded for the record):** a non-crit delta
   that happens to fall inside a WRONG candidate's crit window keeps that candidate
   alive (no false rule-outs — by design — but weaker discrimination on that hit). The
   alternative (a `CritObserved` event that tightens the window to the crit range for
   THAT hit only) is a possible Step-4 refinement; not worth it for BSS (crit base
   rate 1/24, gens 7–9).
3. **Verified NOT bugs (D48 hunt — do not re-investigate):** (a) **crits** — the
   likelihood accepts the crit window (`damage_likelihood`, 3.4-1); (b) **OHKO moves**
   — the calc reports the guaranteed KO (`min=max=defMaxHP`, `ohko:true`) and the KO-clip
   (D40) consumes it; (c) **multi-hit moves** — the library's window spans ALL per-hit
   values incl. the 1.5× boosted hit (`multiDamageRange` over the damage matrix; Skill
   Link's max hits via the candidate's own ability), and each hit logs its own
   `|-damage|` line = one piece of evidence each; (d) **type-effectiveness lines**
   (`-resisted`/`-supereffective`/`-immune`) — TE is already inside the window (the
   candidate's own typing); (e) **hazards/weather** — `-side damage` = `source_kind="field"`,
   facts only; (f) **KO-by-residual** (burn KO line) — facts only, no move evidence;
   (g) **`last_hit` reflector input** — FIXED: now move-hit-only, matching the fork
   (its counter volatile / `attackedBy` is set from the move-hit loop alone,
   `gotAttacked` at battle-actions.ts:1051 — residuals are never reflected);
   (h) **1-HP micro-divergence (accepted, documented):** a move that HIT for 0
   damage then a same-turn Counter — the fork's `damageCallback` does
   `volatiles['counter'].damage || 1` (reflects 1), our calc fails it (0).
   Such hits are near-impossible in BSS (real 0-damage outcomes are
   miss/immune/block, which are not hits at all), so we keep the cleaner
   "no prior hit ⇒ reflector fails" semantics the calc already documents.
4. **Still open from 3.6/3.7 (unchanged):** the 6 benign `prepare`/charge-up `Unknown`
   events in the 3.7 fuzz report (candidate future `MovePreparing` event — an adapter
   gap, not a failure); the `node_calc.js` **batch endpoint** for deep-tree search
   (Step 5+; the 3.8 bench showed the per-turn Node spawn is acceptable, deep-tree is
   where it hurts).

**Post-D48 fix wave (P0/P1/P2 + combat context + T5 wiring) — ✅ CLOSED 2026-11-16, full gate 310/310 (full entry D49):**

- **P0 reflector collapse — FIXED** (`b057fd6`, regression `test_reflector_preserves_narrowed_posterior`):
  the calc now receives tracked state via the generic `_move_with_state(move, facts)`
  pipe; **no hardcoded move lists in Python** — the Node calc is the semantic
  authority (D37: @smogon/calc = damage-model truth; Python is the data pipe).
- **P2 + combat context — SHIPPED** (`9917ab0`): full field state (Trick Room /
  Wonder Room / terrains / Reflect / Light Screen / Aurora Veil / Stealth Rock /
  Sticky Web / Safeguard / Spikes / Toxic Spikes) via the generic
  `FieldEffectChanged` event; two-level `Facts` reset (`turn_reset()` on E.Turn vs
  full `reset()` on a new mon; `last_hit` persists — reflector memory); and
  **`BeliefState.get_context()`** = the single belief-state snapshot the AI trains
  on (turn, fainted, active, field, per-mon hp_frac / statuses / boosts / moves /
  last_hit / hit_this_turn / last_move_failed / stockpile, posterior top-K +
  confidence).
- **T5 wiring — SHIPPED** (`fd0dccf`): the two remaining never-set facts now set —
  `hit_this_turn` in `_apply_damage` (move-source, `delta > 0` per the fork's
  avalanche gate `attackedBy.some(p => p.source === target && p.damage > 0 &&
  p.thisTurn)`, moves.ts:912-915) and `last_move_failed` in the `MoveFailed`
  dispatch branch (fork gate `moveLastTurnResult === false`, moves.ts:18050/19186 —
  stompingtantrum/temperflare; both sides). Both cleared on E.Turn (fork resets
  `thisTurn` at battle.ts:1720-1727). +3 regressions (set / pipe / 0-damage /
  turn-reset).
- **OPEN (new, for user review — all rare-in-BSS, none block Step 4):**
  1. ~~**`last_move_failed` unreachable from real logs**~~ — **✅ CLOSED 2026-11-16
     (`85729b9`, D50):** the adapter now maps this fork's real failure lines —
     `|-fail|<user>` (after the `|move|` announce; move name recovered from the
     announce) and `|cant|<user>|<reason>` for the move-attempt aborts
     (par/slp/frz/flinch/recharge/nopp — all the fork's
     `moveThisTurnResult=false` paths; other cant reasons stay silent). Adapter +
     end-to-end regressions; the 3.7 fuzz on real battles stays green.
  2. ~~**`stockpile_layers` never increments**~~ — **✅ CLOSED 2026-11-16
     (`85729b9`, D50):** `MoveUsed(move="stockpile")` now adds a layer (fork cap
     3 — the 4th use fails and adds nothing). **Fork divergence found while
     wiring it:** layers are a fork VOLATILE with no duration — they PERSIST
     across turns (a multi-turn Stockpile→Spit Up sequence would have been
     broken by the old per-turn wipe) and clear only on switch-out / new mon.
  3. ~~**`hit_this_turn` approximation**~~ — **✅ CLOSED 2026-11-16 (D53):** now
     PER-SOURCE — `attacked_by_this_turn: set[(side,slot)]` records which mon(s) hit
     this one; `_move_with_state(move, facts, defender)` computes `defender in
     attacked_by_this_turn` at the two `_damage_evidence` call sites; the calc is
     unchanged (still reads the `hitThisTurn` bool). Covers Avalanche (gen 8-9) AND
     Revenge (gen 4-7, same fork gate). +1 test (`test_hit_this_turn_is_per_source`);
     full gate **320/320** (was 319).
  4. ~~**`spikes` 0/1 bool + `confidence` `-0.0` + 3.7 "benign prepare" Unknowns**~~
     — **✅ CLOSED 2026-11-16 (D51):** Spikes/Toxic Spikes now COUNT the per-layer
     `|-sidestart|` lines (the fork emits one line per layer, NO count in the line —
     `moves.js:17536-17543`; caps 3/2) and reset on `|-sideend|`; `confidence`
     clamped to [0,1]; `-prepare` (client animation line, `SIM-PROTOCOL.md:594`)
     moved to `IGNORED_TAGS`. Full gate **316/316** (was 313).
  5. **`revenge` (gen 4-7) not in the calc's `STATE_BP_MOVES` set** — it has the SAME
     per-source gate as Avalanche (`attackedBy.some(p => p.source === target &&
     p.damage > 0 && p.thisTurn)`, moves.js:15085-15093) but `node_calc.js` only models
     `avalanche`. So gen ≤7 formats would mis-model Revenge (no doubling). NOT a BSS
     issue (gen 9); surfaces only when targeting early gens. **Recommendation:** add it
     to `STATE_BP_MOVES` + `special_bp_moves.json` when we sweep early-gen formats.
  6. ~~**Reflector stale-hit divergence**~~ — **✅ CLOSED 2026-11-16 (D54):** the fork's
     reflectors are PER-TURN — Counter/Mirror Coat recreate their volatile via
     `beforeTurnCallback` with **`duration: 1`** and `onTry` FAILS (`slot === null`)
     unless the user was hit by a damaging move THIS turn (moves.js:3025-3060); Metal
     Burst/Comeuppance require **`thisTurn`** (moves.js:2739/11721). Our `last_hit`
     (magnitude) persists across turns as memory, but `_move_with_state` now gates
     `lastHitDamage` on `attacked_by_this_turn` (this-turn, move-source, damage>0) — so
     a STALE hit (previous turn) no longer reflects (the calc's `lastHit==null → 0`
     path fires), while the same-turn case (the common path) still reflects 2×. +1 test
     (`test_reflector_stale_hit_not_reflected`); full gate **321/321** (was 320).
  7. ~~**Reflector CATEGORY gate (fork-accuracy)**~~ — **✅ CLOSED 2026-11-16 (D55):** the
     fork reflects only a MATCHING category — Counter/Metal Burst record a **Physical**
     hit, Mirror Coat/Comeuppance a **Special** one (moves.js `onDamagingHit`:
     `getCategory(move) === "Physical"` / `"Special"`); a MISMATCH means the reflector
     FAILS (0). Our calc applied `2× lastHit` to Counter/Mirror Coat **regardless of the
     hit's category** (node_calc.js:486-493) and `facts.last_hit` recorded no category.
     **Fix (Python data pipe; calc UNCHANGED — D37 keeps @smogon/calc the authority):**
     `Facts.last_hit_category` is set alongside `last_hit` in `_apply_damage` from the
     hit move's DB category (persisted as reflector memory, cleared in `reset()`);
     `_move_with_state` attaches `lastHitDamage` **only to a reflector** (the 4 in
     `_REFLECTOR_REQUIRES`) **and only when** `facts.last_hit_category` matches that
     reflector's requirement — a mismatch falls to the calc's `lastHit==null → 0` path
     (base power, no over-report), and non-reflector moves never carry `lastHitDamage`.
     **Tests:** `test_reflector_preserves_narrowed_posterior` re-paired to a SPECIAL hit +
     Mirror Coat (fork-accurate); `test_reflector_stale_hit_not_reflected` re-paired to a
     PHYSICAL hit + Counter; +1 `test_reflector_category_gate` (special opens
     Mirror Coat/Comeuppance, NOT Counter/Metal Burst; physical the reverse; tackle never
     carries it). **Gate: 322/322** (was 321).

### 3b. Step 4 stage (observation encoder / belief sampler) — STAGE LAID, NOT BUILT (2026-11-16)

> Per user: "continue setting up the stage for part 4" — this lays the ground
> (observation completeness + the open network decisions + recommendations) so
> Step 4 can start cleanly. The Step-4 net/sampler code is **NOT implemented yet.**
> The calc-endpoint PREREQUISITE is now RESOLVED: the user chose **batch (Option 2)**
> — "let's implement the batch for now, its the safer route" (2026-11-16, D52) — implemented
> + validated (`calc_damage_many`, ~22.5× faster than N individual spawns, identical
> results). Step 4 can now start once the B/C decisions below are confirmed.

**A. Observation surface — what `get_context()` already exposes (the net's input):**
- `turn`, `fainted` {p1,p2}, `active` {p1,p2 slot}
- `field`: weather, terrain, trickroom, wonderroom, tailwind_side, reflect /
  lightscreen / auroraveil (per-side), **spikes (0-3, D51)**, **toxicspikes (0-2, D52)**,
  **stealthrock, stickyweb, safeguard (D52)**
- `own[slot]`: hp_frac, statuses, boosts, moves, tera, last_hit, hit_this_turn,
  last_move_failed, **stockpile (0-3)**, **used_protect (D52)** + name/ability/item (known)
- `opponent[slot]`: species, hp_frac, statuses, boosts, moves, tera, last_hit,
  hit_this_turn, last_move_failed, stockpile, **used_protect (D52)**, **posterior top-K**
  (ability/moves/weight), **top_moves** (marginal), **confidence** (now clamped [0,1], D51)

**B. Observation decisions (dim-sensitive):**
1. ~~**Toxic Spikes (0-2) + Safeguard tracked in `self.field` but NOT surfaced**~~ —
   **✅ RESOLVED (D52):** both now surfaced in `get_context()` (toxicspikes per-side
   0-2 count + safeguard per-side); `used_protect` also added (doubles-critical).
   Hazard state is now complete for the Step-4 observation.
2. **Posterior shape:** currently top-K (ability/moves/weight) + a scalar confidence
   + top_moves. Alternative: feed the full posterior vector (one dim per candidate set).
   **Recommendation: keep top-K + confidence (compact, already there); move to the
   full vector only if the net under-learns set uncertainty.**

**C. Network architecture (the actual Step 4 build) — OPTIONS, not built.** User
approved **3-4 hidden layers** (revises the prior 2-layer GRU-256); 6v6 singles,
50-100-turn horizon; observation ~205-dim (final count follows the B list).
1. **Recurrent core (RECOMMENDED): 3-4 layer GRU** over the turn sequence (each turn
   = one `get_context()` step) + per-mon encoder (128) for the active/bench slots +
   **policy head** (action logits: move × {enemies/allies/item/tera/switch}) +
   **value head** (scalar V(state)), LayerNorm between layers. Rationale: the
   50-100-turn per-turn structure is exactly what a recurrent net is for; 3-4 layers
   give the depth the user wants for the algorithmic/reasoning capacity.
2. **Alternative: Transformer over the turn window** (attention across the last K
   turns). More long-range look-back, but heavier and easier to overfit on a small
   battle corpus; only if the GRU plateaus.
3. **Heads:** policy = per-slot action distribution (action space must be enumerated);
   value = scalar. (A third "belief-critic" head predicting posterior entropy is an
   option, NOT recommended now — keep the net's job focused.)

**D. Non-goals for THIS stage:** no Step-4 code (net/sampler); no training loop.
(The calc-endpoint PREREQUISITE is now DONE — batch, D52.)

**Gate to start Step 4:** (1) calc endpoint — **DONE (batch, D52)**; (2) user confirms
B1 (hazard/protect obs — **DONE, D52**) + C1 (3-4 layer GRU) + the action-space
enumeration; (3) **319/319** stays green.

**E. hit_this_turn per-source fix — ✅ RESOLVED (D53, 2026-11-16).**
Root cause: `Facts.hit_this_turn: bool` = "was hit this turn by ANYONE", but the fork's
Avalanche/Revenge gate is PER-SOURCE (`attackedBy.some(p => p.source === target &&
p.damage > 0 && p.thisTurn)`, moves.js:926-934 / 15085-15093) — so "mon X hits me, I Avalanche
mon Y (≠X)" over-reported (doubled Avalanche when the fork wouldn't). **Fix (implemented,
320/320):** the bool is now `attacked_by_this_turn: set[(side,slot)]` (opponent slots that hit
this mon this turn, delta>0), set in `_apply_damage`, cleared on `turn_reset()`/`reset()`;
`_move_with_state(move, facts, defender)` computes the per-source PAIR `defender in
attacked_by_this_turn` at the two `_damage_evidence` call sites (both pass
`defender=(dmg.target_side, dmg.target_slot)`); the calc is UNCHANGED (still reads the
`hitThisTurn` bool). Observation keeps the model-facing bool `hit_this_turn = len(set)>0`
(no dim change). Tests: `test_hit_this_turn_is_per_source` (X-hits → Avalanche-X doubles,
Avalanche-Y does not) + updated `test_hit_this_turn_set_and_turn_reset`. **Remaining open
items from this class:** §3a-5 (Revenge not in calc, gen ≤7) + §3a-6 (reflector stale-hit).

**F. Per-gen move data (D56 + D57) — ✅ RESOLVED 2026-11-16 (Option A, user-approved; A2 done).**
Move metadata (type/category/power/accuracy/pp/priority) AND the effect flags (side_condition/side_status/boosts/drain/recoil) are now **per-generation in the DB** (`move_gen`, engine-keyed, fork-sourced, 1628 override rows) and `lookup.get_move(name, gen)` is gen-aware — so the Step-4 net + team-building see the correct move data for whatever format they train on (champions Slash 80, gen 1-3 Dragon Claw special, gen-1 Light Screen is a boost not a screen, Tail Glow +2 not +3). The D55 reflector **category** gate + the **speed-order priority** evidence both key off the format's engine key (`formats.engine_key(fmt)`). **A1 (6 data cols) + A2 (5 effect flags) both DONE** — move data is fully per-gen (schema v6).

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
| D29 | **This fork's DATA = source of truth for raw game data** (type chart, species typing/stats, move/ability/item tables) — *not* for damage formulas (those come from `@smogon/calc`; see the Sources-of-truth block, D37). Type chart verified to match the official chart (20 cells checked against `dist/data/typechart.js`). Species typing DOES deviate: `ninetalesalola` = **Ice/Fairy** (official: Fire/Fairy). Earlier note claiming "chart deviates (Fire→Fairy 1x, Rock→Bug 1x, …)" was an **axis mix-up** — Rock→Bug is 2× (super-effective); the 1× cell was Bug→Rock | tests assert fork-verified values for game data; never re-derive from memory of the official chart | 2026-11-07 |
| D30 | **Step 2 engine = @smogon/calc v0.12.0** (the official Smogon calc library) wrapped as a Node CLI (`damage/node_calc.js`) reading JSON from stdin; Python `damage/calc.py` spawns it and exposes `calc_damage(...)` + `compare_speed(...)`. Damage, critical hits (separate value), and immunity (→0, the library *throws* on 0-damage so we catch it) all come from the library — no formula re-implementation. Installed at project root (`package.json`); `node_modules/` stays gitignored | plan says "prefer wrapping"; the library is literally the Smogon calculator, so "match the official calc within rounding" is satisfied by construction and cross-checked against an independent formula | 2026-11-08 |
| D31 | **Speed helper & natures follow @smogon/calc / this fork (ground truth), not memory.** Natures: `Jolly=+Spe/−SpA`, `Adamant=+Atk/−SpA`, `Bold=+Def/−Atk`, `Modest=+SpA/−Atk`, `Naive=+Spe/−SpD`, `Serious=neutral` (fork `data/natures.ts` == @smogon/calc `NATURES`). **Gen 7+ paralysis = ×0.5** (not the classic ×0.25) — matches BOTH @smogon/calc `getFinalSpeed` and this fork's `par` handler, so the bot's speed reasoning stays consistent with the sim it fights on. Crit bypasses Reflect in the library model. Speed order: base → boosts → Tailwind(×2) → Scarf(×1.5) → par(×0.5) | my own nature/paralysis memory was wrong twice (Jolly, par ×0.25); always source these from the library/fork, never from memory | 2026-11-08 |
| D33 | **Battle-level bug found & fixed** (follow-up to D32; user: "champions not supporting level 100? don't want bugs left"). In this fork the battle level is a FORMAT rule, and our BSS target is a **level-50 format**: `Flat Rules` = "Adjust Level Down 50" (`data/rulesets.ts`), and `team-validator.ts:611-615,1144` rewrites any team at ≥50 to level 50 before battle. Our old DB `default_level=100` was `ruleTable.defaultLevel` (the pre-adjustment placeholder), NOT the battle level — so every quoted BSS number (242–288) was a level-100 number; real BSS battle numbers are **128–152** (verified). **Fix:** (1) schema v4 — `format` gains `min_level/max_level/adjust_level/adjust_level_down`; `default_level` := canonical battle level (BSS/VGC/champions **50**, LC **5**, ladder **100**), values baked into the registry from a live `TeamValidator` ruleTable dump of THIS server build, and `test_format_levels_match_fork` re-derives all 50 against the fork on every run (D29). (2) `legality.lookup.battle_level(fmt, level=None)` mirrors the validator (default → adjustLevelDown → adjustLevel; out-of-range level → ValueError, the validator would reject such a set). (3) damage API resolves each mon's level through the format when `fmt` is given — no silent level-100 default in a level-50 format; raw `gen=` path unchanged (library default 100). Reference values (Garchomp Adamant 252 Atk EQ vs Tyranitar Bold 252 Def): L100 242–288 · L50 128–152 · L5 24–30; champions 120–144 at ANY level. Also verified: the library hard-codes level 50 for champions (`pokemon.js:39`) — consistent with the fork's `Adjust Level = 50`, so champions + level 100 was always going to be silently mis-handled. **Gate: 68/68 green.** | user's hunch confirmed and extended to BSS; all values verified against fork team-validator/rulesets + library dist, not memory | 2026-11-08 |
| D32 | **Format-awareness investigated (within-gen mods DO exist).** `[Gen 9 Champions] *` formats (`mod: champions`, fork `data/mods/champions/`) change ~25 move base powers vs vanilla gen 9 (Slash 70→**80**, Anchor Shot 85→90, Beak Blast 100→120, Bloodmoon 120→130, First Impression 90→100, Bonerush 25→30, …), plus types (Growth→Grass, Snaptrap→Steel), accuracy (Crabhammer→95), PP (Beak Blast→5), 201 moves banned, and status mechanics (par = ×0.5 **and** 1/8 move-skip; sleep/freeze durations). **@smogon/calc v0.12.0 implements champions natively as engine key `gen=0`** (`MECHANICS[0]` in `dist/calc.js`); verified library champions Slash BP=80 == fork `mods/champions/moves.ts` (gen-9 Slash=70). Our DB already contains 2 champions formats (`gen9championsou`, `gen9championsvgc2026regmc`; `gen=9`, `mod='champions'`). **Therefore the engine key for a format = its `mod` (`genN`→N, `champions`→0), NOT its `gen` column** — a naive `format.gen` lookup would silently mis-calc the champions rows. User: calc must serve **all** formats (BSS primary). `Move(gen, name, {overrides:{…}})` deep-merges into the formula if finer per-format tweaks are ever needed. **Implemented (user-approved 2026-11-08):** `legality.formats.engine_key(fmt)` (registry `mod` → key; single source of truth) + `fmt=` param on `calc_damage`/`effective_speed`/`compare_speed` (fmt wins over `gen`; unknown format → clear error). Tests: BSS ≡ gen-9 (242–288 unchanged), champions EQ = **120–144** (level-independent at L50 and L100 — the champions engine is a *different damage model*, not just BP tweaks; the fork's own `mods/champions/scripts.ts` implements its `modifyDamage`, so the library model tracks the sim), champions Slash ≠ gen-9 Slash. **62/62 green.** Follow-up (optional, before champions formats are used in training): cross-check a champions matchup against a live fork battle | investigation requested by user ("champions is gen 9 technically? different base powers?"); verified against fork ground truth + library dist, not memory | 2026-11-08 |
| D34 | **calc↔sim cross-check DONE + a real engine-contract bug found & fixed.** (a) **Verified:** `@smogon/calc` matches the LIVE fork sim for the standard damage pipeline at level-50 BSS — 11 mechanics (plain P/S, multi-hit, burn, Life Orb, Choice Band, +2 boost, Rain, Sun, Reflect, Adaptability) × **24 random seeds, all 11/11** (`scratch_crosscheck_calc_vs_sim.js`, seeded/reproducible). (b) **The prior "Liquidation category-flip" root cause was WRONG:** fork and `@smogon/calc` AGREE on category/basePower/type/multihit for all **640 gen-9 damage moves** (0 mismatches; the only calc gaps are the 11 Tera-boost moves, which don't apply to BSS). The flaky FAILs were a **harness measurement bug**: the sim logs each move's damage TWICE (absolute `157/170` + a `% of max` `93/100`); summing log lines double-counted. Fixed by measuring the defender HP delta and filtering `[from]` residuals + the `%` line. (c) **REAL BUG FIXED (belief-state accuracy):** `@smogon/calc` resolves **item & ability by DISPLAY NAME** and **silently ignores ids** (`lifeorb`/`adaptability` → no effect), but the AI/legality-DB and the sim fork all carry **ids** → every item/ability effect would have been dropped for real teams. Fixed in `damage/node_calc.js` (`normItem`/`normAbility` translate id→name using the fork's own items/abilities tables as ground truth; accepts ids AND names; `effectiveSpeed`'s `Choice Scarf`/`Quick Feet` checks made id/name-agnostic). Species & move names are already normalized by the library (id or name both work) — only item/ability needed it. **+3 regression tests** (`test_item_id_matches_display_name`, `test_ability_id_matches_display_name`, `test_item_id_not_silently_dropped`). **Gate: 71/71 green.** | user: "test the damage calc for ~10 random matchups… our assumptions of the AI must be accurate, the belief state needs to be clean" | 2026-11-09 |
| D35 | **Guts / Aurora Veil / burn — re-verified against the fork, and one of MY OWN test expectations was wrong (now corrected).** (a) **Guts:** fork ground truth = `guts.onModifyAtk` (any status ⇒ Atk ×1.5) + `battle-actions.js:1556` (burn halving **skipped** for Guts users) ⇒ a **burned Guts user hits at 1.5×, not 0.5×**. `@smogon/calc` implements exactly this (gen789: `hasAbility('Guts') && status && Physical` ⇒ atMod 6144; `applyBurn` requires `!hasAbility('Guts')`). The earlier "Guts missing" reading was the id-vs-name contract bug again (id `guts` silently dropped → 0.5×) — already fixed by `normAbility`, so **no engine change was needed**. (b) **Aurora Veil:** fork `auroraveil.onAnyModifyDamage` has **no category filter** → the veil halves **ALL** move categories (not just special), and does not stack with Reflect/Light Screen. The library matches the fork. My first test ("veil doesn't affect physical") was **wrong** — corrected to `test_aurora_veil_halves_physical_too` (852-1004 → 426-502, live-sim confirmed 249/325). (c) **Burn:** halves physical only, special unaffected (Torkoal 97-115 with/without burn); crits ignore the burn (healthy 852-1004 → burned-Guts crit 1920-2258 = ×1.5). (d) **Weather:** Rain ×1.5 Water (366-432) / Sun ×1.5 Fire (107-126) unchanged & re-asserted. (e) **+10 permanent tests** (`test_guts_*`, `test_aurora_veil_*`, `test_burn_*`), values pulled from the engine + Guts & veil cross-checked vs the live fork sim (13/13 over 4 seeds). **Gate: 91/91 green.** | user: "can you check with weather, aurora veil and burn again? also the guts ability" | 2026-11-09 |
| D36 | **Damage-ability audit (official @smogon/calc = ground truth, per user) + the ONE real wrapper gap fixed: current HP was never passed, so every HP-gated effect was wrong.** (a) **Ability name contract = 192/192 clean** — every ability the official calc's `hasAbility(...)` set branches on maps id→display-name correctly through `normAbility` (`scratch_enum_abilities.js`); none silently dropped, so the Guts-class "id-vs-name" bug (D34/D35) is fully closed. (b) **"Missing ability" reframe ruled out** — abilities the official calc does NOT model (Mycelium Might, Skill Link, Mind's Eye, Stalwart, Stance Change, Stench, Propeller Tail, Serene Grace, Magician) are intentionally not applied; matching the official calc is the requirement, so we do not add them. (c) **REAL GAP FIXED:** @smogon/calc gates effects on `curHP()`, but `buildPokemon` never passed it (library assumed FULL HP) → low-HP attacker effects never fired AND full-HP-only defender effects always did. Fixed in `node_calc.js buildPokemon` + documented in `calc.py`: each mon now accepts **`hp`** (0.0-1.0 fraction of its max HP — the AI-facing, level/IV/EV-agnostic form the belief state knows) and **`curHP`** (absolute, clamped [1,maxHP]); omitting both preserves full-HP semantics (all prior tests unchanged). Now correctly modeled: **Defeatist** (≤½ HP, Atk&SpA ×0.5), **Overgrow/Blaze/Torrent/Swarm** (≤⅓ HP, matching-type ×1.5), **Eruption/Water Spout/Flail/Reversal** (BP scales / inverse-scales with own HP), defender-side **Multiscale/Shadow Shield/Tera Shell** (only AT full HP; in gen789 Tera Shell caps type effectiveness to 0.5, so vs a 2× move it reads ×0.25), **Brine** (defender ≤½ → ×1.5). (d) **Consciously out of scope for BSS gen 9 (flagged for the Step 3 belief sampler, NOT fixed):** `abilityOn` (Flash Fire/Plus/Minus "active" state), `gender` (RBY only), `isDynamaxed`/`alliesFainted` (Dynamax/Parental Bond); weather abilities (Drought/Drizzle/Sand Stream/Snow Warning) are correctly modeled via the `field.weather` state, not the ability — do NOT "fix". **Tests:** +21 `damage/tests/test_curhp_abilities.py` — every effect asserted by direction + concrete engine-verified range + id/name agreement, plus `curHP` alias, out-of-range → error, and full-HP controls — **gate 112/112 green.** | user: "look for other abilities that might influence damage on the official smogon calculator and fix issues with those before progressing" + "please use smogon damage calc as ground truth" | 2026-11-09 |
| D37 | **T1 (non-standard-damage) sweep DONE + source-of-truth disambiguation.** (a) **Reframe (user, 2026-11-10):** `@smogon/calc` is the **source of truth for the damage model**; the fork is the **battle environment + validation oracle** (where the AI observes damage, so the calc must match it) and the **source of truth for raw game data** (Dex typing/stats/tables). Recorded in the Sources-of-truth block at the top; D29 scoped to raw game data. (b) **T1 sweep (`scratch/sweep_t1.js`):** all 11 `damageCallback` moves enumerated; **8 BSS-usable** verified against the fork (the environment) and the calc's coverage mapped. **`@smogon/calc` already handles:** `finalgambit` (attacker HP), `naturesmadness` + `guardianofalola` (defender HP-fraction). **`@smogon/calc` returns 0 (gaps to complete):** `superfang` (`target.hp/2`), `ruination`/`naturesmadness` (`floor(target.hp/2)`), `endeavor` (`target.hp − attacker.hp`), and the 4 same-turn reflectors `counter`/`mirrorcoat` (`2× last hit`) / `metalburst`/`comeuppance` (`1.5× last hit`, both `thisTurn`-gated; Counter/Mirror Coat priority −5, the other two priority 0 so speed-ordered). `psywave` (Past) also 0. Fork self-consistency all PASS (superfang 162, ruination 162, finalgambit 85, endeavor 215, counter 32=2×16, mirrorcoat 18=2×9, metalburst 3=1.5×2, comeuppance 40=1.5×27). (c) **Decision on gen-specific moves (user idea: `superfang(g1)` vs `superfang`):** **do NOT gen-tag move names** — `gen` is already a first-class context in both `@smogon/calc` (`MECHANICS[gen]`) and the fork (`forGen(N)`), and the move id stays `superfang`; gen-tagged names would create parallel ids that don't match the fork/calc and duplicate the `gen` param. **Instead: complete the calc to be gen-aware** (e.g. Super Fang respects Ghost immunity in gen 2+ but not gen 1; Strong Jaw does not boost it) using the `gen`/`fmt` the wrapper already receives. No DB/move-name changes; it's a Step 2 addition. | user: "should we change moves in the db by their gen… maybe we can just fix the damage calc?" + "the smogon damage calc [is] the source of truth… change that in the documentation" | 2026-11-10 |
| D38 | **T2–T4 special-damage sweep DONE + type-immunity gate rebuilt fork-exact (146/146).** (a) **T2 level-scaling:** Nightshade/Seismic Toss = attacker level (fork `damage:"level"`); Psywave = 50% defender HP (library already). (b) **T3 OHKO:** Fissure/Guillotine/Horn Drill = always KO — reported as defender maxHP with an `ohko:true` flag; Sheer Cold = KO or Ice-immune. (c) **T4:** Explosion/Self-Destruct/Misty Explosion need no change (library base-power path; ranges 401–472 / 321–378 / 31–37 fork-sample-consistent); faint-support moves (no standalone damage) report 0. (d) **Type-immunity gate — rebuilt.** `@smogon/calc` applies **no type immunity** to formula/damageCallback moves, so `node_calc.js` now gates them with a **fork-exact precomputed table** `damage/special_move_immunity.json` (schema `{engineKey 0–9: {moveId: {immune:[types], ground?, ohkoIce?}}}`; regenerate: `scratch/gen_special_move_immunity.js` — uses the fork Dex per format: champions→key 0, gen1–9→randombattle formats; reads move type, `ignoreImmunity`, `ohko`, chart `damageTaken===3`). The fork's `runImmunity` (`pokemon.js:1664`) DOES apply to formula moves — **immunity (0×) only**: no 0.5×/2× multipliers, no STAB (nightshade vs Ghost = level; superfank vs Steel/Rock = plain ½ HP; superfank/nightshade/seismictoss carry `ignoreImmunity:true` in gen-1 fork data → they HIT Ghost in gen 1, hits preserved by `test_superfang_hits_ghost_in_gen1`). **Why a table, not a live chart lookup:** the calc must stay quick + API-reachable for deep-tree AI calls (static JSON, no per-call fork spawn), AND a prior gate attempt used **move types from memory and was wrong** (naturesmadness=Fairy not Psychic, finalgambit=Fighting not Psychic, seismictoss=Fighting not Normal, ruination=Dark not Fairy, guardianofalola=Fairy, comeuppance=Dark) — the fork's data was correct throughout. (e) **Request-data-dependent rules mirrored from the fork's dispatch order** (`battle-actions.js:1400`): Ground moves (Fissure) immune vs Flying/Levitate/Eelevate/Air Balloon **unless Iron Ball** (fork `isGrounded`, `pokemon.js:1592` checks Iron Ball FIRST — it grounds and overrides Flying+Levitate); Sheer Cold's Ice-immunity via its `ohko:"Ice"` rule. (f) **+12 tests** (`damage/tests/test_special_damage_moves.py`, now 29): T2 level moves vs Gengar (deliberately not Chansey — a Normal target would be immune anyway), T3 maxHP+`ohko` flag / partial-HP KO / Sheer Cold vs Articuno, T4 ranges + faint-support 0, immunity cases (seismictoss/guillotine/horndrill/endeavor vs Ghost→0, nightshade vs Normal→0, fissure vs Flying→0, fissure vs Levitate→0, **fissure vs Levitate+Iron Ball→135 KO**), positive controls (nightshade vs Ghost=50, no 2×; superfank vs Steel/Rock=72, no 0.25×). **Gate: 146/146 green.** | user: "sweep through BSS moves first… narrow it down to moves that could cause bugs due to different damage outputs or special effects" + belief-state correctness (immunity-aware damage → clean HP/EV inference, e.g. Seismic Toss reveals max HP) | 2026-11-10 |
| D39 | **T5 state-dependent base-power (bpCallback) sweep DONE + champions-engine hardening (163/163).** (a) **Scope (narrowed, per user):** of the 54 `bpCallback` moves, @smogon/calc models ~40 (Payback/Hex/Grass Knot/Low Kick/Wring Out/Flail/… in the gen789 `calculateBasePowerSMSSSV` switch); a **gen-9 learnset filter** (only moves a gen-9 species can learn are BSS-relevant) leaves **9 actionable, all BSS-legal**: `avalanche, beatup, firepledge, grasspledge, waterpledge, lastrespects, spitup, stompingtantrum, temperflare` (iceball/pikapapow/return/trumpcard/veeveevolley have **no gen-9 learners** → doc-only). (b) **Root cause:** the library falls to `default: basePower = move.bp` — `beatup`/`spitup` have data basePower **0** → returned **0 damage** (hard belief-state bug: the AI would think they deal nothing); the rest returned the correct no-state baseline but could not see the transient state that re-scales them. (c) **Unified mechanism (user-approved: one formula, quick, API-reachable):** per-engine base-BP table `damage/special_bp_moves.json` (regenerate: `scratch/gen_special_bp_moves.js`; **fork = source of truth for the raw move data**; champions bans the 3 pledges → `null` there; gen2–4 beatup base 10, gen5+ 0; gen5 pledges 50) + `effectiveBP(gen, moveId, st)` computing the effective BP from **belief-state flags** (fork formulas, `scratch/probe_t5_formulas.js`, 20/20 fork-verified via the fork's own `basePowerCallback`) → handed to @smogon/calc via `Move(gen, id, {overrides:{basePower}})`, so STAB/typing/boosts/immunity/crits stay in the ONE library pipeline (no second damage model, no per-call fork spawn). (d) **Formula corrections (my earlier memory was wrong; fork was right):** Stomping Tantrum/Temper Flare double on **previous-move-FAIL** (`moveLastTurnResult===false`), NOT "if hit"; Last Respects = **absolute** `50 + 50×fainted` (a normal damage move), NOT "50% of fainted ally HP"; pledges on a combo = **absolute 150**, not ×1.5; Spit Up = `100×stockpile layers` (0 layers → fails); Beat Up = multi-hit, each ally `5 + floor(baseAtk/10)` BP with its OWN stats. (e) **State-flag API (calc.py, all optional, absent = no-state baseline):** `move.hitThisTurn` (avalanche), `move.lastMoveFailed` (ST/TF), `move.pledgeCombo` (pledges), `move.allyFainted` (lastrespects), `move.stockpileLayers` (spitup), `move.beatupAllies` (array of ally specs). (f) **Champions-engine hardening (2 real bugs found & fixed this session):** (1) pledges are `isNonstandard` in engine 0 → `stateMoveDamage` returned `null` and `handleDamage` **fell through to the normal path, which crashed** on the banned/inherit move data (`Cannot read properties of undefined (reading 'hp')`); a STATE_BP move that is unusable in the engine now reports **0/immune** (never `null`, never falls through). (2) The champions species table is a **359-species BSS whitelist** (library `CHAMPIONS_LIST` in `dist/data/species.js`); a species outside it (e.g. **chansey, hitmonlee are NOT champions-legal**) made the library's Pokemon constructor crash opaquely inside `calcStat`; `buildPokemon` now checks `Generations.get(gen).species.get(id)` and throws a clean `pokemon "X" is not legal in the gen-N engine` error — **never an opaque crash, never silent numbers for a species that cannot exist in that format** (the AI builds teams from the legality DB, so this is a loud guard against bad state, not an expected path). (g) **+17 tests** (`damage/tests/test_state_bp_moves.py`): each move's baseline + triggered state engine-verified (e.g. avalanche 122-144 → 243-287 hit; lastrespects 60-72 / 118-141 / 178-210; spitup fail→0, 1 layer 28-34, 3 layers 84-99; ST 151-178 → 301-355; TF 229-271 → 459-541; beatup no-ally fail→0, mankey+charizard 55-65), pledges 0 in champions (Blastoise defender — Fire hits 2×, so any leak would show), illegal-species clean error, Ghost-immunity (lastrespects vs Normal→0) and crit-block sanity. **Gate: 163/163 green.** | user: "lets continue with the tier plan please" (T5 tier) + "i need the calc to be right to correctly deduct EVs/max HP" + "a unified formula for multiple moves if they work the same" | 2026-11-15 |
| D40 | **T1–T5 double-check COMPLETE (user: "please double check with the smogon damage calc all previous move types").** Re-verified every special move type shipped in Step 2 (T1 reflectors/fixed-damage, T2 level-scaling, T3 OHKO, T4 self-damage, T5 state-BP) with `scratch/verify_all_special_moves.js`: **29/29 scenarios PASS + 20/20 fork basePowerCallback (layer-A) checks PASS; full suite 163/163.** **Conclusion: zero calc bugs in T1–T5 — all 8 prior failures were harness/scenario artifacts** (fork and @smogon/calc agree in every case, evidence from fork code): (1) **fissure** — fork emits the OHKO KO damage line **twice** (`|-damage|…|0 fnt` ×2); harness was summing the phantom 0. (2) **guillotine/horndrill/sheercold** — 30%-accuracy OHKO missed under the fixed seed; harness now retries up to 100 seeds until the KO lands. (3) **mirrorcoat** — Mirror Coat only reflects **SPECIAL** moves; scenario used Aqua Jet (physical). (4) **lastrespects** — Tackle (Normal) is immune to Houndstone (Ghost), so nothing ever fainted; correct setup = ally faints at 1 HP, attacker then switches in (BP100). (5) **stompingtantrum / temperflare** — trigger is a failed **ATTACKING** move (fork `moveLastTurnResult===false`); a *successful* status move (Splash) does NOT set it — scenario premise was wrong, now a Normal-vs-Ghost miss. (6) **comeuppance** — KO clipping (see 8). (7) **avalanche** — the sim value was a **crit**; harness is now crit-range-aware. (8) **KO clipping = real log-semantics fact (keep for Step 3 belief state):** when a move KOs, the log line is `|-damage|…|0 fnt` and the measured "damage" is **maxhp, not the true (higher) damage** (Come Up Pance true 171 vs Samurott maxhp 170) → belief-state damage inference must treat a KO-hit as `damage ≥ maxhp`. No calc code changed this session; `node_calc.js`, data tables, and tests unchanged. | user: "yes please continue / but before, please double check with the smogon damage calc all previous move types" | 2026-11-15 |
| D41 | **T6 recoil/drain sweep DONE (14/14 live-fork harness, 180/180 gate) + the gen1–4 FLOOR-recoil root cause found & fixed.** (a) **Scope:** the 29 recoil/drain moves form the SAME set in all 10 engines; per-engine values differ (fork-exact table `damage/recoil_drain_moves.json`, regenerate `scratch/gen_recoil_drain.js`): Struggle gen1 recoil [1,2] bp50 → gen2–3 [1,4] → gen4+ struggleRecoil; Double-Edge [25,100] gen1–2 → [1,3] gen3–4 → [33,100] gen5+; Drain Punch / Giga Drain drain [1,2] everywhere; Mind Blown / Chloroblast = `round(maxhp/2)` in all engines. (b) **ROOT CAUSE of the off-by-1 (headline):** the fork has TWO `applyRecoilDamage` variants — base uses `Math.round` (`dist/sim/battle-actions.js:1216`) and **`dist/data/mods/gen4/scripts.js:167` OVERRIDES it with `Math.floor`**; the mod inheritance chain is gen1→gen2→gen3→gen4→gen5→base, so **gens 1–4 use FLOOR, gens 5–9 + champions use ROUND** — nothing else overrides. Live-verified: gen1 Struggle dealt 29, recoil [1,2] → `floor(14.5)` = **14** (round would say 15). The same split governs drain (`dist/sim/battle.js:1793-1800`: gen≤4 `clamp(floor(dmg×n/d),1)`, gen>4 round) and struggleRecoil (gen1–4 `floor(baseMaxhp/4)` vs gen5+ `round(baseMaxhp/4)`); mindBlown/chloroblast are `round(maxhp/2)` in BOTH variants. Fixed in `node_calc.js` (recoil block + header doc). (c) **Clamped-damage semantics (verified):** recoil/drain fire only when the move DEALT damage (`battle-actions.js:905`) and are computed on the CLAMPED damage — `battle.damage` clamps to the defender's REMAINING HP (live: champions Brave Bird 237 computed vs 154-HP Blastoise → 154 dealt → recoil `round(154×33/100)` = 51); `heal` is capped by the user's MISSING HP — that cap is left to the AI, the calc reports the raw heal bracket. (d) **Life Orb:** uniform `floor(baseMaxhp/10)` in all 10 engines in this fork (`dist/data/items.js:3427-3443`), non-Status guard, gen6 mod adds `!move.ohko` (fork OHKO set: fissure/guillotine/horndrill/sheercold). (e) **The gen1 "2× damage divergence" was a CRIT — resolved:** live gen1 Pikachu Struggle dealt 29 = a gen1 crit (`dist/data/mods/gen1/scripts.js:636-785`: `level*=2` on crit, randomizer 217–255, crit roll `randomChance(critChance,256)`), and 29 ∈ the library's crit range 27–32 — **fork and @smogon/calc AGREE on gen1 damage**; the only real gen1/2 gap is the ±1 stat-model divergence (fork uses the modern formula; the library uses the authentic RBY DV/stat-EXP model; they agree at IV31 + EV0). Note: gen1 Struggle is type Normal in the fork data, but a gen1 Electric-only attacker has no Normal typing → no STAB (probe-verified). (f) **`critChance` response field (new):** base fork crit rate per move/engine — 1/24 gens 7–9 + champions, 1/16 gens ≤ 6, 1/8 for critRatio-2 moves (e.g. Slash); absent on special-damage/immune moves; battle-state modifiers (Super Luck, Scope Lens, Focus Energy, Lucky Chant, Merciless, Battle/Shell Armor) are left to the belief state. (g) **Tests:** +17 in `damage/tests/test_recoil_drain.py` (recoil/drain/struggleRecoil/lifeOrb/status-immune × engines; gen1–2 cases at EV0 where the models agree); harness `scratch/verify_recoil_drain_fork.js` 14/14 live-fork; T1–T5 keeper harness re-run green (29/29 + 20/20). | user: "continue with t6 please" + belief-state correctness (recoil/drain observed in battle must be inferred exactly) | 2026-11-15 |
| D42 | **Step 2 audit (user: “no, please check for mistakes and bugs first and clean up the code”) — 3 REAL bugs found, fixed in both files, +42 tests, 222/222 gate.** (a) **B1 (BSS-critical) — Tera type missing from the special-move immunity gate.** The gate checked only BASE types, but the fork's `getImmunity` (`dist/sim/dex.js:226`) uses `target.getTypes()`, which returns the terastallized type when terastallized and not Stellar (`dist/sim/pokemon.js:1583`); the library likewise overrides effectiveness with `defender.teraType` (`dist/mechanics/gen789.js:292-294`). Consequence: Super Fang wrongly “hit” a Tera-Ghost Sandshrew in BSS (0× in battle). Fixed with an `effectiveTypes(gen, mon)` helper (engines **9 and 0 only** — Tera doesn't exist in gens 1-8; 'Stellar' ignored; `toID`-normalized), used in `specialMoveImmune` (chart list + the Flying/Ground check) and `specialMoveDamage` (Sheer Cold's Ice check). (b) **B2 — paralysis speed factor wrong for gens 1-4 and 6.** Fork: gens 1-4 ×0.25 (`mods/gen4/conditions.js:30`), gen 6 ×0.25 (`mods/gen6/conditions.js:31`), gen 5 ×0.5 (base, no override), gens 7-9 + champions ×0.5. Old code used ×0.5 everywhere → gen 1-4/6 speeds were 2× too high. Fixed with the per-gen table. (c) **B3 — the speed endpoint was missing most of the library's `getFinalSpeed`** (`dist/mechanics/util.js:131-173`): Swift Swim+Rain, Chlorophyll+Sun, Sand Rush+Sand, Slush Rush+Hail|Snow, Surge Surfer+Electric terrain, Quark Drive/Protosynthesis, Unburden, Quick Feet+status, Slow Start, Iron Ball/EV items ×0.5, Quick Powder+Ditto, `chainMods`/`pokeRound` rounding (`util.js:487,696`), the gen 1-2 **999 cap**. Now a full hand-mirror (the library does not export `getFinalSpeed`; `index.js:41` exports only). New OPTIONAL inputs (backward-compatible, all prior tests unchanged): mon flags `abilityOn` / `boostedStat`, and a top-level `field` (weather/terrain) on the speed request; a `main()` wiring bug (field not forwarded) found & fixed during the smoke test. (d) **B4 cleanup:** triplicated `pct` helper unified into `pctOf`; docstrings updated in both files. **Verified NOT bugs (don't re-investigate):** Struggle CAN crit in the fork (data default `critRatio:1` for ALL moves) → 1/24 `critChance` is fork-consistent; `applyBoost` tables matched fork `pokemon.js:302` exactly; gen 1-2 library EV default 252 (tests rely on it). **DECISION (flag to user): gen-5 paralysis — FORK ×0.5 wins over the library's ×0.25** (all other gens agree; the fork is the battle environment, so who-moves-first follows the fork); documented in code, `calc.py` docstring, and pinned in tests (`ours=75` vs library `37`, Jolteon L50). **Evidence:** `dex.js:226-236`, `pokemon.js:1583-1586` + `runImmunity` `pokemon.js:1664`, base `dist/data/conditions.js:43-62` + the gen4/gen6 par mods, `util.js:58-91/131-173/512-549/696`, `dist/pokemon.js:39-90`, `gen789.js:292-294`. **Tests:** +42 in `damage/tests/test_audit_fixes.py` (13 Tera/immunity incl. champs-whitelist clean-error, 10 per-gen par incl. the gen-5 divergence pin, 19 full-mirror speed cases, compare_speed); keeper `scratch/verify_speed_tera.js` **40/40** (28 speed scenarios EXACT vs the library's own `getFinalSpeed` + 12 Tera cases); T1-T5 keeper 29/29+20/20 and T6 keeper 14/14 re-run green; **gate 222/222** (was 180). | user: “no, please check for mistakes and bugs first and clean up the code” (pre-Step-3 quality pass) | 2026-11-15 |
| D43 | **Ability-reveal environment fact (reserved early; full entry lands at 3.8).** In this fork the mon's ability is **NOT revealed at switch-in** — there is no `\|ability\|` line on entry. It is revealed **contextually** when an effect fires, as `[from] ability: <Name>` with the `[of] <holder_slot>` tagging the holder (e.g. Guts boosting Atk, Levitate resisting a Ground hit). The 3.1 log→event adapter attributes the reveal to the `[of]` holder slot and emits `AbilityRevealed`. **Belief consequence (the "make unknown gamestates known" principle):** the ability stays **hidden in the prior** — every candidate keeps its ability dim and the prior spreads mass over the species' legal abilities — and is narrowed to the revealed ability ONLY on an `AbilityRevealed` event (HARD evidence, exact 0/1 via `revealed_ability`). This is why the prior keeps ability as a real dim (not fixed at init) and `test_prior_*` assert it. Full entry for the §5 battle-log protocol note. | the ability is unknown until the log reveals it — never assume it at switch-in | 2026-11-16 |
| D44 | **Step 3.4 SOFT-evidence gate — a real latent bug in the belief calc bridge found & fixed (the 3.3 suite never exercised the SOFT path, so it shipped untested).** `evidence.mon_options` read `spec["name"]`, but a belief **candidate carries no `name`** — the species is a separate observation on `SlotBelief.species` (the candidate is only the hidden set dims: moves/item/ability/nature/evs/tera). Every candidate-side window/speed call (`defender_window`, `attacker_window`, and the `_pair_speed` candidate `mon_options`) raised `KeyError: 'name'`, so the entire 3.4 damage + speed SOFT evidence was dead on arrival. The `defender_window(candidate, species, …)` / `attacker_window(…, species, …)` signatures already carried the `species` they were supposed to use but never did — the fix makes `mon_options` accept an optional `name` (defaults to `spec['name']`, so own-team specs are unchanged) and the candidate callers pass `name=species` (the belief slot's species). Also normalized a `WeatherChanged('none')`/`TerrainChanged('none')` clear-signal to `field=None` (was the literal string `'none'`; harmless to the calc but semantically wrong). **Verified against the Step-2 calc (source of truth) before pinning any number** — the handoff's hardcoded windows were partly stale (e.g. Adamant 252 Atk Earthquake vs Blastoise is [180,213] here, not [163,193]; Duskull/Insomnia is [68,81], not [81,95]), and its KO-clip scenario was unworkable by design (the no-item candidate's *crit* already KOs, so the KO-clip rule `max>=maxhp or crit_max>=maxhp` — which is semantically correct, crits are possible — counts it as KO-consistent); a clean KO/never-KO pair (Garchomp strong [286,339] vs weak [166,196], crit 294 < 301 max, Golduck) was used instead. **Tests:** `belief/tests/test_step3_4.py` **13/13** (damage rule + both directions + KO-clip + immunity + speed order (incl. priority-differs & switch-breaks) + fact sheet + field/Tailwind + live seeded-battle pipeline); belief suite **64/64**; **full gate 286/286**; keepers 40/40 + 29/29+20/20 + 14/14. | user: “i want strong checkpoints… we need to catch errors” + “i need the calc to be right” (3.4 gate) | 2026-11-16 |
| D45 | **Step 3.5 sampling + move-probabilities gate — implemented per the §3a bold recommendation (reversible, pure belief ops, no calc touch).** `SlotBelief.sample(n, seed)` is a SEEDABLE multinomial draw of **full sets** (not per-move marginals): the correlation-preservation property the user asked for ("the moves are correlated — wish means 90% protect") falls out by construction, since a drawn set already carries its co-moves. Fresh `random.Random(seed)` per call ⇒ reproducible AND no RNG state leaks across calls or slots; `copy.deepcopy` per sample ⇒ mutating a draw can't corrupt the live posterior (a perf-bench optimization in 3.8 if `deepcopy` ever shows up). `move_probabilities()` = posterior mass over the sets CONTAINING each move — membership, order-blind (3.3 invariant), values in [0,1], sum ≈ expected move count (~4), NOT 1 (a mon carries several moves; mass(move) = mass of the sets holding it); PP ignored (unobservable in the log). No new dependency, no calc change. **Tests:** `belief/tests/test_step3_5.py` **9/9** (reproducible-with-seed + stateless + copy-safety + `n<0` raises + ruled-out≈never-drawn + membership order-blindness + range/sum + the spec Wish⇒Protect 0.9 correlation both via `move_probabilities` and via `sample`). **Self-audit (user: "bug fix and question your own code") — 14/14 edge probes green, no functional bug:** underflow of a mismatched candidate to a literal 0.0 weight (sample() still works — the correct set is always the positive one), a single fully-contradicted candidate never empties (renormalizes to 1.0), deep-copy safety on nested `evs`, update→sample posterior shift, seeded reproducibility. **One CONTRACT trap found & hardened (not a code bug, a misuse risk):** `move_probabilities()` returns per-move MARGINALS; multiplying two to get a joint is the independence error the user explicitly forbade — P(Wish∧Protect)=0.5 but P(Wish)·P(Protect)=0.25 — so both docstrings now say "use `sample()` for any joint/correlated question" and `move_probabilities()` is also NOT a per-turn usage probability (it's in-set membership; turn prediction is the policy's job). Added `test_marginals_are_not_independent` (pins the correct correlated joint via `sample()`). `belief/tests/test_step3_5.py` **10/10**; belief suite **74/74**; **full gate 296/296**. | user: "i like the bayesian way… the moves are correlated" + "i want strong checkpoints" (3.5 spec) | 2026-11-16 |
| D46 | **Step 3.6 max-HP / EV-bucket inference — regression gate + a max-HP-consistency EV evidence (the "residual-damage → maxHP" signal, in its clean IV-safe form).** (a) **Regression gate (Part A):** the 3.4 window machinery now does the three things the user cares about, locked in `belief/tests/test_step3_6.py` — **window** (Seismic Toss = flat attacker level = a degenerate [50,50] band at BSS level 50, non-critting; tested at the WINDOW level, since a single-candidate belief can't be "ruled out" — the never-empty EPS floor renormalizes it back to 1.0), **EV-narrowing** (a defender's SpD EVs and an attacker's Atk EVs shift the calc window, so an observed hit discriminates the spread; picked hits in the non-overlapping halves), **Guts** (burned Guts ×1.5 Atk [86,102] vs burned non-Guts ×0.5 Atk [28,34] — the 3.4 window already discriminates; no new code). (b) **The new piece (Part B) — max-HP-consistency evidence:** the switch-in line reveals the mon's max HP (verified 3.1: the adapter learns it from the opponent's switch-in incl. the maxhp==100 dedupe + slot-reuse cases). max HP is a function of the HP-EV spread, so the observed max HP is EV evidence. `maxhp_for(candidate)` reads the Step-2 calc's `maxhp` field (= the DEFENDER's maxHP at IV 31, i.e. the spread's maxHP *maximum* — probe-verified immune-safe: a Normal-immune defender still returns its maxhp; one-sided ordering EV0 165 ≤ EV252 197; Machamp base HP 90 in this fork). `maxhp_likelihood` is ONE-SIDED / IV-safe: `observed > candidate` ⇒ IMPOSSIBLE ⇒ ruled out (EPS floor); `observed ≤ candidate` ⇒ CONSISTENT (the real mon may run lower IVs) — so a higher-HP-EV candidate is NEVER falsely ruled out. Wired on the p2 switch-in (NEW-mon branch; idempotent on re-entry — a consistent candidate keeps the same relative weight, so a same-mon re-entry is a no-op, not a double-count). (c) **The "residual-damage (burn/poison)" reframe — OPEN DECISION (user asleep 2026-11-16, flagged for review, NOT user-approved):** the §3.6 plan says "residual-damage likelihoods (burn 1/16, poison, recoil) land here". I reframed that into the max-HP-consistency evidence because a burn = max(1,⌊maxhp/16⌋) and a poison = max(1,⌊maxhp/8⌋) — BOTH are just another observation of the SAME max HP, and the switch-in line is the direct observation of it. **A (implemented, recommended):** max-HP-consistency on the switch-in line — cleanest, IV-safe, no false rule-outs, fully gated. **B (not implemented; a local swap inside `_apply_maxhp_evidence`):** consume the literal burn/poison damage LINE (derive the implied maxHP band from the damage via the fork-verified rate, then apply the same one-sided rule). **DEFERRED (documented, not implemented):** Toxic (stage-dependent rate), recoil (attacker-side, no opponent-set signal), hazard damage, and any two-sided maxHP band (IV-unsafe → false rule-outs). **Gate:** `belief/tests/test_step3_6.py` **5/5**; belief suite **79/79**; full gate **301/301**. No calc/damage change (reuses the existing `maxhp` field). | user: "continue with 3.6" + "write down open decision options since i am going to sleep" | 2026-11-16 |
| D47 | **Step-1 DB data bug found & fixed (unblocked 3.7) — learnset filter + "Past"-tag override (Option B) + D33 level.** The 3.7 fuzz gate was RED on battle 0: `no prior for 'machamp' in 'gen9bss3v3': no sets and no legal moves`. **Root cause (correcting the prior handoff, which wrongly said "the DB is stale; a rebuild fixes it" — a rebuild reproduces the gap):** (1) this fork's `data/learnsets.ts` is missing gen-9 entries for ~454 common species (learnset codes cap at gen 8 — machamp max-gen 8, garchomp 9), so the fork's dex marks them `isNonstandard:"Past"`/`tier:"Illegal"`; (2) the extractor's `checkSpecies` rejected them → absent from `format_form` → `legal_moves()` = [] → `build_prior` raised. **The fork is internally inconsistent:** its own `validateTeam` *accepts* a machamp/abra set in BSS/OU while `checkSpecies` *rejects* it as "Past" — so the tag is a data artifact, not a deliberate ban. **Fix (Option B, user green-lit via "3.8 is next / you can do this, boss"):** (a) learnset filter widened from `c[0]==="9"` to the fork's real rule `3 ≤ learnedGen ≤ dex.gen` + `realMoves` (moves carry forward; 0 orphan move ids); (b) the extractor now allows a **real base-form species** (`num`+`exists`+`!placeholderFor`+`!forme`) blocked *only* by the "Past" tag — mega/gmax/CAP/LGPE/Legends stay rejected (scoped + verified against the original DB); (c) **D33 fix:** `format.default_level` was `rt.maxLevel` (BSS→100, wrong) → now the battle level `bl` (BSS→50), matching `test_format_levels_match_fork` exactly (my rebuild had clobbered it; extractor and D33 test are now in sync). **Result:** DB rebuilt — `form_learnset` 65872→121420, `format_form_move` 201350→327548, BSS forms 946→1238; `build_prior('machamp','gen9bss3v3')` = 210 candidates (was crashing). **Test reconciliation:** `test_no_prior_raises_for_absent_species` now targets a CAP (**colossoil** — genuinely nonstandard: no sets AND no legal moves) instead of abra (which now legitimately gets a legal-move fallback prior); the never-empty guard is unchanged and still meaningful. `.gitignore` gains `legality/legality.db.*` (protects the 27MB backup from accidental commit). **Gate:** Step-1 24/24, belief 3.1–3.8 all green, full suite green (see 3.8 bench). | user: "i dont want bugs left in the code, so lets investigate" + 3.7/3.8 milestone green-light (asleep 2026-11-16) | 2026-11-16 |
| D48 | **Post-milestone bug-hunt (user: "review your code and … continue on bug fixes") — the residual-vs-move classification bug found & fixed + 2 more findings.** (a) **THE BUG (belief-state accuracy, the user's core stress point — "make unknown gamestates known"):** residual damage lines (burn/poison/recoil/drain/confusion ticks) were classified as MOVE damage and pinned to the last `|move|` line (`_attribute_move_source`); the tick's delta is far outside the true candidate's window, so `damage_likelihood` FLOORED THE CORRECT CANDIDATE to ~EPS — collapsing a narrowed posterior to uniform after a single burn tick. **Fix (fork-verified protocol fact, live probe `scratch/probe_residual.js` 10/10 + static fork source):** a pure MOVE HIT logs exactly two payloads (`|-damage|slot|hp/max`); every residual/ability/item/recoil/drain line adds a bracketed context tag as the 3rd payload — burn `… brn|[from] brn`, recoil `|[from] Recoil`, confusion `|[from] confusion`, powder `|[silent]`, drain `|[from] drain`, weather `|[from] Sandstorm`, leechseed `|[silent]`. **The one open question is now resolved (the `!name` edge):** the `spreadDamage` default branch (`sim/battle.ts:2159-2178`) logs NO 3rd payload only when `effectType==='Move'` (a real hit) OR `!name`; but `fullname = data.fullname || name` (`sim/dex-data.ts:125`) and EVERY real damage-source effect carries a name — brn/psn/tox (Conditions), recoil→'Recoil'/drain→'Drain' (`sim/dex-conditions.ts:700-707`), weather, and the generic getByID `else` sets `name:id` — so the name-less `!name` path is unreachable for real damage (only `EMPTY_CONDITION` id='' qualifies, and it carries none). Hence a non-move damage line ALWAYS carries a bracketed 3rd payload — the signal is airtight. The adapter now classifies `-damage`/`-heal` on that 3rd payload: `source_kind="residual"` + NO move attribution (`_CONTEXT_TAG` = `^\[(from|silent|of|zeffect)\]`); the belief's existing 3.4 gate (`source_kind != "move" ⇒ facts only`) then makes residuals update the HP/maxhp FACT with ZERO set evidence — exactly the 3.4-6 contract, now actually true. (b) **Finding 2 (latent, fork-verified):** `facts.last_hit` (the reflector input for Counter/Mirror Coat/Metal Burst via `lastHitDamage`) was overwritten by residual ticks; the fork sets its counter volatile / `attackedBy` ONLY from real move hits (`gotAttacked`, `battle-actions.ts:1051` — a burn tick never feeds a reflector) → `last_hit` is now updated for move-source damage only. (c) **Finding 3 (doc bug):** the `IGNORED_TAGS` comment claimed `-immune/-resisted/-supereffective/-crit/-ohko` were "promoted to events in 3.4" — they are NOT (and don't need to be: TE is inside the window from the candidate's own typing, crits are inside the crit window the likelihood accepts, OHKO is the calc's guaranteed-KO formula consumed via the KO line D40); comment corrected so a future reader doesn't chase a phantom event. **Verified NOT bugs (evidence in the §3a D48 addendum):** crits, OHKO, multi-hit (incl. the 1.5× boosted hit + Skill Link via candidate ability), TE lines, hazards (`field` kind), KO-by-residual, `%/100` dedupe. **Tests:** +3 regressions (adapter classification `test_step3_1.py`; end-to-end posterior-collapse `test_step3_4.py::test_residual_burn_tick_does_not_disturb_posterior`; `test_step3_4.py::test_residual_tick_does_not_clobber_last_hit`). **Gate:** belief **84/84**; full gate **306/306**. | user: "still asleep. i want you to review your code and look at the project in the grand scheme … compile an open questions review for me later and continue on bug fixes" (2026-11-16) | 2026-11-16 |
| D49 | **P0/P1/P2 bug-fix wave + combat context + T5 state wiring — CLOSED (the "if the calculator presents a wrong state, it ruins the pipeline" class the user flagged).** (a) **P0 (belief-state corruption):** `_damage_evidence` passed reflector moves to the calc as bare strings; `_normalise_move` dropped `lastHitDamage`; the calc returned `immune/[0,0]`; `damage_likelihood` → 0.0 for EVERY candidate; `apply_evidence` floored all to EPS → **an already-narrowed posterior collapsed to uniform on the first Counter/Mirror Coat**. Fixed with the generic `_move_with_state(move, facts)` pipe: ALL tracked state (`lastHitDamage`, `hitThisTurn`, `lastMoveFailed`, `stockpileLayers`) attached on every move; the calc reads what it needs; **zero hardcoded move lists in Python** (the Node calc is the semantic authority per D37; Python is the data pipe). Regression `test_reflector_preserves_narrowed_posterior` (narrow → Counter → still narrow). (b) **P1:** stale comment corrected (amended into the P0 commit). (c) **P2 + combat context:** the adapter dropped most field effects (Trick Room → `Unknown`, Reflect/Spikes/etc → silently `None`); now `-sidestart/-sideend` (reflect, lightscreen, auroraveil, stealthrock, stickyweb, safeguard, spikes, toxicspikes) and `-fieldstart/-fieldactivate/-fieldend` (trickroom, wonderroom, terrains, mist, tailwind) map to a generic `FieldEffectChanged` → `BeliefState.field` (fork protocol verified in `client/src/battle-text-parser.ts` + `sim/side.ts:413`). `Facts` gained T5 per-turn state with **two-level reset** (`turn_reset()` on E.Turn vs full `reset()` on a new mon; `last_hit` persists — reflector memory). **`BeliefState.get_context()`** = the single belief-state snapshot the AI trains on (turn / fainted / active / field + per-mon hp_frac, statuses, boosts, moves, last_hit, hit_this_turn, last_move_failed, stockpile, posterior top-K, confidence). (d) **T5 wiring (`fd0dccf`):** the two remaining never-set facts are now set — `hit_this_turn` in `_apply_damage` (move-source, `delta > 0`, mirroring the fork's avalanche gate `attackedBy.some(p => p.source === target && p.damage > 0 && p.thisTurn)`, moves.ts:912-915; the fork's 0-damage entries exist but never qualify, so a resisted/immune hit must not set it) and `last_move_failed` in the `MoveFailed` dispatch branch (fork gate `moveLastTurnResult === false`, moves.ts:18050/19186 — stompingtantrum/temperflare; both sides; own_facts created on demand like the other fact handlers). Both cleared on E.Turn, matching the fork's end-of-turn `thisTurn=false` reset (battle.ts:1720-1727). +3 regressions (set / state-pipe / 0-damage / turn-reset). **Known approximations (documented, all rare-in-BSS):** `hit_this_turn` does not track WHICH mon hit (per-source gate — over-report only if the hitter switches out same turn); **`last_move_failed` is unreachable from this fork's real logs** (the fork never emits `movefail` — verified dist/sim + client parser; the observable signal is `|move|` + `|-fail|<user>`, currently ignored — needs an adapter mapping to fire in live play); **`stockpile_layers` still never increments** (needs a `MoveUsed(move="stockpile")` hook — same T5 class). **Gate:** test_step3_4 **19/19**; full gate **310/310**. Commits `b057fd6` (P0+P1), `9917ab0` (P2+context), `fd0dccf` (T5 wiring). | user: "please proceed with the combat context we discussed so we have everything handled in one belief state handler that can be forwarded to the ai agent" + "yeah go ahead with the small fix … p1 in a bit afterwards" (2026-11-16) | 2026-11-16 |
| D50 | **The two T5 move gaps closed (user: "can you fix the two move bugs while youre on it please?") + a fork divergence found while doing so.** (a) **`last_move_failed` was unreachable from this fork's REAL logs** (the fork never emits `movefail` — the previous session's adapter comment assumed a `|-fail|<user>` line but never verified it). Fork-verified protocol (2026-11-16): an ANNOUNCED move that fails (miss/immunity/Protect/blocked) logs `|move|...` then **`|-fail|<user>`** (`battle-actions.ts:656-659` — no move name in the payload; `attrLastMove('[still]')`); an ABORTED attempt (par/slp/frz/flinch/recharge — `conditions.ts` `onBeforeMove` returns false; nopp — `battle-actions.ts:340`) logs **`|cant|<user>|<reason>`** BEFORE any `|move|` line (the `BeforeMove` check at `runMove` line 316 precedes the announce at `addMove` line 521). The fork sets `moveThisTurnResult=false` on ALL of these paths (runMove 319/331/342 + `hitResult===false`), which is exactly the next-turn gate for Stomping Tantrum / Temper Flare (`moveLastTurnResult === false`, moves.ts:18050/19186) — note a MISS also doubles, and a Protect block does NOT (`NOT_FAIL`/`true` paths). Adapter: `|-fail|` → `MoveFailed` (move name recovered from the user's own `|move|` announce — a move executes atomically, so the fail line always immediately follows it); `|cant|` → `MoveFailed` ONLY for the six move-attempt reasons (nopp carries the move name in the line; other cant reasons — trp, ability blocks — stay silent). (b) **`stockpile_layers` now increments** on `MoveUsed(move="stockpile")` with the fork cap of 3 (`moves.ts:17956-17966` `onTry`/`onRestart` return false at 3 → the 4th use fails, adding nothing); documented edge: a blocked sub-cap Stockpile leaves one phantom layer (rollback can't tell a capped failure from a real one — rare-in-BSS, accepted). (c) **FORK DIVERGENCE FIXED (self-review catch):** Stockpile layers are a fork **volatile with no duration** (`data/conditions.ts` — `noCopy`, `onStart`/`onRestart`, no expiry) — they PERSIST ACROSS TURNS and are cleared only by switch-out (`clearVolatiles`) or a new mon. The previous session's `turn_reset()` wiped them every turn, which would have corrupted a multi-turn Stockpile→Spit Up sequence (3 turns of Stockpile then Spit Up = BP 300 in the fork; our calc would have seen 1 layer). Now: `turn_reset()` keeps them; `reset()` (new mon) and the same-mon switch-in branch (switch-out cleared the volatile) clear them. **Tests:** +3 — adapter protocol (`test_step3_1.py::test_adapter_move_failure_lines_to_movefailed`, incl. the silent non-move-failure cant case), stockpile cap/persist/switch-clear + pipe (`test_step3_4.py::test_stockpile_layers_track_cap_and_switch_clear`), end-to-end real-log MoveFailed (`test_step3_4.py::test_last_move_failed_from_real_log_lines`). **Gate:** full **313/313** (incl. the 3.7 fuzz on real seeded battles — misses in real logs now map to MoveFailed instead of being dropped). Commit `85729b9`. | user: "we need to fix the calc first before doing anything … can you fix the two move bugs while youre on it please?" (2026-11-16) | 2026-11-16 |
| D51 | **Pre-Step-4 belief/observation bug wave — 3 real bugs fixed + the Step 4 stage documented (user: "continue setting up the stage for part 4 … we are not continuing before all bugs are fixed … i have not yet decided on the node/endpoint decision").** (a) **Spikes/Toxic Spikes layer count was DROPPED** (`state.py._apply_field_effect` stored `1 if active else 0` — a bool — so 2-3 Spikes read as "present"; hazard damage under-reported). Fork-verified: the fork emits ONE `|-sidestart|<side>|Spikes` line PER LAYER with NO count in the line (`dist/data/moves.js:17536-17543`: `onSideStart` sets `layers=1`+emit; `onRestart` does `layers++` + the SAME emit; caps 3 Spikes / 2 Toxic Spikes) — the count IS the number of start lines. Now counted (capped) in the belief and reset to 0 on `|-sideend|`; the observation carries the true 0-3 / 0-2 count. (b) **`-prepare` was surfacing as the 3.7 "benign prepare" Unknowns** — it's the client charge/aim animation line (`sim/SIM-PROTOCOL.md:594` `|-prepare|ATTACKER|MOVE`); the real move is the `|move|` line we DO map, so no belief evidence → moved to `IGNORED_TAGS` (the `Unknown` bug-catcher stays live for genuinely-unmapped tags). (c) **`confidence` could render `-0.0`** (a float entropy a hair over 1.0) → clamped to [0,1]. **Tests:** +3 — `test_spikes_and_toxicspikes_track_layer_count`, `test_confidence_clamped_to_unit_interval`, `test_adapter_prepare_line_is_ignored`. **Gate:** full **316/316** (was 313). **Still open (accepted, rare-in-BSS):** `hit_this_turn` per-source approximation (§3a-3). **Step-4 stage laid in §3b (NOT built); the calc endpoint (batch vs persistent worker) is STILL AWAITING USER DECISION — no calc work started.** | user: "please continue setting up the stage for part 4 / we are not continuing before all bugs are fixed / i have not yet decided on the node/endpoint decision" (2026-11-16) | 2026-11-16 |
| D52 | **Batch calc endpoint (Option 2 — user: "let's implement the batch for now, its the safer route") + Protect observation + complete hazard observation — the final pre-Step-4 wave (resolves the D51 open calc-endpoint decision).** (a) **Batch endpoint:** `damage/node_calc.js` gained a shared per-request `dispatchOne(r)`; `main()` branches on input shape — a JSON ARRAY in → an ARRAY of results IN ORDER with per-item isolation (one bad item → `{error}` for that item, the rest still return) — while the SINGLE-request contract stays BYTE-identical (object in → object out; bad → `{error}` + exit 1) so all 313 prior damage tests pass untouched. `damage/calc.py` gained `_node_batch(requests)` (ONE Node spawn for N calcs) + public `calc_damage_many(items)`, where each item is the same arg dict a single `calc_damage` takes (`attacker`/`defender`/`move` + optional `field`/`fmt`/`gen`). **Why:** the forward-eval path is "my N moves × enemy team" = 24+ calcs per decision; batching avoids N Node spawns — probe `scratch/batch_probe.py`: 24 calcs **266 ms batch vs 5984 ms individual (~22.5×), identical results**. The persistent-worker option is kept documented (deferred), not built. (b) **Protect observation (user: "we should put a 'used-protect-last-turn' observation since that is huge for doubles"):** `Facts.used_protect: bool`, set on `MoveUsed` when the move is in `_PROTECT_FAMILY` (`protect`/`detect`/`kingsshield`/`spikyshield`/`guardsplit`; moves.js:14010/3579/17556/7991 + kingsshield) — detected by MOVE NAME because the fork's volatile differs per move, so the move name is the one stable signal we already have. Per-turn (the fork clears the protect volatile at end of turn), cleared in `turn_reset()`. Surfaced on both own and opponent mon blocks in `get_context()`. Documented minor edges (both rare-in-BSS, low impact): a FAILED protect-family use still sets it; Guard Split marks the user, not the ally it shields. (c) **Complete hazard observation (user: "if the hazards are not correctly observed as you said, please add it" — §3b B1):** Toxic Spikes (0-2) + Safeguard were tracked in `self.field` (D51 wired the layer count) but NOT surfaced in `get_context()` — now both are (toxicspikes per-side count + safeguard per-side). Hazard state is now complete for the Step-4 observation. **Tests:** +3 — `test_calc_damage_many_batch_matches_individual` (batch==individual, order, per-item isolation), `test_used_protect_set_and_turn_reset`, `test_hazard_observation_includes_toxicspikes_and_safeguard`. **Gate:** full **319/319** (was 316). **Still open (PROPOSED, NOT implemented):** the `hit_this_turn` per-source approximation — see §3b E, awaiting the user's implement-vs-defer call. | user: "lets implement the batch for now, its the safer route" + "we should put a 'used-protect-last-turn' observation since that is huge for doubles" + "if the hazards are not correctly observed as you said, please add it" (2026-11-16) | 2026-11-16 |
| D53 | **hit_this_turn per-source fix implemented (user: "yeah for sure fix the issue now before we get issues when testing this in another format").** Replaced `Facts.hit_this_turn: bool` ("was hit this turn by ANYONE") with `attacked_by_this_turn: set[(side,slot)]` (which opponent mon(s) hit this one this turn, delta>0) — matching the fork's PER-SOURCE gate `attackedBy.some(p => p.source === target && p.damage > 0 && p.thisTurn)` (Avalanche moves.js:926-934; Revenge moves.js:15085-15093 — the SAME gate, so both are covered). Set in `_apply_damage` (records `ev.source_side`/`ev.source_slot`; guarded on delta>0 + a real, distinct source); cleared in `turn_reset()` (and `reset()` via it). `_move_with_state(move, facts, defender)` now takes the target's `(side,slot)` and computes `hitThisTurn = defender in attacked_by_this_turn`; the two `_damage_evidence` call sites both pass `defender=(dmg.target_side, dmg.target_slot)` (the mon being hit). **The calc is UNCHANGED** (still reads the plain `hitThisTurn` bool — the per-source decision is made in the Python data pipe, keeping @smogon/calc the semantic authority per D37). Observation dim UNCHANGED: `get_context()` still exposes the model-facing bool `hit_this_turn = len(attacked_by_this_turn) > 0`. The `defender=None` fallback (only reachable if a caller omits the target) keeps the old was-hit-at-all over-report — documented. **Tests:** `test_hit_this_turn_is_per_source` (p1a hits me → Avalanche on p1a doubles, Avalanche on p1b does NOT) + updated `test_hit_this_turn_set_and_turn_reset` / `test_zero_damage_hit_does_not_set_hit_this_turn`. **Gate:** full **320/320** (was 319). **Also surfaced (NOT fixed, for user review — §3a-5/6):** (a) Revenge (gen 4-7) has this same per-source gate but is NOT in the calc's `STATE_BP_MOVES` set — a gen ≤7 gap; (b) the fork's reflectors are per-turn (Counter/Mirror Coat `duration:1` volatile; Metal Burst/Comeuppance `thisTurn`-required) while our `last_hit` persists across turns — a potential stale-hit over-report. | user: "yeah for sure fix the issue now before we get issues when testing this in another format" + "what else caught your 'eye'?" (2026-11-16) | 2026-11-16 |
| D54 | **Reflector stale-hit gate fixed (user: "investigate the counter bug and fix it" + "the logic to be there and it just not getting extra damage in singles").** Root cause: the fork's reflectors are PER-TURN — Counter/Mirror Coat recreate their volatile each turn (`beforeTurnCallback`, `duration: 1`) and `onTry` FAILS (`slot === null`) unless the user was hit by a damaging move THIS turn (moves.js:3025-3060); Metal Burst/Comeuppance require `thisTurn` (moves.js:2739/11721). Our `Facts.last_hit` (the magnitude) intentionally persists across turns as reflector memory, but `_move_with_state` fed it to the calc UNCONDITIONALLY → a mon hit N-1 turns ago that used a reflector reflected 2× that STALE hit, while the fork would FAIL it (0). **Fix (minimal, no new state, no calc change):** `_move_with_state` now attaches `lastHitDamage` only when `attacked_by_this_turn` is non-empty — the SAME per-turn, move-source, damage>0 signal the D53 Avalanche gate uses. Consequence: at decision time in singles (no hit yet this turn) the gate is CLOSED → the calc's `lastHit==null → 0` path fires (base power, "no extra damage", exactly the user's preference); once the hit lands in the log (the common same-turn path) the gate OPENS → 2× as before. The reflection LOGIC is untouched (still 2× for Counter/Mirror Coat, 1.5× for Metal Burst/Comeuppance in node_calc.js). **Calc UNCHANGED** (the Python data pipe makes the per-turn decision, keeping @smogon/calc the semantic authority per D37). **Tests:** +1 `test_reflector_stale_hit_not_reflected` (hit this turn → gate OPEN reflects; after E.Turn the per-turn signal clears but last_hit memory persists → gate CLOSED, no lastHitDamage); the existing `test_reflector_preserves_narrowed_posterior` (same-turn path) still passes. **Gate: 321/321** (was 320). **Surfaced (OPEN, §3a-7, awaiting user):** the fork reflects only a MATCHING category (Counter=physical-only, Mirror Coat=special-only, moves.js `onDamagingHit` `getCategory(move)` check) but our calc applies 2× regardless of the hit's category and `last_hit` records no category — a category-mismatch over-report (belief-accuracy only); fixing it requires tracking `last_hit_category` + updating the special+Counter test scenario. | user: "investigate the counter bug and fix it" + "i'd prefer the logic to be there and it just not getting extra damage in singles" | 2026-11-16 |
| D55 | **Reflector CATEGORY gate fixed (user: "yes, fix that calculator error next please").** The fork reflects only a MATCHING category — Counter/Metal Burst record a PHYSICAL hit, Mirror Coat/Comeuppance a SPECIAL one (moves.js `onDamagingHit`: `getCategory(move) === "Physical"` / `"Special"`); a mismatch means the reflector FAILS (0). Our calc applied `2× lastHit` to Counter/Mirror Coat **regardless of category** (node_calc.js:486-493) and `last_hit` recorded no category → a special-hit + Counter (or physical-hit + Mirror Coat) over-reported 2× where the fork fails at 0. **Fix (Python data pipe; calc UNCHANGED — D37 keeps @smogon/calc the semantic authority):** (a) `Facts.last_hit_category: Optional[str]` set alongside `last_hit` in `_apply_damage` via `_move_category(ev.move)` (DB lookup, `None` on unknown) — persisted as reflector memory like `last_hit` (excluded from `turn_reset()`, cleared in `reset()`); (b) module const `_REFLECTOR_REQUIRES = {counter/metalburst: "Physical", mirrorcoat/comeuppance: "Special"}` (fork-verified); (c) `_move_with_state` attaches `lastHitDamage` **only to a reflector** (`required is not None`) **and only when** `facts.last_hit_category == required` — a mismatch falls to the calc's `lastHit==null → 0` path (base power, no over-report) and non-reflector moves never carry the field (calc reads it only for the 4 reflectors — no non-reflector consumer in node_calc.js). **Tests:** `test_reflector_preserves_narrowed_posterior` re-paired SPECIAL hit → **Mirror Coat** (fork-accurate; was the fork-inaccurate special+Counter); `test_reflector_stale_hit_not_reflected` (D54) re-paired to a **PHYSICAL** hit (Aqua Jet) → **Counter** so it reflects, + `last_hit_category == "Physical"` assertion; +1 `test_reflector_category_gate` (special opens Mirror Coat/Comeuppance but NOT Counter/Metal Burst; physical the reverse; non-reflector tackle never carries `lastHitDamage`). **Gate: 322/322** (was 321). | user: "yes, fix that calculator error next please" | 2026-11-16 |
| D56 | **Per-gen move data now lives in the DB (Option A — user-approved).** Move metadata is mechanics-scoped and changes per generation (Dragon Claw SPECIAL gen 1-3 → PHYSICAL gen 4+; Protect priority 2→3→4; champions Slash 70→80). **A1 (implemented):** `move_gen` gains `category` + `priority` (schema v5, `db.py` bootstrap ALTER); `legality/extract_move_gen.js` (NEW, fork-sourced via `Dex.forFormat`, bakes in mod overrides) emits an **overrides-only** row per **engine key** (0=champions, 1-9=gen; `move_gen.gen` = engine key, NOT raw gen) where a move's type/category/power/accuracy/pp/priority differs from the gen-9 base — **1602 rows** (k0=134 k1=391 k2=380 k3=374 k4=141 k5=110 k6=36 k7=24 k8=12 k9=0). `lookup.get_move(name, gen)` merges the override onto the gen-9 base (backwards-compatible: empty `move_gen` ⇒ unchanged); effect flags side_condition/boosts/drain/recoil stay gen-9 on the base row (the calc models those per-gen — **A2 deferral**). Both belief consumers are now **gen-aware** via `formats.engine_key(fmt)`: the D55 reflector **category** gate (`_move_category(move, gen)`) + the **speed-order priority** evidence (`_pair_speed`). **A2 (deferred, not started):** per-gen effect flags, only when a consumer needs them. **Gate: 327/327** (was 322). | user: "the move data should live in the database and change per generation (in gens 1-3 all ice moves are special for example) … so we go for A" | 2026-11-16 |
| D57 | **A2: per-gen EFFECT FLAGS now live in the DB (the "catch more bugs" follow-up to A1 — user: "lets do A2 now and try to catch more bugs before step 4").** A probe across all 10 engine keys found **36 real per-gen flag diffs** (all in gen 1-5 + champions; gen 6-8 clean). Biggest bug class: **gen-1 Light Screen / Reflect / Mist were stat-boost moves, NOT side conditions** — so the gen-9 `is_screen` flag misclassified them; plus boost-magnitude shifts (Tail Glow +3→+2, Amnesia gen-1 boosts BOTH SpA+SpD, minimize/stringshot/sweetscent ±1 not ±2) and recoil shifts (Double Edge 1/3→1/4, Struggle 1/2→none in gen 9). **Implemented (same A1 pattern):** `move_gen` gains side_condition/side_status/boosts/drain/recoil (schema v6, `db.py` bootstrap ALTER); `extract_move_gen.js` now emits all 11 columns (sorted-key `boosts` for an order-stable diff compare); `lookup.get_move` merges the 5 flags (is_hazard/is_screen/is_tailwind recomputed from the merged side_condition). **1628 override rows** (was 1602, +26 flag-only diffs). Fork-verified. **A1 (6 data cols) + A2 (5 effect flags) both DONE** — move data is fully per-gen. **Gate: 331/331** (was 327). | user: "lets do A2 now and try to catch more bugs before step 4" | 2026-11-16 |

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
- **Champions engine (key 0) has its own 359-species whitelist** (library `CHAMPIONS_LIST`,
  `node_modules/@smogon/calc/dist/data/species.js`): the table copies gen-9 (SV) entries
  for the listed species ONLY — **chansey and hitmonlee are NOT champions-legal**, and
  `Pokemon(0, 'chansey')` crashed opaquely inside the library (`calcStat` reading `hp` of
  undefined). `node_calc.js buildPokemon` now checks the engine's species table first and
  throws a clean `pokemon "X" is not legal in the gen-N engine` error (D39). The library
  also hard-codes **level 50, no IVs, no EVs** for gen 0 — consistent with D33.
- **Ingest pitfall (fixed):** payload rows containing only PK columns produced an empty
  `DO UPDATE SET` ("incomplete input") — `api.ingest` now falls back to `DO NOTHING`.
- Legacy `Hidden Power <type>` (16 variants) all carry server id `hiddenpower` →
  extractor skips name/id mismatches; the set ingester maps variant names → `hiddenpower`.
- **Battle-log protocol (verified in this fork, 2026-11-16 — don't re-derive):**
  - `|move|` line = `|move|source_slot|MoveName|target_slot` (battle-actions.js
    `addMove("move", pokemon, movename, target)`) — move is payload **[1]**, target [2].
  - **The `movefail` tag does NOT exist in this fork** (absent from `dist/sim` and the
    client parser). A failed move logs the `|move|` announce line first, then
    `|-fail|<user>` (NO move name) — so the move name is always in the announce, and
    "no damage line followed the move" is the observable failure signal for the
    belief evidence engine. The `|movefail|slot|MoveName|` handler in
    `belief/log_events.py` is kept for upstream-format logs only.
  - Each **switch-in is logged TWICE back-to-back** (`hp/maxhp` + a `% /100` dup), same
    as HP lines; the adapter dedupes identical consecutive switch lines (the 3.1 smoke
    test's ">= 2 p2 switch-ins" previously only passed via this dup bug).
  - KO line = `|-damage|slot|0 fnt` (no denominator), emitted twice; true damage ≥ maxhp
    (D40). Ability is NOT revealed at switch-in — contextually via
    `[from] ability: Name` + `[of] holder_slot` (holder = the `[of]` slot, full entry
    D43 lands at 3.8).

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
