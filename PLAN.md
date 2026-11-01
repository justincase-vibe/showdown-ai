# Showdown AI — Plan & Roadmap

> Single source of truth for **what we're building, the decisions we've made, and where
> we are**. `README.md` is the intro/how‑to‑run; this file is the plan. Update the status
> section as work lands so we never lose the thread.

**Goal:** a strong Pokemon Showdown RL bot. One shared policy across **Gen 6–9 × {OU, UU, RU}**
(+ random battles). It must clearly **beat random and dumb rules**, play with real strategy,
and stay **robust to unorthodox play**. Single machine (16 cores, RTX 5090). Throttling stays **ON**
(bounded, not unlimited) — we want ~10 battles/min, not "no throttle."

Reference: Chen & Lin, *"Gotta Train 'Em All"* (Stanford CS230, 2018) — PPO + hand‑crafted
features + self‑play to avoid "learned helplessness."
<https://cs230.stanford.edu/projects_fall_2018/reports/12447633.pdf>

---

## 1. Two‑part system

- **Part 1 — PLAYING (current focus).** Given a fixed legal team of 6, decide each turn:
  move / switch / (gen‑dependent) Mega·Z·Dynamax·Tera. Reward **+1 win / −1 loss**.
  One shared policy spans Gen 6–9 in a unified **963‑action move‑identity** space
  (every move in the gen6‑9 union + 6 switches + 3 gimmick transforms), so a single mask
  makes one policy exact across generations. Actions are a move *by identity* (not a slot),
  so training and live play can never drift.
- **Part 2 — DRAFTING (later).** A value head `V(team) → predicted win‑rate`. Reward the
  drafter for *accurate judgment* (predicted vs. actual outcome), so it learns to evaluate
  team strength without playing; later use `V(team)` to *select* the strongest team.
  This is the paper's "learned helplessness" fix.

## 2. Training recipe (DECIDED)

**Imitation (expert replays) → Self‑play PPO → Eval.** "Slow is fine" — this is a
data + time game, not a tricks game.

1. **Richer state** (`obs.py` upgrade) — unblocks everything.
2. **Datamine top‑tier replays** — tiers **RU→OU** across Gen 6–9, **plus random battles**.
   Move‑by‑move, for imitation *and* the eval benchmark. **Owner: assistant. Status: NEXT / later.**
3. **Imitation phase** — supervised (behavioral cloning): train the policy to predict the
   expert's move/switch given the state. Fast + cheap; the bot plays "decently" from day one
   instead of wasting early hours as pure random.
4. **Self‑play PPO** — **~10k games** as the *first checkpoint* (≈ ~17 h at 10/min; trained
   play ends sooner, so faster). It should clearly beat random + dumb rules. A *strong* bot
   likely needs more than 10k → treat 10k as iteration 1 and keep going.
5. **Eval** — (a) **position benchmark:** feed real expert positions to the value net and check
   its "who's winning" call vs. the actual outcome; (b) **win‑rate** vs. random and vs. dumb rules.

## 3. State representation (DECIDED design)

Base: **892‑dim** hand‑crafted vector (v3), generation‑agnostic, robust to hidden info
(safe accessors; missing → 0). Replaces the old 328‑dim vector. `OBS_DIM` is the single
exported constant — `env.py` / `policy.py` / `train.py` / `imitate.py` pick it up automatically,
so the policy **rebuilt at 892** with no other changes.

**Implemented (v2):**
- **Neutral identity** — species / ability / item as *process‑stable multi‑probe hashes*
  (`hashlib`, so all 4 worker processes agree — Python's salted `hash()` would **not**). Identity
  only, **not** a hard‑coded moveset prior; the net learns tendencies from play and adapts.
- **Base‑stat profile** (tank vs. sweeper vs. wall) — a gen‑stable species signal (was unused).
- **Richer move semantics** per revealed move: power / accuracy / priority / drain / heal / recoil
  / inflicts‑status / inflicts‑volatile (+ PP for own mon). *(Also fixed a latent bug: move
  category was compared as a `MoveCategory` enum against plain strings, so it was silently
  all‑zero — now compared to real enum members.)*
- **Battle cues**: must‑recharge / protect / tera / dancing + curated volatile‑effect one‑hot
  (confusion, encore, disable, destiny‑bond, curse, disguise, gastro‑acid, taunt, trick, traps).
- **Bench identity** per slot (own: fainted/hp/status/identity; opp: fainted/revealed/identity/type).
- **Speed proxies** (true move‑order speed isn't exposed by the API): "opp has a revealed
  priority move" + "relative base speed (opp − own)".
- **Move identity** (v3): each revealed move gets a 4‑dim positional code
  `[x, sin(πx), cos(πx), sin(2πx)]`, `x = move_id/954`, tying it to its slot in the 963‑action
  space — so the policy maps "this move I see" → "that action" without a hard‑coded per‑gen table.
- **Moveset‑shape prior** (v3): per active mon, a 26‑dim "what could this species do" vector
  (type coverage, phys/spec bias, hazard/setup/recovery/priority caps) from its natural (L/E)
  moves across gens 6–9 — the "Garchomp lead ⇒ Stealth Rock + setup" reasoning signal (gen‑agnostic).
- **Learned stat inference** (still true): we don't read EVs; the value net infers bulk from
  observed damage + the HP signal.

**KEY DECISION — no hard‑coded opponent prior.** We do **not** feed "this species usually runs
X" as a known‑moves feature. Rationale: a static prior is **exploited by unorthodox play**; a
*learned* opponent model adapts to the tail. Smogon sets are still used for **our own team
generation** and (optionally) imitation — never as an *opponent* assumption in the state.

## 4. Opponent inference (DECIDED)

The bot **infers** the opponent from **observable cues**, not from a label — exactly how a
human reads a Garchomp lead:

- **Stealth Rock** → chip damage when our mon switches in, or the SR move revealed.
- **Focus Sash** → the mon survived a hit that should have KO'd it (dropped to 1 HP).
- **Choice Scarf** → moves first *despite* lower speed + move‑locked (same move every turn).

PPO + value net + **GAE** learns these multi‑turn abstractions. Self‑play teaches the
inference *skill*; the *base rates* (which sets are common) come from our team distribution +
the imitation data. Conditions: the **cues must be in the state**, and the **data must contain
the structure**. Both are covered above.

## 5. Formats (DECIDED)

- **Gen 6–9 × {OU, UU, RU} = 12 formats**, all confirmed present + tier‑enforcing on the server:
  `gen6ou gen6uu gen6ru gen7ou gen7uu gen7ru gen8ou gen8uu gen8ru gen9ou gen9uu gen9ru`.
- **+ random battles** (separate track) — keeps the bot from overfitting to "random‑battle
  movesets are normal."
- **Singles (1v1).** Doubles (2v2) is a separate future track (positioning, spread moves).
- **Opponent modes:** `live` self‑play (default — a **shared** policy plays both sides, so both transitions train the model), `frozen` (P1‑only, lagging copy), `random` (P1‑only baseline).
- **Weights:** default **equal** across active formats (bias later).

## 6. Throughput & logistics (MEASURED + DECIDED)

Measured battles/min at **16 concurrent battles** (random‑vs‑random = the *longest* case):

| Client | battles/min | note |
|---|---|---|
| 1 process, serial | ~7.9 | current `train.py`; waits serialize |
| **4 processes** | **~10.4** | **target** |
| 8 processes | ~12.4 | server/CPU ceiling ~12–13/min |

- **Ceiling is server + 16 cores** (16 Node battle processes competing for 16 cores), not the
  client. 20/min would need >16 concurrent battles → core thrash → the "fry the CPU" we avoid.
- **TARGET: ~10/min = 4 worker processes / 16 concurrent battles** — bounded, ≤ core count,
  CPU‑safe. 10k games ≈ **~17 h** (conservative; trained play is shorter → faster).
- **Server config:** `config/config.js` → `simulator: 16` (concurrent‑battle governor),
  `nothrottle = false` (**throttling ON** = bounded), `noguestsecurity = true`;
  `server/monitor.ts` gates raised to `countPrepBattle` 240/3min, `countConcurrentBattle` 20.
  Server runs from `dist/` → `node build` after editing `server/*.ts`.
- **GPU:** battle generation is **CPU‑only** → keep GPU idle during data collection. GPU is for
  **PPO updates only**, started **on explicit GO and only when the GPU is free**.
- **DECIDED — server scaling:** run **ONE** Showdown server and scale **concurrency** (worker
  processes / concurrent envs, capped at 16 battles) — **not** multiple servers. Each extra
  server multiplies the throttle budget → N× CPU load → thrash. Throttle stays **ON**
  (bounded load is a feature); throughput ceiling ≈ 12–13/min is accepted. GPU stays idle
  until explicit GO. *Goal: a clean, bounded workflow that never fries the hardware.*

## 7. Data tasks

- **Team pools** (`bot/team_pools.json`): **✅ RESOLVED — all 12 formats** (gen6‑9 × OU/UU/RU) now have **coherent, server‑verified teams** (the real 6 that go together), built from **Smogon's council sample teams** (via the crob.at JSON mirror) + a few community teams to top up, each team validated by the local server's own TeamValidator (`pokemon-showdown validate-team`) so every stored team is guaranteed runnable.
  `teams.py` now has **`StrongTeamBuilder`** (one coherent team — the self‑play default) alongside **`RandomTeamBuilder`** (weak random‑bag baseline, kept for the synergy‑contrast tool). Self‑play uses `--team-mode strong` (default); `--team-mode random` for the weak baseline.
  - **⚠️ Count reality (data ceiling, not a bug):** Smogon only publishes a handful of curated sample teams per tier, so the "20‑25 strong teams/format" target is met for **gen6ou (19), gen7ou (23), gen8ou (21), gen9ou (25)** but only partially for the smaller tiers: **gen6ru (3), gen6uu (4), gen7ru (5), gen7uu (12), gen8ru (15), gen8uu (14), gen9ru (6), gen9uu (4)**. All are legal + coherent; there simply aren't more Smogon samples to draw from.
  - **Decision needed:** accept fewer‑per‑format (self‑play still works fine — it does today), **or** top up the short tiers with synthesized legal teams (type/role‑balanced, weaker) to reach 20. **Default kept: real teams only (no synthesis)** — quality over quantity, per the "strong teams" goal.
- **Datamine top‑tier replays (RU→OU + random battles)** — for imitation + eval. **Owner:
  assistant. NEXT (later).** Avenues: Showdown ladder/battle API, community replay dumps.
- **Smogon sets** — for our own team gen + (optional) strong‑set imitation. Not an opponent prior.
- **Two reference docs (later, not yet):** (1) per‑mon moveset tables + legality by gen;
  (2) pre‑built teams by gen scraped from Smogon.

## 8. Pipeline status

- ✅ Server runs locally (port 8000); throttle **ON** (bounded); simulator = 16.
- ✅ Random‑vs‑random battles start/finish reliably (gen6ou / gen7ou / gen7randombattle).
- ✅ **"Agent is not challenging" timeout — FIXED (was a red‑herring "torch bug").** Real cause: duplicate‑species random bags got rejected by the server (Species Clause). `load_verified_pool` now dedupes by species; `train.py --smoke` PASSES (CPU); see Morning‑decision #7.
- ✅ Team pools server‑verified (gen6ou, gen7ou).
- ✅ Env + PPO scaffolding (`env.py`, `obs.py`, `policy.py`, `train.py`); `np.int64` action fix.
- ✅ Throughput measured; concurrency + design decisions locked (see §2–§6).
- ✅ **(1) `obs.py` upgrade → 808‑dim** (verified: internal‑consistent block sizes; live battles
  pass on gen6ou / gen7ou / gen7randombattle; policy rebuilds at 808; full CPU PPO smoke test
  passes: masked act → store → one update).
- ✅ **(2) datamine COMPLETE.** Fetcher + parser built and validated across all 15 formats.
  **30,000 battles · 2,094,742 decisions · 317.6 MB.** Gens 6–9 × {OU, UU, RU} + Random.
  Gimmicks: Mega (gen6/7), Z (gen7), Dynamax (gen8), Tera (gen9) all detected.
  **NEXT: (3) imitation phase → (4) concurrent self‑play loop (4‑proc) → (5) eval.**
- ✅ **(3) move‑identity action‑space redesign** (Steps 1–5 of the plan). Action = a move
  **by identity** (name), not a slot — deletes the train/live reveal‑order mismatch by
  construction. Global **963‑space**: 954 moves (gen6‑9 union, Z/G‑Max as distinct moves)
  + 6 switches + 3 transforms (mega/dynamax/tera). `action_space.py` (new, single source of
  truth), `parse.py` emits 0..962, `obs.py` adds move‑identity + moveset‑shape (808→892),
  `env.py` builds the 963‑mask + **translates 963→poke‑env before `step`**, `imitate.py`
  sized to 963. **Live‑verified:** gen7ou + gen6ou battles run to completion (reward ±1);
  offline: obs finite, imitate actions in [0,962] (move/switch/transform all present).
  **v1 caveat:** Z/G‑Max are identities in the space but the live v1 mask issues *plain* moves
  (Z‑use + "gimmick‑with‑a‑specific‑move" are v2); mega/tera/dynamax are single transform
  actions (paired move chosen at step‑time). **NEXT: (4) imitation precompute (GPU, on GO)
  → (5) concurrent self‑play → (6) eval.**
- ✅ **(4a) smoke‑test fix: Showdown protocol edge states handled correctly.**
  (a) **Wait requests** (server `"wait"` flag, e.g. while the opponent must choose its next
  mon): all‑zero mask ⇔ nothing to decide. `policy.act` now **raises** on an empty mask
  (was: silent full‑space sampling → the "illegal action 264/386/818" smoke crash). Loops in
  `train.py` detect the empty mask and **pass** (TraineeEnv sends the default −2 order),
  recording the step via `policy.store_pass()` (no decision stored; terminal reward still
  attributed if the battle ends on such a step; `value()` helper added).
  (b) **Stale‑mask re‑validation:** our mask is a snapshot of the last request; the server's
  valid orders can be narrower by send‑time (observed: Taunt → moves‑only, Whirlwind forced
  switch → "[still]"). `TraineeEnv._valid_action` checks the conversion against
  `battle.valid_orders` (via `SinglesEnv.action_to_order(strict=True)`) and re‑picks the first
  masked action that IS valid (`re_picks` counter), so **stored action == executed action** —
  poke‑env's strict=False random‑move fallback (which corrupted ~5% of steps in one battle)
  is no longer reached. **Verified:** `train.py --smoke` PASSED ×2 (0 invalid‑order warnings,
  re_picks exercised) and `verify_env.py --formats gen7ou,gen6ou --battles 1` ALL PASSED.
  **Known gap (flag):** opponent‑side transitions are not stored (frozen/live opponents act
  but never learn) — self‑play currently trains only the trainee side.
- ✅ **(5) Imitation pre-training COMPLETE** (CPU — GPU reserved for PPO per the scaling
  decision). **1,045,840 samples × 892 dims** across all 15 formats (30k battles); 97.1% of
  samples carry a known species; 91.3% resolve a non-empty legal‑move prior. Trained 10
  epochs (batch 256, lr 3e‑4, **ε=0.05** legal‑move prior, weighted CE with (elo/1200)² ×
  format weight). ~90 s/epoch on CPU.
  **Eval (`eval_imitation.py`)** — metric FIXED to the **masked top‑1 accuracy** (apply the
  per‑sample legal mask → argmax → compare to the expert action) = what the model does *live*.
  The old "top‑1 is legal / illegal‑argmax" numbers measured the **raw unmasked** argmax — a
  red herring for a hard‑masked actor (its illegal‑move logits are unconstrained by the loss).
  Correct numbers (200k samples, 2026‑10‑29): **new hard‑mask actor 85.2%** (moves 76.0% / switch
  100% / transform 100%) vs **old label‑smooth‑only 75.6%** (72.7% / 81.1% / 38.7%). The raw
  numbers even *invert* the truth (new 3.6% vs old 56.1% raw) → **never trust the raw metric**.
  **A/B:** ε=0.05 is the default (ε=0.10 was worse on every metric).
  **Memory fix:** `precompute_dataset` now pre‑allocates the states buffer once and fills it
  in place (was list‑append + `np.stack` = ~7.5 GB double‑allocation, which OOMs on this
  8‑GB‑free box). Pre‑allocation peaks ~4 GB. Dataset = 105 MB npz.
  **Models:** `bot/data/imitation_actor.pt` (**hard‑mask + ε=0.05 — THE one to use / warm‑start**),
  `bot/data/imitation_actor_labelsmooth_only.pt` (pre‑hard‑mask backup, for A/B),
  `bot/data/imitation_actor_eps010.pt` (A/B artifact).
  **NEXT: (6) concurrent self‑play PPO (warm‑started from the imitation actor) → (7) eval.**
- ✅ **(6) P2‑perspective self‑play bug — FIXED + verified.** The P2 opponent was acting on a
  **P1‑perspective observation** (own/opponent features swapped): `env.py` always rebuilt the
  observation from P1's view, so P2 was *systematically handicapped* → **20/0 self‑play** instead of
  ~50/50 (the earlier "20/0 warm‑start" number was INVALID). Fix: `embed_battle` now resolves the
  side per agent (`side = "p2" if battle is self.battle2 else "p1"`) and passes it through
  `state_from_log` (`_perspective(side)` / `_build_observation(side)` / `StateReconstructor(side=)`).
  The P1 path (imitation) is unchanged — regression‑checked. **Verified:** live wiring test
  (a1→p1, a2→p2 both served in a real battle); object‑level own↔opponent swap **90/90**;
  20‑battle self‑play **55/45** random‑init (not 20/0) and **80% warm‑start** (expected — the
  imitation actor is P1‑biased, so as P2 it is out‑of‑distribution = flag #4 below, not a bug);
  `imitate --smoke` + `eval_imitation` regression pass. Committed `bf1ff67`.
- ✅ **(7) CPU self-play validation + two real fixes (2026‑10‑29, this commit).**
  (a) **`train.py` `buffer_size()` update‑gate bug — root cause of the earlier “timeouts”.** The old
  `buffer_size()` counted **live in‑progress episodes** (`policy._episodes`) toward the “should I update”
  gate, but `update()` only drains the **merged** buffer (`policy._buffer`). Result: the gate fired
  prematurely → the inner collect loop `break`ed after the **first env** (starving the others) →
  `update()` spun on an empty merged buffer (a long stream of empty `upstats:{}` records in the first
  run) → battles never finished. **Fix:** `buffer_size()` now returns only `len(policy._buffer["act"])`
  (or 0). **Verified:** clean end‑to‑end CPU run, `EXIT=0`, battles completed, updates real.
  (b) **Critic‑calibration progress metrics (`policy.py`).** The 50/50 self‑play win‑rate is useless by
  design (same net both sides), so the *built‑in* progress signal is the **value net’s calibration**:
  at episode flush we record the critic’s opening read (`ep.val[0]`) vs the realized terminal reward
  (`ep.rew[-1]`), and `update()` reports (then resets) `calib_rmse` / `calib_mae` / `calib_bias` /
  `calib_corr` / `calib_sign_hit` (did the bot read the game’s *direction* right — the most
  interpretable single number) / `calib_confidence` (mean |opening read|) / `calib_n`.
  **Monitor these in `bot/checkpoints/stats.jsonl`, not the raw win‑rate.**
  (c) **CPU‑vs‑GPU throughput expectation:** `live` mode is **~1 battle/min on CPU** (2× `policy.act()`
  CPU inferences per step + observation reconstruction). This is **expected, not a defect** — GPU will
  be much faster. Do not interpret slow CPU validation as a bug.
- 🚩 **Flagged for review (decisions I made / gaps you should weigh in on):**
  1. **✅ RESOLVED — ~23% illegal‑argmax was a *raw‑metric artifact*, now crushed.** Hard legal‑mask
     added to the imitation loss (default ON) + corrected masked eval. The prior
     (5% mass spread over legal moves) can't fully override the net's learned bias toward
     globally‑common moves (e.g. U‑turn). Harmless in *live* play — `env.py` hard‑masks to
     the true moveset — but it means the imitation actor alone is not fully prior‑compliant.
     **Option (your call):** hard‑mask the imitation loss to the legal set (drop illegal
     logits to −inf) instead of / in addition to label smoothing → illegal rate → ~0.
  2. **✅ RESOLVED — Learnset/forme data gap (two‑part fix).**
     (a) **Forme‑union:** `action_space.py` now unions a species' own learnset key with its base‑species
     key (e.g. `Blastoise‑Mega`→`Blastoise`, `Charizard‑Mega‑X`→`Charizard`, `Rotom‑Heat`→`Rotom`,
     `Arceus‑Fire`→`Arceus`). Root cause: poke‑env's `learnset.json` is incomplete *both ways* (some
     forme keys exist but list no moves; others are absent), so "exact‑match‑first" still dropped moves.
     This cut the replay "gap" (moves the data used that the legal set excluded) **18.94% → 12.21%**
     (200k sample); masked eval held at **85.2%** (no regression).
     (b) **Residual 12.2%** = genuine poke‑env base‑species incompleteness (Ferrothorn/Leech Seed,
     Smeargle/Baton Pass, Blissey/Seismic Toss only‑gen3, Vaporeon/Heal Bell no‑gen8, Mamoswine/Icicle
     Crash only‑gen6). Covered by the `dataset_moves[species]` union in the imitation prior — do **not**
     hack `legal_moves` to ignore gen to "close" this (would break its gen contract).
  3. **✅ RESOLVED — Self‑play PPO device = GPU** (your choice). `train.py --device cuda`.
     The scaling decision said GPU‑on‑GO, but the GPU is ~90% used
     by agent inference and RAM is only 8 GB free. A 10k‑game self‑play on **CPU** is the
     safe default (bounded, matches the "never fry the hardware" rule) but is **slow**
     (PPO updates on CPU). **Your call: CPU (safe/slow) vs GPU (fast, on explicit GO)?**
  4. **✅ RESOLVED — both‑sides self‑play (shared policy).** The opponent was P1‑only before
     (it acted but never learned). `--opponent live` (now the **default**) shares one policy across both sides:
     `env.py` exposes `last_opp` (P2 state + decision + zero‑sum reward) and `train.py:_store_opponent_side`
     stores P2's transition under a distinct env id (`e.id + 1_000_000`) → a separate trajectory for
     per‑episode GAE. **Verified:** real gen7ou battle → P1 46 + P2 50 decisions → buffer 105 rows →
     PPO update clean; real `run()` loop 4 battles / 4 updates green. `random`/`frozen` remain P1‑only
     baselines. **Caveat:** with a shared policy the win‑rate hovers ~50/50 (same net both sides + first‑move
     edge) — the learning signal is the win/lose reward on both sides, not the win‑rate.

  5. **📌 NOTED (Decision 5) — OBS_DIM 248/244, keep as‑is.** The opponent‑active block is
     **N_MOVES (4) dims shorter** than the own‑active block because the opponent's per‑move PP slot is
     hidden (measured: own_active **248**, opp_active **244**, so the team block starts at **492**).
     It is **self‑consistent** (training and live use identical `embed_battle`), so it does **not** break
     the current model. **Decision: keep as‑is** for now (no re‑encode — that would invalidate the
     trained imitation warm‑start). The correct *eventual* fix is a **symmetric 248/248 re‑encode**
     (re‑build `OBS_DIM` + retrain the imitation actor) — do later, after the team‑pool blocker.

## 8b. Unattended self-play continuation — agent-run 2026‑10‑30 (decisions + uncertainties)

> You asked me to (1) snapshot the net, (2) run **1,000 more** self-play battles, and (3) log my
> decision uncertainties while you're away. **State as of this note:** run-1 checkpoint preserved;
> the two-seat eval is finishing (verdict → `bot/_eval2.log`); the 1k continuation is launching via
> the watchdog. Below is what I chose, why, and the things I could **not** decide and want your call on.

**Done + verified this session:**
- ✅ Snapshotted run-1 → `bot/checkpoints/ppo_run1_1k.pt` + `stats_run1_1k.jsonl` (byte-identical). The continuation overwrites `ppo.pt`; run-1 is safe.
- ✅ Extracted + **load-verified** the bare actor → `bot/data/actor_run1_1k.pt` (255,043 params; loads into a fresh `MaskedPPO.actor`). `bot/extract_actor.py`.
- ✅ Error-checked the two-seat eval end-to-end (smoke test, no crash) — full checklist in this session's chat.

**Decisions I made unattended (flag if you disagree):**
- **D1 — Eval *before* self-play (sequencing).** I do **not** run the eval and the 1k self-play concurrently: both hammer the same server + CPU, the eval would be starved, and its verdict (real skill vs P1-seat-bias) decides whether "1k more battles" is even the right move. So: let the eval finish → capture verdict → launch continuation.
- **D2 — Continue from run-1's ACTOR, not the imitation actor.** `--init-actor bot/data/actor_run1_1k.pt` = a true continuation (builds on the 1k run). **Caveat:** `--init-actor` carries the **actor only** (critic + optimizer reset to init) — the established warm-start pattern (run-1 + the morning CPU session both did this). The actor is what plays; the weak critic (sign_hit ~0.5) re-warms from reward over ~100–200 battles. Want full actor+critic+opt continuity? → see **U3** (a ~5-line `train.py` enhancement).
- **D3 — GPU.** Matches run-1 (proven over 8.65 h *with* agent inference active). The watchdog pops an `ERROR` box on any OOM/crash, so a failure is caught, not silent. Prefer CPU-safe? Relaunch with `--device cpu` (one flag).
- **D4 — Watchdog launcher.** `bot/run_gpu_watch.py` for the unattended run: server preflight + 15‑min stall watchdog + `DONE/ERROR/STALLED` pop-ups + beeps.
- **D5 — Same hyperparams as run-1** (lr 1e‑4, buffer 4096, minibatch 128, 16 envs, `--opponent live`). Changing several knobs at once while you're away is the recklessness we agreed to avoid.

**Uncertainties I could NOT resolve (need your call when you're back):**
- **U1 — THE BIG ONE: is the ~94% self-play real skill or a P1‑seat‑bias artifact?** The two‑seat eval answers it. §8 flag #5 already notes the obs is **asymmetric (248 own / 244 opp)**; the *eventual* fix is a **symmetric 248/248 re‑encode**. **If the eval shows P1 strong / P2 ~50% → it's the bias artifact**, and 1,000 more self-play battles would *reinforce* the bias rather than build balanced skill. I proceeded with the continuation as you asked, **but if the verdict confirms strong bias, the higher-value move is likely the symmetric re‑encode first** (rebuild `OBS_DIM` + retrain the imitation actor), then re-run self-play. ← **please read the eval verdict first.**
- **U2 — Critic is still a coin flip** (`sign_hit` 0.5 in run-1). More self-play *may* fix it; track `calib_*` in `stats.jsonl` (want sign_hit/corr ↑, rmse ↓, confidence ↑).
- **U3 — Actor-only warm-start (D2).** Want full actor+critic+opt continuity? Small enhancement — say the word.
- **U4 — After this 1k:** apply the §9b verdict rule (win-rate vs random ≥ 60% **and** sign_hit ≥ 0.60 → scale to 2k+; else more self-play **or** the bias fix).

**What to check when you're back (in this order):**
1. `bot/_eval2.log` — the two-seat verdict (real skill vs seat bias) → decides **U1**.
2. `bot/run_gpu.log` + any desktop pop-up — did the continuation finish / error / stall?
3. `bot/checkpoints/stats.jsonl` — `calib_*` trend for the continuation (**U2**).
4. Checkpoints: `ppo.pt` = continuation (run-2); `ppo_run1_1k.pt` = preserved run-1.

## 8c. 🛑 BLOCKER — bot is switch‑degenerate & worse‑than‑random (found 2026‑10‑30, pre‑run)

> **I did NOT launch the requested 1,000‑battle continuation.** The two‑seat eval + step‑level
> diagnostics show the current bot (and its imitation warm‑start) **cannot attack** — it is
> *worse than a random opponent*. Running 1k more self‑play on this foundation would just reinforce
> the defect. Below is the evidence chain, the localized root cause, and the fix I recommend.
> **Please review before any more training.**

**Measured facts (reproducible via the `bot/diag_*.py` scripts):**
1. **Eval vs random (2‑seat):** `ppo.pt` wins **0/21 (P1)** and **1/21 (P2)** vs a random‑move
   opponent. A weak bot sits ~50/50 vs random; **0/21 means it systematically loses** — *worse than
   random*, not "untrained".
2. **Replays (bot P1, gen6ou 1400–1406):** the bot **fainted all 6 mons in every battle** (P2 fainted
   0–4). Switch‑heavy: one battle had 50 switch lines vs 16 move lines.
3. **Live logits (masked, `diag_action.py`):** on real observations the actor puts **move≈0.001 /
   switch≈0.999** probability mass — for **BOTH** `ppo.pt` AND `imitation_actor.pt`.
4. **Effective vocabulary (`diag_collapse.py`):** argmax lands on **only 10 of 963** actions —
   3 transforms (mega/tera) + 6 switches + 1 move (0.1%). It **cannot pick a real move**.
5. **Training data is fine (`diag_data.py`):** labels are **61% move / 38% switch / 1% transform** —
   experts attack; the data is not the problem.
6. **Collapse is from TRAINING, not init (`diag_init.py`):** fresh random‑init spread = **0.30**
   (flat); trained actor spread = **15.64**, **move logits −5.7**, **switch logits +4.2**.
   Training *drove* every move down and every switch up.

**Root cause (localized to the imitation training; leading mechanism = the hard‑mask):**
`imitate.py` hard‑masks the 954 non‑legal moves to −1e4 each sample. Since any *specific* move is
legal for only a few species, nearly every move is "illegal" on most samples, so the −1e4 penalty
drags the **entire move band down**, while the always‑relevant switch/transform band stays high.
Net: a move‑suppressed, switch‑dominated actor. The earlier "85% masked top‑1" (§8 flag #1) clearly
did **not** transfer to a live‑capable actor — live, masked, it is ~100% switch. (I have not yet
proven the exact gradient path end‑to‑end; that's the first thing to verify in the fix.)

**Why self‑play didn't save it:** run‑1 was warm‑started from this broken actor (`--init-actor`),
so self‑play inherited a switch‑spammer on *both* sides; the ~94% P1 self‑play number is a
first‑mover/degenerate artifact, **not** skill (confirmed by the 0/21 vs‑random floor).

**Recommended fix (needs your GO — it's a training‑design change, so I did not unilaterally run it):**
1. Fix the imitation mask/loss so it does **not** over‑suppress the active's legal moves — e.g. keep
   the active's legal moves **+ switches** in‑set, use a **softer** penalty than −1e4, or drop the
   hard‑mask and rely on the legal‑prior target. **Retrain + re‑eval.**
2. **Gate before resuming self‑play:** a healthy actor must (a) reach high **true** masked top‑1
   (re‑measure with the real per‑state legal set), (b) be **move‑capable** (live masked mass: moves
   clearly > 0), and (c) **beat random > 50%** in a 2‑seat eval. Only then self‑play.
3. Diagnostics added (keep them): `bot/diag_action.py`, `bot/diag_data.py`, `bot/diag_trainobs.py`,
   `bot/diag_collapse.py`, `bot/diag_init.py`.

**State on disk (clean):** run‑1 checkpoint preserved (`ppo_run1_1k.pt`); actor extracted
(`actor_run1_1k.pt`); eval verdicts in `bot/_eval2.log` + `bot/_eval_baseline.log`; server still
LIVE. Nothing is broken beyond the pre‑existing actor defect. If you *do* want the 1k run anyway,
the launcher is ready: `python bot/run_gpu_watch.py --init-actor bot/data/actor_run1_1k.pt`
(though it will very likely reproduce the same switch‑degenerate result).

## 8d. 🩹 FIXED — illegal‑switch re‑picks + “forced into an unintended mon” (2026‑10‑31)

> You asked to *fix the bug, not punish it*, and to let the agent keep choosing (re‑ask until
> legal, with a slight penalty). Done + verified on CPU.

**Root cause (measured, not guessed):** instrumenting the real decision path showed **every**
re‑pick was **P2, switch band, `nV=1` (server allows 1 move) vs `nM=7` (mask allowed all 6 switches
+ a move)**. That is the **whole‑bench‑fainted** state: `available_switches` is **empty**, but
`get_action_mask` treated “empty” as *allow all six slots* (the `if avail_sw and bs not in
avail_sw: continue` guard fell through). The policy then picked a switch the server correctly
rejected → the “re‑pick / forced into an unintended mon” you saw. **A mask bug, not a policy problem.**

**Fix (`bot/env.py`):**
1. `get_action_mask` now un‑masks a switch **only if that mon is in `available_switches`** (an
   empty set ⇒ no switches), exactly matching the server / poke‑env → **`illegal_picks` → 0**.
2. Unified decision path `_decide()` (replaces the old `_valid_action` + `_first_valid_switch` +
   `_opp_decision` split): the **agent always picks**; if the pick is illegal we **mask it out and
   re‑ask** (the agent keeps choosing) until it is legal, then a **slight capped penalty**
   (0.05/pick, max 0.20 — can’t flip a win/loss). This also **honors the agent’s own switch choice**
   in forced‑switch states (U‑turn/Volt Switch/Roar/Dragon Tail/Whirlwind) instead of slot 0.
3. `step(actor1, actor2)` now returns a consistent `(action, logprob)` per side
   (`last_self` / `last_opp`) so PPO’s ratio stays correct; both sides train (`--opponent live`).

**Reward (terminal core unchanged + one small nudge):** Win **+1.0** / Loss **−1.0** / Forfeit
**−1.5** (backstop — a forfeit is *not* a win; should now be ~0) / illegal‑pick **−0.05·count**
(capped 0.20).

**Verified (CPU, server 8000):** `bot/test_mask_fix.py` (empty‑bench ⇒ no switches; force‑switch ⇒
only bench, no moves); `bot/test_decide.py` (illegal→re‑ask→legal; legal→0; exhaust→safe fallback;
empty→wait); `train.py` 3‑battle live self‑play (**0 forfeits, 0 illegal_picks**, PPO update OK);
and `train.py --smoke` (**re_picks=0**). `train.py` / `quick_eval.py` updated to the new `step()`
signature; `launcher.py` spawns them as subprocesses (unaffected).

## 8e. \ud83d\udea9 FIXED \u2014 spurious \"forfeit\" replays at run\u2011end (found 2026\u201111\u201101, root\u2011caused + fixed)

> You found forfeits in the last\u20115 replays of the 20\u2011battle run (battles 1852/1855/1856/1857),
> incl. a **turn\u20111** one, and said *\"we did not fix the bugs \u2026 please please fix it.\"* It was
> **not** a U\u2011turn bug, **not** a policy choice, and **not** the `reset()`/`put_nowait` race the
> earlier handoff was chasing. It was a **teardown** bug.

**Root cause (proven, not hypothesized):** `poke_env`'s `PokeEnv.close()` **forfeits every battle
that isn't finished yet** (`if self.battle1 and not self.battle1.finished \u2192 ForfeitBattleOrder`),
then blocks on `_challenge_task.result()`, which only unblocks *because* that forfeit ended the
battle. The run loop hits `max_battles` with each env holding a **fresh in\u2011progress battle** (re\u2011armed
the instant one finishes); the `finally:` block calls `close()` on all envs \u2192 each in\u2011progress battle
gets a `/forfeit` \u2192 the server writes \"X forfeited / win Y\" into those replays. **One spurious forfeit
per env, at run end, every time.**

**Evidence that nails it:** the run log shows `forfeits: 0` on *every* stats line and the `safe_reset`
guard *always* read `finished=True` (so `reset()` was never the culprit); the 4 forfeit replays are
exactly the battles **not** followed by a `safe_reset ENTER` (still live when `done. battles=20` fired);
1856/1857 forfeit on **turn 1\u20132 with no faints** (brand\u2011new in\u2011progress battles); the \"U\u2011turn\" in 1855
was just the last action before the run ended.

**Fix (root cause, keeps poke\u2011env; files `bot/env.py` + `bot/train.py`, not yet committed):**
1. **`ShowdownEnv.close()` override** \u2014 **never sends a forfeit**: no `/forfeit`, no blocking wait, no
   `reset_battles()` raise; it abandons a live battle cleanly (drop refs + drain queues). A forfeit is
   not an action the 963 policy can take, so this is correct by design \u2014 the *guarantee*.
2. **`train.py` drain** \u2014 after `max_battles`, play out any in\u2011progress battle to a natural win/loss
   (bounded by step + time caps), store its transitions, then one final PPO update captures them. Also
   stop *starting* a new battle once the target is reached. This is the *proper* graceful end.

**Verified (real battles, server 8000, CPU):** 12\u2011battle / 4\u2011env / mixed gen7ou+gen7randombattle /
`--opponent live` \u2192 log shows `draining 3 in\u2011progress battle(s)\u2026 env 1 finished after 12 / env 2 after 7
/ env 3 after 94 steps`, clean exit; **new replays 1922\u20131936 all have `forfeited`=0** and each ends
`|win|\u2026`. `train.py --smoke` PASSES. Only residual log noise is a **benign** poke\u2011env parse warning on
**Pursuit** (hits the switching mon) \u2014 does not affect correctness.

**If it regresses:** look at (a) any code path that calls `close()` while a battle is live, or (b) a new
`ForfeitBattleOrder` source. Re\u2011check = run a small multi\u2011env batch and grep the newest replays for
`forfeited` (expect 0). Note: current reward is terminal\u2011only **+1 win / \u22121 loss** (no forfeit reward) \u2014
a forfeit is a *harness artifact*, not a reward to tune. *(\u00a78d's \"Forfeit \u22121.5\" line is stale; the code
is +1/\u22121 only.)*

## 9. Open decisions

- ~~Replay sources / player tier / exact format coverage~~ — **DONE:** HolidayOugi HF dataset,
  top‑2000 by ELO per format, 15 formats (gen6–9 × OU/UU/RU + 3 random).
- **Iteration depth:** 10k is checkpoint 1; decide how far to go after first eval.
- **Format weights:** equal now, or bias toward specific tiers/gens.

### ✅ Morning decisions — RESOLVED (2026‑10‑29)
1. **Self‑play device** — **GPU** (your choice). I prepared the device‑aware preflight
   (`train.py --smoke --device cuda` — prints GPU name + free/total GB; the default `--smoke` is CPU
   so it never contends) + the full launch command in **§9b**. I **do not** launch it — **you do**,
   when the GPU is free.
2. **Overnight 10k self‑play** — ⏳ **you run it** (agent does not launch it; I'm using the GPU for inference). Command in my report.
3. **~23% illegal‑argmax** — ✅ **FIXED + verified.** Hard legal‑mask in the imitation loss (default ON) + a corrected *masked* eval metric. The scary "71.7% illegal" was a **raw‑unmasked‑argmax artifact**, NOT a live defect (the legal mask is always applied at live time). Correct **masked top‑1: new 85.2% vs old 75.6%**; switch/transform imitation jumped to 100%.
4. **Learnset gap** (e.g. U‑turn/Garchomp) — ✅ **FIXED.** Legal set = `legal_moves(species, gen) ∪ dataset_moves[species]` (moves the data actually used), + always‑allow the replay action. Applied in `imitate.py` (train) and `eval_imitation.py`.
5. **OBS_DIM 248/244** — ✅ **keep as‑is + documented** (`_ACTIVE_OWN`=248 / `_ACTIVE_OPP`=244 in `obs.py`). Not re‑encoded (would invalidate the trained weights).
6. **Both‑sides self‑play** — **implemented** (`--opponent live` is now the **default**): one shared policy plays P1 and P2, both transitions train the model (zero‑sum). `random`/`frozen` remain P1‑only baselines. See §8 flag #4.
7. **"torch breaks poke_env" — ❌ RED HERRING, root cause found + fixed.** The `asyncio.TimeoutError: Agent is not challenging` at `env.reset()` was **never a torch/asyncio/threading bug.** It is a **misleading poke‑env message** that really means *the server rejected the team at battle start* (the challenge never completes). **Proven:** DEBUG log showed `|popup|Your team was rejected … You have more than one Landorus`; `import torch + create policy + a *legal* team (StrongTeamBuilder)` resets fine in ~4 s. The real bug: the **`flat` team pools contain duplicate species** (they are the concatenation of every mon from every coherent team), so **`RandomTeamBuilder`** (which `train.py --smoke` and the `--team-mode random` baseline use) could sample two of the same species → **Species Clause violation** → team bounced → timeout. **Fix:** `teams.load_verified_pool` now **dedupes by species** (`_dedupe_by_species`), so any random bag is a legal team; all 12 formats re‑verified legal via the server oracle. Also added a defensive `RuntimeError` in `TraineeEnv.reset` that surfaces "team rejected, not a networking bug" instead of the raw timeout. `train.py --smoke` now **PASSES** (CPU). **The torch/poke_env "separate processes" rewrite is NOT needed.**
8. **Short CPU self‑play session — ✅ RAN + 2 real bugs fixed.** Wide CPU session
   (gen7ou / gen9ou / gen6uu, `--opponent live`, warm‑started from the imitation actor, `--lr 1e‑4` —
   a warm net is more sensitive; 3e‑4 risks destabilizing) completed **clean** (`EXIT=0`, battles done,
   real updates with `calib_n>0`, no spin/crash) after fixing the **`buffer_size()` update‑gate bug**
   (env‑starvation + update‑spin — see §8 entry 7) and adding the **`calib_*` critic‑calibration
   metrics** (the real progress signal in both‑sides self‑play). **First checkpoint for the GPU run:
   1k battles, then eval** — throughput is server‑throttled, so the wide mix costs no extra
   wall‑clock. `frozen` is the fallback opponent if `live` looks unstable.

## 9b. Launch the GPU self‑play (you run it, when the GPU is free)

> Throttle stays **ON** (~10 battles/min, ≤16 concurrent). Battle generation is **CPU‑only**;
> the GPU is used only for PPO updates. Do not run while this agent is using the GPU for inference.
> **All 12 formats now have team pools** (§7 RESOLVED) — the **wide mix is the plan** (start‑wide
> decision: basic skills first — type matchups, sweep setup, switching — and it costs no extra
> wall‑clock since throughput is server‑throttled). CPU `live` ≈ 1 battle/min is expected; GPU much faster.

**Preflight (device‑aware; default `--smoke` is CPU and safe, `--device cuda` sanity‑checks the GPU):**
```
python bot/train.py --smoke --device cuda
```

**Easiest — the watchdog launcher (error pop-up on failure/stall, summary pop-up on done, beeps):**
```
python bot/run_gpu_watch.py
```
It runs the 1k‑battle command below, logs to `bot/run_gpu.log`, kills + alerts if no stats
progress for 15 min (`--stall-min`), and pops `ERROR` / `STALLED` / `DONE` windows. `--no-popup`
prints instead. Extra args override the defaults, e.g. `python bot/run_gpu_watch.py --no-popup
--formats gen7ou --num-envs 1 --max-battles 1 --device cpu` for a quick test.

**Both‑sides self‑play, wide mix, warm‑started — FIRST CHECKPOINT = 1k battles, then eval**
(`--lr 1e‑4` because the warm net is more sensitive; `--opponent frozen` is the fallback if `live` looks unstable):
```
python bot/train.py --formats gen6ou,gen6ru,gen6uu,gen7ou,gen7ru,gen7uu,gen8ou,gen8ru,gen8uu,gen9ou,gen9ru,gen9uu \
  --num-envs 16 --opponent live --buffer-size 4096 --minibatch 128 --ppo-epochs 4 --lr 1e-4 --max-battles 1000 --device cuda --init-actor bot/data/imitation_actor.pt
```
**Monitor `calib_*` in `bot/checkpoints/stats.jsonl` (per‑update) as the progress signal:**
`calib_sign_hit` / `calib_corr` improving, `calib_rmse` falling, AND `calib_confidence` rising =
genuine learning. Raw win‑rate hovers ~50/50 **by design** (shared policy on both sides) — it is
**not** the metric to watch.

**At the 1k checkpoint — EVAL with `bot/eval_selfplay.py`** (the §2.5 "Eval" step, now runnable).
It measures the two decision‑relevant numbers on a FRESH session (clean critic, GPU‑safe):
1. **Critic calibration** — does the value net's "who's winning" read match the real outcome?
   Reports `sign_hit` (want → 1.0), `corr`, `rmse`/`bias` (want → 0), `confidence` (should rise with skill).
   This is the honest "did it learn to read games" number — the same `calib_*` metric, but off‑training.
2. **Win‑rate vs a RANDOM opponent** — the hard floor (self‑play ~50/50 is useless by design; vs random it
   must be clearly >50%).
```
python bot/eval_selfplay.py --model bot/checkpoints/ppo.pt --formats gen6ou,gen7ou,gen9ou --battles 20
```
**Verdict rule:** win‑rate vs random ≥ 60% AND `sign_hit` ≥ 0.60 → scale up (2k+ battles). MARGINAL → more
self‑play. NOTE: the warm‑start `imitation_actor.pt` has a **random‑init critic** → its calibration reads
~50/50 (expected baseline, not a bug); `ppo.pt` has the trained critic.

**Narrow starter variant** (3 gens, quick sanity slice before the full wide run):
```
python bot/train.py --formats gen7ou,gen9ou,gen6uu --num-envs 3 --opponent live --buffer-size 96 --minibatch 64 --ppo-epochs 3 --lr 1e-4 --max-battles 20 --device cuda --init-actor bot/data/imitation_actor.pt
```

---

## 10. Key files

| Path | Role |
|---|---|
| `bot/action_space.py` | **NEW** — global 963 move‑identity action space; legality, Z→base, mask, shape prior. |
| `bot/env.py` | `ShowdownEnv` + `TraineeEnv`; 963‑mask + 963→poke‑env translation; reward. |
| `bot/obs.py` | `embed_battle` → **892‑dim** vector (v3: move‑identity + moveset‑shape, per §3). |
| `bot/policy.py` | `MaskedPPO` (masked actor‑critic PPO + GAE). |
| `bot/train.py` | Self‑play loop (**→ concurrent 4‑proc per §6**). |
| `bot/teams.py` / `bot/teamgen.py` | Team pools + server‑verified pool builder. |
| `bot/team_pools.json` | Verified coherent team pools (all 12 formats: gen6–9 × OU/UU/RU). |
| `bot/verify_env.py` | Torch‑free pipeline check. |
| `bot/eval_imitation.py` | Imitation masked‑top‑1 eval (expert‑match). |
| `bot/eval_selfplay.py` | **NEW** — self‑play checkpoint eval: critic calibration (sign_hit/corr/rmse/confidence) + win‑rate vs random. The 1k‑checkpoint gate. |
| `check_battles.ps1` | **NEW** — GPU‑free "are battles running?" check (stats freshness + node count). |
| `bot/run_gpu_watch.py` | **NEW** — 1k‑battle launcher: server preflight, stall watchdog, OS error/done pop‑ups. |
| `start_selfplay.bat` | **NEW** — double‑click launcher (works from any folder). |
| `bot/datamine/fetch.py` | Download top‑ELO battles from HF (per format). |
| `bot/datamine/parse.py` | Parse Showdown logs → structured decisions (move‑identity, 0..962). |
| `bot/data/replays/*.jsonl` | Curated battle data (gen8ru: 2000 battles). |
| `pokemon-showdown/` | Local server (port 8000); `config/config.js` + `server/monitor.ts` gates. |
