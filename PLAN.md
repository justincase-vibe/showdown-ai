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
- **Opponent modes:** `frozen` self‑play (default), `random` (baseline), `live`.
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

## 7. Data tasks

- **Team pools** (`bot/team_pools.json`): have **gen6ou (20) + gen7ou (21)**, all server‑verified.
  **Build the other 10** via `teamgen.py` (server‑validator as legality oracle) or Smogon sets.
- **Datamine top‑tier replays (RU→OU + random battles)** — for imitation + eval. **Owner:
  assistant. NEXT (later).** Avenues: Showdown ladder/battle API, community replay dumps.
- **Smogon sets** — for our own team gen + (optional) strong‑set imitation. Not an opponent prior.
- **Two reference docs (later, not yet):** (1) per‑mon moveset tables + legality by gen;
  (2) pre‑built teams by gen scraped from Smogon.

## 8. Pipeline status

- ✅ Server runs locally (port 8000); throttle **ON** (bounded); simulator = 16.
- ✅ Random‑vs‑random battles start/finish reliably (gen6ou / gen7ou / gen7randombattle).
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
- ⬜ LATER: build other 10 team pools; Part 2 drafting value head; doubles; **search+RL
  (MCTS / PIMC)** as an amplifier on top of the learned policy+value.

## 9. Open decisions

- ~~Replay sources / player tier / exact format coverage~~ — **DONE:** HolidayOugi HF dataset,
  top‑2000 by ELO per format, 15 formats (gen6–9 × OU/UU/RU + 3 random).
- **Iteration depth:** 10k is checkpoint 1; decide how far to go after first eval.
- **Format weights:** equal now, or bias toward specific tiers/gens.

## 10. Key files

| Path | Role |
|---|---|
| `bot/action_space.py` | **NEW** — global 963 move‑identity action space; legality, Z→base, mask, shape prior. |
| `bot/env.py` | `ShowdownEnv` + `TraineeEnv`; 963‑mask + 963→poke‑env translation; reward. |
| `bot/obs.py` | `embed_battle` → **892‑dim** vector (v3: move‑identity + moveset‑shape, per §3). |
| `bot/policy.py` | `MaskedPPO` (masked actor‑critic PPO + GAE). |
| `bot/train.py` | Self‑play loop (**→ concurrent 4‑proc per §6**). |
| `bot/teams.py` / `bot/teamgen.py` | Team pools + server‑verified pool builder. |
| `bot/team_pools.json` | Verified legal pools (gen6ou, gen7ou → expand to 12). |
| `bot/verify_env.py` | Torch‑free pipeline check. |
| `bot/datamine/fetch.py` | Download top‑ELO battles from HF (per format). |
| `bot/datamine/parse.py` | Parse Showdown logs → structured decisions (move‑identity, 0..962). |
| `bot/data/replays/*.jsonl` | Curated battle data (gen8ru: 2000 battles). |
| `pokemon-showdown/` | Local server (port 8000); `config/config.js` + `server/monitor.ts` gates. |
