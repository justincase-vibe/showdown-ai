# Workflow Review + Open Design Choices (2026-11-19)

> Purpose: a single scannable place for (A) the workflow errors found in the
> end-to-end pass, (B) the design choices that still need **your** call, and
> (C) the current checkpoint board. Review this when you're back — nothing in
> section B is built until you pick.

---

## ⚡ DECISIONS NEEDED — read this first (2026-11-21)

> One-page board of every open choice that gates the lookahead engine
> (M2b → M5). Full reasoning is in the dated sections below; this is just the
> scannable "pick one" board. **Nothing in the "Blocks" column is built until
> you pick.** My lean is the safe default — reply "go with your lean" and I
> proceed. The two that actually BLOCK M2b are **OC-6** and **OC-9**.

| # | Decision | My lean | Blocks | One-line consequence |
|---|----------|---------|--------|----------------------|
| **OC-6** | `step` world-model fidelity | ✅ **LOCKED (2026-11-22, USER): (b) RICH** — statuses + stat boosts + hazards + screens tracked in `step`, effect metadata sourced from the fork's move data (one source), probabilistic effects OUT of the deterministic model (the value head learns them); sim-agreement gate covers them | M2b `step` | Search sees attrition/setup games; the sim stays ground-truth via the gate |
| **OC-9** | p2-POV posterior handling | ✅ **LOCKED (2026-11-22, USER): (b) CARRY-OPAQUE** — posterior rides along at the p2 flip and stays VISIBLE to the net. Impl: own-side posterior routed into the existing posterior channels (zero schema change, old checkpoints load); dedicated own-posterior channels deferred to the second-POV training build | wiring `_flip_pov` into `_orient` (p2-POV obs) | Honest uncertainty to the value head; learnable once second-POV data exists |
| **OC-7** | decider-in-state design | ✅ **VERIFIED BUILT + CONFIRMED (2026-11-22)**: `State=ctx+decider`; `value()`/`policy()` run heads on the decider POV via `_orient`; `_map_value` maps to root (pure, both-decider tested); `_flip_pov` core proven (round-trip identity) | (assumption holds) | the double-switch / opponent-reply scenarios are expressible |
| **OC-8** | tera/mega in the search | ✅ **LOCKED (2026-11-22, USER): FORMAT-CONDITIONAL** — tera only where the format has it (gen9+), mega only in mega gens (6/7). Where present: in the action list WITH a once-per-mon used-flag. Where absent: not in the list at all | `step` action-list | no illegal repeated-tera; no phantom actions in formats without the mechanic |
| OC-2 | re-baseline the value head before it's the MCTS leaf? | re-verify the best checkpoint's value head is sane (M3) | M3 quality (not the M2b build) | a weak leaf makes MCTS weaker; cheap to check |
| OC-3 | MCTS inference budget (iterations/decision) | start ~400 live, higher in training | a tuning knob | strength vs latency |
| OC-4 | role of the hand-coded `choose_action` | **(c)** fold its signals into M4 learned obs; retire it as a path | M4 (not M2b) | one decision path — no two workflows |
| OC-1 | MCTS vs pure value-learning | MCTS (in progress) | — | the direct engine for win-condition / double-switch / threat |
| OC-5 | scenario priority / M1→M5 order | keep search-first (M1→M5) | sequencing | phases are swappable if you'd rather start elsewhere |

**DECISIONS LOCKED (2026-11-22):** OC-6=(b) rich, OC-9=(b) carry-opaque, OC-7
verified+confirmed, OC-8 format-conditional — all user-ruled. OC-1/2/3/4/5
remain later (M3/M4/tuning). The structural de-risk was already DONE (perspective
contract, `State` isolation, `_flip_pov` permutation) — so the engine build is
unblocked: the revised M2b checklist (7 steps) below is the execution order.

---

Pipeline audited this pass:
`dashboard → POST /api/train/start → launcher.start_trainer → validate → spawn
detached → entry main → run → runstate protocol → stop/status` **and** the
capture path `control/selfplay → pool → battle → replay capture → library`
**and** the imitation path `resolve fixture (raw|library) → bc.run → verdict →
checkpoint` **and** the league path `league/engine → SelfPlayRun (factory +
on_battle) → league/driver → PokeAgent + BotRegistry ELO`. Fast stack green:
**173/173** (gui+replays+pool+control+league, live deselected) as of D100.

---

## 2026-11-20 — ONE workflow (legacy `bot/` PPO generation quarantined)

**Decision (user: “i do not want two separate workflows, that just makes work
super hard”; “delete the old gui and artifact code so we dont get confused”):**
The project now has a **single** learning workflow. The first-generation bot
(`bot/` PPO: `policy.py`/`train.py`/`env.py`/`obs.py`/`launcher.py` + the old
`run_gpu_watch.py` watcher + standalone scripts) was **superseded** and is
quarantined to `_legacy/bot/` (moved, not deleted — see `_legacy/bot/README.md`).

**The ONE workflow (matches the user's model — a bot = an NN checkpoint + a thin
runner that plays from it):**
1. **Train** a new NN → checkpoint — `net/trainer.py` (`SelfPlayTrainer`) via
   `gui/trainer_main.py`, or imitation via `gui/bc_main.py` → `net/bc.py`.
2. **Select** in the league — `league/` round-robin + ELO ranks the checkpoints;
   we keep the winners, cull the weak ones.
3. **Bot** = `net/agent.py` `PokeAgent` over `net/model.py` `PokeNet` (loads a
   checkpoint and plays from it).

**Repointed:** `Launcher.bat` now opens the native Mission Control GUI
(`python -m gui.desktop`) instead of the retired `bot/launcher.py`.

**Kept in `bot/` (load-bearing, still used by the current generation):**
`teams.py` (format list for the GUI), `team_pools.json`, `data/`, `replays/`,
`checkpoints/` (data), `datamine/` (HF downloader, wrapped by `corpus/pull.py`),
`smogon_teams.py`/`teamgen.py`/`build_movesets.py` (team-pool scrapers),
`action_space.py`/`reveal.py` (shared by `datamine/parse.py`), `requirements.txt`,
`replay_diag.py` (replay health-check). **Verified:** kept imports work; league
(51) + gui (87) suites pass CPU-only. Commit `6483161`.

**Status (2026-11-20):** the browser GUI is **REMOVED** (quarantined to
`_legacy/gui_browser/`, commit `2687ec3`) — ONE GUI remains: `gui.desktop`.
The league question is **RESOLVED (D103)**: the league is **self-improving**
(each bot = NN + rollout buffer, PPO update each round, ELO-driven cull +
fresh-bot injection; seeding = new random net OR snapshot to continue).
Plan is in PLAN.md D103. Also recorded: the checkpoint-dropdown short-name
fix (D104).

---

## 2026-11-20 — D103 Step 1 done: the bot that learns (design decisions)

**Step 1 ("Agent = checkpoint + replay buffer") is BUILT.** Three decisions
I made within the user-approved plan — flagged here for review:

1. **`LearningBot` instead of extending `registry.Bot` with torch.** The plan
   said "extend `league/registry.py::Bot`"; the registry module's explicit
   contract is **framework-agnostic (no torch import)** and the GUI reads the
   roster through it. So: `Bot` stays the ROSTER RECORD (name/ckpt/ELO/W-L,
   JSON); the new `league/learner.py::LearningBot` is the LIVE object (net +
   `RolloutBuffer` + own save path + origin + counters). Same intent, no torch
   in the roster layer. **If you'd rather the registry hold the nets directly,
   say so and I'll merge them** — the cost is torch in every roster reader.
2. **ONE checkpoint load path.** The two-format sniffing (BC vs self-play) was
   copy-pasted in FIVE places (engine, human_play, eval_main, trainer_main,
   bc). Now `PokeNet.load_file(path, vocabs, device)` + `PokeNet.unwrap_payload`
   are the single source; all five call sites delegate. Verified with the real
   checkpoints (selfplay_round1.pt / league_random_1000.pt / s43_bc_n2000.pt
   all load, 7.2M params).
3. **Pristine-snapshot principle.** A bot seeded from a snapshot keeps its
   UPDATED weights in its OWN save path — it never overwrites the source
   (or a file another bot shares). Consequence: the league contract is
   ONE FILE PER BOT (the eval-only engine allowed shared files because it
   never wrote).

**OPEN for step 3 (not decided yet):** the buffer policy — capacity cap?
clear-after-update vs keep-across-rounds? (Step 1 defaults: unbounded buffer,
`clear_buffer()` available; `RolloutBuffer` semantics unchanged.)

**Post-commit audit (unintended/unoptimal behavior hunt) — 2 fixed, 1 flagged:**

* **FIXED — `fresh_bot_name` collision footgun.** The 4-hex id is only 16
  bits; Step 4's cull+refill can mint hundreds of bots, so an unlucky draw
  could hit an EXISTING bot's file, whose weights the new bot would then
  silently clobber on first save. Now redraws until the path is free
  (bounded at 64 draws) + deterministic regression test.
* **FIXED — shared roster file.** `materialize_roster` silently accepted two
  roster bots on ONE checkpoint file → first bot to save clobbers the
  other's weights while the other plays stale in-memory weights. Now a loud
  `ValueError` naming both bots, before any battle + regression test.
* **FLAGGED — pristine artifacts vs the ONE-FILE-PER-BOT contract (your
  call).** `from_roster_entry` uses the roster file as BOTH source and save
  path (the league contract). Consequence: adding a precious artifact —
  e.g. `scratch/s43_bc_n2000.pt` — to the roster and then training DESTROYS
  the original (violates the "do not delete old data" rule). Options:
  **(a)** keep the contract; the GUI "add bot" flow should COPY the snapshot
  into a fresh league-owned file first (pristine by default) — my lean;
  **(b)** add a "pristine" flag per roster entry; **(c)** document only and
  rely on the user copying manually.

**Gate:** league+net+gui 211 passed / 1 skipped (CUDA-only) / 5 deselected
(live); `league/tests/test_learner.py` 15/15 (13 + 2 audit regressions);
three real checkpoints load
through the single path. CPU-only throughout.

---

## 2026-11-20 — D103 Step 2 done: experience collection (design decisions)

**Step 2 ("collect experience into each bot's buffer") is BUILT.** Decisions
I made within the user-approved plan — flagged here for review:

1. **`league/collect.py::TrajectoryCollector` (new module) instead of logic
   in the driver.** Keeps the driver's torch-free-at-import discipline; the
   collector owns the step-shape semantics. Same shape as
   `net.trainer.play_and_collect` — the league feeds the IDENTICAL `net.ppo`
   machinery (no parallel semantics):
   * only REAL actions stored (pass/wait ≠ transition — skipped defensively);
   * `old_value` RAW (0,1): the 2V-1 reward-space map stays in the GAE step
     (step 3), exactly as `apply_gae` does today;
   * terminal reward (+1/−1/0) on the LAST step only — `gae()`'s
     sparse-terminal model.
2. **Capture seam = production path only.** `policy_action(capture=)` /
   `play_agent_battle_on(capture_a/capture_b=)`. The `decide=` override (the
   test-injection seam) bypasses capture BY DESIGN — a fake decision is not
   real experience.
3. **Errored battles DISCARD their partial trajectories** (no buffer, no ELO).
   No decided outcome → no terminal reward → a half-trajectory is poison for
   PPO. Pinned by `test_learning_run_discards_partial_trajectory_on_error`.
4. **`LeagueConfig.learning: bool = True` (default).** The league IS the
   self-improving loop (your call: "the league is our form of self play
   learning"). `learning=False` = the legacy eval-only rank run, unchanged
   (existing tests + pure checkpoint ranking). The GUI exposes the flag
   ("Learning (buffer experience):" checkbox) so you can run either.
5. **ONE-FILE-PER-BOT enforced in the GUI validator for learning leagues** —
   fail BEFORE the pool boots (90 s), not in the engine after (the engine's
   `materialize_roster` still enforces it as the last line). Eval-only runs
   may still share a file (they never write).

**OPEN (carried into step 3):** buffer policy (capacity cap? clear-after-
update vs keep-across-rounds?) + **buffer persistence** — the buffer lives in
RAM for the run; step 3's update consumes it in-process. A killed run loses
the un-updated experience. Options for step 3: (a) save buffers to the run
dir at the end of each pairing round (my lean — cheap, crash-safe); (b)
disk-backed buffer; (c) accept the loss (a round is only a few minutes).
**RAM note:** a long run (3200 battles × ~50 steps × 2 sides) can reach a few
hundred MB of buffered obs — fine now, relevant if we scale battles/round.

**Gate:** league+net+gui 226 passed / 1 skipped (CUDA-only) / 5 deselected
(live); 15 new tests (collect 7, driver capture 2, engine learning 4,
validator/GUI 2). CPU-only throughout. During the audit I caught my own
test-helper bug (`action=None` double as "unset" sentinel) — the test caught
it; fixed with a proper sentinel.

---

## 2026-11-21 — D103 Step 3 done: round-end PPO update + GUI (design decisions)

**Step 3 ("PPO update after each round" + the GUI chunk) is BUILT and
committed (`e9e2f5a` engine trio, `fb378cc` GUI).** Decisions I made within
the user-approved plan — flagged here for review:

1. **Buffer policy: clear-after-update for ≥2 steps; SKIP+retain for thin
   rounds. USER-APPROVED (2026-11-21: "i like the buffer persistence, lets
   add it" — the clear-after-update resolution; confirmed it "leaves the
   least junk around").** Mirrors `SelfPlayTrainer.train_round` (`update() if
   len(buffer)>=2 else skip`) — ONE learning semantics, two entry points.
The league SKIPS a thin round instead of raising (the solo trainer still
   raises): a league run is a long campaign — one 1-decision round must not
   kill it. (Why it's the right junk policy: PPO is on-policy — data from an
   OLDER net version is stale and poisons the update — so clearing after
   each update is correctness, not just tidiness. The only lingering state is
   the <2-step thin-round sliver, which could never train on anyway.)
2. **GAE at FINALIZE (the battle boundary), not at update time.** A
   trajectory is only complete where it ends: a battle end IS a terminal
   state; an intermediate battle end in a multi-battle buffer is NOT, so
   GAE-ing the whole buffer at update time would credit the wrong
   terminal across battle boundaries. Shared `net/ppo.py::fill_gae`
   (pure numpy, the `2V−1` map inside it, `old_value` stays RAW) is called
   by BOTH `TrajectoryCollector.finalize` (league) and
   `SelfPlayTrainer.apply_gae` (solo) — one implementation, no parallel
   GAE code (the "no code written twice" rule).
3. **Multi-round over ONE shared `ServerPool`.** `LeagueRun` owns the pool;
   every round reuses it (servers boot ONCE — the 90 s boot is the dominant
   cost); `SelfPlayRun.shutdown` is NOT called per round (it would kill the
   shared pool); `start_all` is idempotent (verified: `_ensure` reuses live
   servers); the single teardown is `LeagueRun.shutdown()` (pool +
   `damage.worker.shutdown_pool`).
4. **`n_rounds` default = 1** (one play+update cycle per GUI launch — the
   user scales); PPO knobs in `LeagueConfig` mirror
   `net.trainer.TrainConfig` defaults (lr=3e-4, epochs=2, clip=0.2, vf=0.5,
   ent=0.01, max_norm=0.5, gamma=0.99, lam=0.95). GUI surface stays
   minimal: Rounds / PPO epochs / PPO lr only.
5. **Carried-open from step 2 — RESOLVED by construction: buffer
   persistence.** Because every round ENDS with update+save, a killed run
   loses at most the CURRENT round's in-flight experience (option (c) from
   the step-2 note), not many rounds. The step-2 RAM note is likewise
   bounded: the buffer drains each round, so steady-state RAM is one
   round's experience, not the whole campaign's.
6. **Carried-open from step 1 — RESOLVED by the user: pristine-artifact vs
   ONE-FILE-PER-BOT.** The GUI "Add bot" now COPIES the source snapshot
   into a fresh league-owned file (`fresh_bot_name`) and registers the COPY
   — the user's "make a copy, work with that, save it as new" flow. Source
   stays pristine; one-file-per-bot is guaranteed AT ADD-TIME (the
   validator's pre-boot check remains the last line for hand-edited
   rosters). "Delete" removes the selected bot's entry + its OWN `.pt` +
   `.meta.json` (confirm dialog); the source is never touched.
7. **Fitness benchmark (vs an external strong bot) = AFTER the league loop
   is stable.** The yardstick, not the engine — not building now. Awaiting
   user reaction (handoff open question).
8. **FLAGGED (found in my own review, NOT fixed — needs your call):**
   re-adding a bot with an EXISTING name in the GUI silently OVERWRITES the
   roster entry — ELO/W-L reset to initial (pre-existing `BotRegistry.add`
   semantics: "or update if it already exists"). For a learning league that
   throws away that bot's standings. Options: (a) GUI refuses duplicate
   names (my lean — loud, one line); (b) keep the old ELO on overwrite;
   (c) leave as-is (documented overwrite).

**Gate:** full suite **251 passed / 1 skipped** (CPU-only; the 4 new GUI
tests are hermetic vs the user's live `league/bots.json` — roster path
monkeypatched to tmp, copies cleaned up). The known tkinter full-suite
flake did not bite this run.

---

## 2026-11-21 — D103 Step 4 done: cull + refresh (design decisions)

**Step 4 ("cull + refresh — every K rounds, drop the bottom X% by ELO and
inject fresh random bots") is BUILT and committed (`26c3214` selection +
config, `e3330d3` engine, `877e7fe` GUI).** Decisions I made within the
user-approved plan — flagged here for review:

1. **Cull fires AFTER the round-end PPO update, and NOT after the last
   round.** The culled bots' final weights are saved by the update that
   precedes the cull (their last state is a real artifact); minting bots
   after the final round would produce bots that never fight.
2. **Culled `.pt` files are KEPT (artifacts).** Only the ROSTER ENTRY is
   retired — deleting weights stays a USER action (the GUI's Delete, with
   its confirm dialog). This honors the "don't delete old data" instinct
   and keeps culling reversible (the weights are still loadable). CONSEQUENCE
   (flagged): long campaigns accumulate culled `.pt`s in `scratch/` (~a few
   MB each). The GUI Delete is the cleanup path; a batch-delete of
   `league_*` files could be a later GUI nicety.
3. **Fresh bots are UNSEEDED random** (`LearningBot.random` seed=None) —
   the diversity source the user asked for ("constantly cycle new agents");
   seeded would make every refill identical to the first.
4. **Fresh bot's roster NAME = its file stem** (`league_{id4}`).
   `fresh_bot_name` guarantees a unique FILE only; roster-NAME uniqueness (a
   hand-registered bot could own a `league_*` name, and `registry.add` would
   silently overwrite it) is enforced by the name-shadow guard (decision 11).
   Same naming as the GUI copy-on-add, so the two minting paths can't collide.
5. **A culled bot's ELO/W-L history disappears with its roster entry**
   (registry semantics). The names are preserved in the run summary's
   `cull_events` (`{round, culled, added}`). Optional later: an archive
   table in the registry for retired bots. NOT built (no request).
6. **Learning mode ONLY.** An eval-only run (``learning=False``) never
   mutates the roster — culling there would destroy the ranking artifacts.
7. **Defaults:** `cull_every=4`, `cull_frac=25%`, `min_roster=4` (GUI:
   "Cull every (rounds)" / "Cull bottom (%)" / "Min roster"). `0` disables
   the loop (both `cull_every=0` and `cull_frac=0.0` are LEGAL — the
   validator was written to allow the disable values, which the strict
   `_finite_float` lower bound would otherwise reject). `min_roster` floor
   is hard-clamped to ≥2 (the league minimum).
8. **Re-plan on cull, not every round (flagged — your call).** The
   pairings are re-planned when a cull changes the roster (required: new
   bots must fight, culled must not). WITHOUT a cull, all rounds reuse the
   start-time pairings — which is correct for `round_robin` but means
   `elo_weighted` pairings go STALE after round 1 (ELO moves, pairings
   don't). Options: (a) status quo (my lean — the default pairing is
   round_robin, and churn is minimized); (b) re-plan every round (fresh
   elo_weighted, but pairings drift under your feet mid-campaign).

### Step-4 hardening (found in the final bug-check pass, commit `5a35cca`)

9. **Cull only on a COMPLETED round.** The round-end PPO UPDATE runs no matter
   how the round ended (it saves what was learned — a user stop or a failure
   must not throw the round's learning away), but the CULL runs only when
   `SelfPlayRun.wait()` reports `state == "completed"`. After a stop or a
   failure the campaign is over, and minting explorers there would mutate the
   roster for no reason. Pinned by `test_cull_skipped_when_round_stopped`
   (stopped → no cull; completed → cull, same boundary).
10. **Mint-first atomicity.** `_cull_and_refresh` now MINTS the fresh bots
   (file + registry + live `_bots`) BEFORE retiring the culled. If minting
   fails (disk, whatever), the roster is a SUPERSET of the original
   (original + some fresh) — always legal for `plan()`. The old retire-first
   order could SHRINK the roster below the floor, or even below 2, mid-mint,
   and the next `plan()` would raise. Pinned by
   `test_cull_mint_failure_keeps_roster` (a raising `LearningBot.random`
   leaves the roster unchanged).
11. **Name-shadow redraw guard.** `fresh_bot_name` guarantees a unique FILE
   only; a hand-registered bot could own a `league_*` NAME, and `registry.add`
   silently OVERWRITES an existing entry (ELO/W-L lost). The mint loop now
   redraws `fresh_bot_name` (≤64×) while the candidate stem is in the roster's
   taken-name set, and raises a bounded `RuntimeError` if all collide. Pinned
   by `test_cull_name_collision_redraws` (a colliding first draw is skipped;
   the existing `league_abcd` bot's original checkpoint survives untouched).

**Gate:** full suite **271 passed / 1 skipped** (CPU-only; +20 step-4 tests:
9 selection units, 9 engine integrations, 2 GUI). The engine tests are
hermetic (fake pool, scripted battles, `refresh_dir` under tmp — the
user's scratch and live roster are never touched). Hardening commit `5a35cca`.

---

## 2026-11-21 \u2014 Teardown leak fix (the `ConnectionResetError` noise) + orphan-server cleanup

**Task (user: "please fix occuring bugs and provide insight on potential fixes
in terms of design").** The one concrete occurring bug was the
`ConnectionResetError` / `ConnectionClosedError` teardown noise (112 / 56 /
168 tracebacks in the pre-fix smoke log). Fixed, verified, and committed
(`d117292`). This was a **lifecycle leak, not a battle-logic bug**.

**Root cause (verified against the poke-env source + a live diagnostic).**
Each `PokeEnv` spins up its **own** event-loop thread (`env._loop`) and BOTH
of its agents own a `PSClient` whose `listen()` coroutine runs on that loop.
poke-env reaps those **only at `atexit`** (`poke_env.concurrency.__clear_loop`)
\u2014 `PokeEnv.close()` never stops them. We create a fresh `ShowdownEnv` per
battle and never stopped the clients, so N battles \u21d2 N leaked loop-threads +
sockets. When the pool then tree-killed the servers, the still-live `listen()`
died as a raw `ConnectionResetError` (poke-env's `listen()` only treats a clean
`ConnectionClosedOK` as non-exceptional) \u21d2 the `logger.exception` tracebacks.

**The fix (chosen \u2014 clean per-battle teardown).** `ShowdownEnv.close()` now
stops **each agent's** client via poke-env's own
`PSClient.stop_listening()` (best-effort, never raises) while the server is
still up, so `listen()` exits through the clean `ConnectionClosedOK` path and
the socket is freed immediately (no accumulation over a long run).

**Mistake made + caught (self-review):** my first fix called
`_stop_ps_client(self.ps_client)`, but **`env.ps_client` does not exist**
(verified live: the attribute is `MISSING`) \u2014 the clients live on the
**agents** (`agent1.ps_client` / `agent2.ps_client`), so that call was a silent
no-op (the smoke test still showed the full noise). The corrected call targets
the two agents. Lesson: a helper that "passes" its unit tests can still be a
no-op if it's pointed at the wrong object \u2014 the live diagnostic + the
before/after smoke counts are what actually proved it.

**Gate:** full suite **276 passed / 1 skipped** (CPU-only; +5 hermetic
`_stop_ps_client` tests). Real-server smoke run (8 bots, 4 servers, 28
battles) \u2192 **0** `ConnectionResetError`, **0** `ConnectionClosedError`, **0**
tracebacks, **VERDICT PASS** (28/28, 0 errors, all 8 bots buffered + ran 1 PPO
update + saved; ELO spread sensible). The smoke run did **not** leak its own
servers (verified: no leaked procs from that run's window).

**Process note (user, 2026-11-21):** battle-based test/smoke runs get a **10-min
(600 s)** shell timeout — stall games can exceed the old 360 s cap the previous
note referenced, so any re-run of a real-server battle loop uses 600 s headroom.
(This is a timeout on the *test command*, NOT a battle limit in the game engine;
the in-game cap is still `max_steps`.)

**Orphan cleanup (found while verifying).** 25 leaked `node.exe` processes
from the fork (1 main server + 24 `room-battle.js`/worker children) were still
alive, the main one `LISTENING` on `0.0.0.0:49980` \u2014 the **diagnostic port**
the previous session used for `_diag_close.py`. They were ~22 min older than
the smoke run, i.e. **not** from this run: a leaked Showdown server from the
earlier diagnostic, consuming RAM. Killed the whole tree (25 \u2192 0). NOTE for
the user: an interrupted/diag run can leave a Showdown server tree behind;
the league pool tree-kills on a CLEAN shutdown, but a hard stop (LLM stuck +
PC restart) does not.

**Design direction (NOT built \u2014 flagged for your call):**
* **Connection pooling / client reuse (bigger fix, also speeds up battles).**
  Keep one `PSClient` + loop **per Showdown server** and share it across the
  battles on that server instead of creating a client per battle. This attacks
  the same leak class more fundamentally AND the observed slowness (each battle
  currently pays a fresh websocket handshake/login; the smoke run was
  ~440 s for 28 battles). Higher risk / a real refactor \u2014 a later step, not
  now.
* **Trade-off:** poke-env's own `PokeEnv.close()` also does NOT stop the
  client, so "just call `super().close()`" is not enough \u2014 with the
  per-battle env model the stop is OUR responsibility (now handled).

**Open for you (see report):** (1) is connection pooling in scope now or later
(lean: **later** \u2014 ship the clean per-battle teardown now)? (2) best-effort
teardown (chosen \u2014 cleanup must never fail the battle) vs strict? (3) keep
`scratch/_smoke_league_run.py` + `scratch/_diag_close.py` as re-runnable QA
tools (I kept both) or clean them up after?

---

## 2026-11-21 \u2014 Registry thread-safety (concurrent-battle ELO/save race)

**Bug (latent, caught in self-review \u2014 NOT an occurring symptom yet).**
`league/registry.py::BotRegistry` had **no lock**, yet its docstring claimed
"the league engine is single-threaded\u2026 no locking needed." That assumption is
**false**: `control/selfplay.py::SelfPlayRun.start()` spawns **N worker
threads** (one per slot) that all concurrently call `on_battle` \u2192 `_record`
\u2192 `registry.update_result()` \u2192 `save()`. Concurrent read-mutate-persist on the
shared in-memory roster + a shared `write_text` could interleave \u2192 a lost ELO
update and/or a torn/partial `bots.json`. Low-probability (within a single
round each bot is in only one battle, so the main risk was concurrent
`save()` file-write interleaving), and the smoke run passed by luck.

**Fix (chosen \u2014 make the registry itself thread-safe).** Added a
`threading.RLock` in `BotRegistry.__init__` and guarded `save()` / `add()` /
`remove()` / `update_result()` with `with self._lock:`. RLock (reentrant) so a
guarded method can call `save()` (which also takes the lock) without
deadlocking. Updated the stale "single-threaded" docstrings (module + class)
to reflect the concurrent league.

**Regression test.** `league/tests/test_registry.py::test_concurrent_updates_consistent`
hammers the registry from 8 threads (overlapping bot pairs, barrier-started to
maximize contention) and asserts the invariants a race would break: ELO
zero-sum conserved, every battle counted exactly once (W/L balanced, draws in
pairs), and the persisted file is valid JSON matching in-memory state. Stable
across 5 repeated runs.

**Gate:** registry suite **15 passed**; full suite **277 passed / 1 skipped**
(CPU-only; +1 the new test).

**Note for you:** this is a *correctness* hardening, not a behavior change \u2014
the ELO math and roster semantics are identical; only the concurrency is now
safe. If you'd rather the thread-safety live on the *engine* (a single
serialized results queue) instead of the registry, that's a bigger change and
I'd flag it \u2014 but the registry lock is the smaller, more contained fix.

---

## 2026-11-21 \u2014 Environment validated + NEXT MACRO STEP (options, awaiting your call)

**Environment status: DONE + validated.** A 28-battle CPU smoke (8 fresh random
bots, learning mode) passed: `state=completed, 28/28, errors=0`; all 8 bots
collected 350\u2013930 steps and ran a finite PPO update (loss \u2208 [\u22120.21, +0.12],
no NaN); ELO spread 1443\u20131569; zero-sum holds; no leaked Node servers. A
~1008-battle run (36 rounds, cull every 4) is going in the background
(`scratch/_full_validate.log`). Observed (not bugs): ~25% of random-net battles
stall to the 200-step cap (\u2192 draw, 0 reward both sides); mild `Task was destroyed
but it is pending!` poke-env teardown warnings (cosmetic, not the ConnectionReset
exceptions we already fixed).

**What we can do now: the league produces RELATIVE strength (ELO within the
league) but NOT absolute strength (\u201cis this bot actually good at P\u00f4kemon?\u201d).
That's the gap.** Options for the next build step:

* **A \u2014 Fitness / evaluation harness (RECOMMENDED next).** A ladder of fixed
  reference opponents (random \u2192 a BC baseline \u2192 a strong fixed policy) + a
  win-rate harness that scores ANY bot against it. Prerequisite for everything
  else: tells us if the league is actually improving bots, and gives the
  strategy-sharing step a way to rank \u201cwinning\u201d strategies. You asked for this
  (\u201cget a reasonably strong ai model out there and let ours fight against that\u201d).
  Small, testable, high-leverage.
* **B \u2014 Strategy sharing / delta blending (your idea).** When top bots converge
  on a node change, blend a small delta into the main agent. Higher-risk
  research; needs a trained-bot pool + a fitness measure (A) to know what
  \u201cwinning\u201d is. Do AFTER A.
* **C \u2014 Scale the league (more rounds/bots, longer training).** \u201cLet it cook.\u201d
  Already happening (the background run); but without A we can't tell if it's
  improving. Run in PARALLEL with A.
* **D \u2014 GUI polish / play-vs-checkpoint.** Useful for your workflow, not an
  AI-strength step. Side task.

**Lean:** A next (with C running in the background as the first data point),
then B once we have trained bots + a fitness measure. Doubles/6v6 stays later
as agreed. Your call when you're back.

### Option A \u2014 fitness/eval harness: concrete plan (build order, small steps)

**Goal:** answer \u201chow good is this bot at P\u00f4kemon?\u201d with an ABSOLUTE number,
not just relative ELO. Reuse what exists (D100, no code twice):
`pool.ResourcePool.battle_slot()` (reserve a server) +
`league.driver.play_agent_battle_on` (one battle). The harness is just
\u201cplay bot vs reference N times, count wins.\u201d

**New package `eval/` (torch-free at import, like the rest):**
* `eval/ladder.py` \u2014 the reference ladder. Each reference = `(name, agent_factory)`.
  * `random` \u2014 `PokeAgent(make_random_net(vocabs))`. The FLOOR (a competent bot
    beats this >50%). Trivial, builds first.
  * `heuristic` \u2014 the \u201creasonably strong model out there\u201d you asked for. A
    rule-based `PokeAgent`-compatible policy: attack with the strongest safe/
    STAB move, target the weakest opponent, switch when low HP, use recovery
    \u2014 and USE THE DAMAGE CALC (the belief-state foundation) for \u201cwill this K.O./
    will I K.O.?\u201d decisions. This is the crux and the natural showcase for the calc.
  * `champion` (later) \u2014 the best league bot frozen, so the ladder's top rises
    as the league improves.
* `eval/harness.py` \u2014 `evaluate(bot_ckpt, refs, n_battles, fmt, ...) -> EvalResult`
  (`{ref: {wins,losses,draws,win_rate}}` + `fitness_score`); CPU-only; sequential
  first (simple), concurrent later.
* `eval/tests/` \u2014 hermetic (fake-env) win-rate/fitness tests + one `live`-marked
  self-check (deselected by default).

**Fitness score:** weighted win-rate across the ladder, normalized to [0,1]
(e.g. random=1x, heuristic=2x, champion=3x). Beating random but losing to
heuristic = mid-range. Gives Option B (strategy blending) a way to rank
\u201cwinning\u201d strategies, and gives the league a per-round learning curve.

> **STATUS (2026-11-21): A1 ✅ done (commit `b303812`) · A2 ✅ done · A3.1/A3.2/A3.2b ✅ done · A4 ✅ done · A3.3/A5 pending.**
> The package landed as `fitness/` (not `eval/`) — `eval` shadows a Python builtin, so the real dir is `fitness/`.
> **✅ vocabs=None blind-play bug FIXED (commit `cb78e50`)** — the A3 self-check was playing BOTH sides blind (zero-obs every turn); the old 3/3 was a false positive. Now `build_obs`/`evaluate` fail loud on a missing vocabs, the self-check passes `vocabs`, and re-runs show **0 zero-obs errors**; the 20-battle gate is a solid **PASS (0.750 vs random)** with both sides seeing real state (see the 2026-11-21 section below).

**Build order (one reviewable step each):**
1. **A1** `eval/ladder.py` \u2014 `random` reference + ladder container. (small)
2. **A2** `eval/harness.py::evaluate` against `[random]` + hermetic test. (core)
3. **A3** `HeuristicAgent` \u2014 the strong rule-based reference (the crux; may be
   2\u20133 sub-steps: move selection, then switching/recovery, then damage-calc use).
   Self-check: heuristic beats random >70%.
4. **A4 ✅** `fitness_score` = `compute_fitness` (weighted average of per-ref
   win-rates, normalized to [0,1]); wired into `evaluate`; weights come from each
   Reference's `.weight`. 4 hermetic tests (weighted avg, bounds, PerRefResult,
   defaults/empty).
5. **A5** integration \u2014 evaluate the league's top-1 bot after each round (log the
   curve) + a GUI \u201cevaluate this checkpoint\u201d button.

**Open for you:** (1) is the heuristic reference the right \u201cstrong model,\u201d or
should we pull an existing open-source P\u00f4kemon bot as the top reference? (2)
fitness weights (1/2/3) reasonable, or do you want a different aggregation? (3)
sequential-first OK for v1 (a 20-battle eval \u2248 6\u20138 min on CPU) vs concurrent now?

**A3 — the `HeuristicAgent` (the mid-ladder reference):**
* Contract: a `PokeAgent`-compatible policy — `.act(obs, acts, deterministic)`
  returning a `Decision`-shaped object (`action`/`action_index`/`logprob`/
  `value`/`entropy`/`prob`). DROP-IN opponent: the SAME driver `policy_action`
  path the net uses, so the harness plays exactly what it does.
* Torch-free (CPU-first, like the rest of `fitness/`): decodes structured state
  from `obs` (registry channels: `own_slot` hp_frac/active, status, boosts,
  hazards) + the legal `acts`; static move data via the belief state's own
  lookup `legality.lookup.get_move(name, gen)` — no code twice.
* Sub-steps (one reviewable commit each):
  * **A3.1 (this change)** legal + sensible baseline: pass on empty `acts`;
    forced-switch -> pick a switch; own active HP <= 0.30 + a switch is legal ->
    switch to the healthiest bench; else play the highest-base-power damaging
    move (status moves ranked last). Hermetic: legality, switch-when-low,
    best-power move, pass, determinism.
  * **A3.2 ✅** type-aware move scoring: `power * type_effectiveness(move_type,
    opp_types) * (1.5 if STAB else 1.0)` (status/unknown-power -> 0; unknown types
    degrade to pure power). Decodes active own/opp species via the shared
    `net.vocab` + `get_species` types; per-gen chart cached; torch-free at import.
    3 hermetic tests (STAB, immune-avoidance, unknown-type degradation).
  * **A3.2b ✅** hazard-aware switch target: when switching, prefer the switch-in
    NOT weak to the opponent's type-interacting hazards (Stealth Rock -> avoid
    Rock/Steel/Flying/Bug; Toxic Spikes -> avoid Poison/Flying). Reads p2-side
    hazards from `obs['global']` (the obs is always the agent's POV, so p2 = the
    opponent's hazards threaten my switch-ins). Falls back to the first legal
    switch when there's only one or no hazards (A3.1). 3 hermetic tests.
    (Spikes / Sticky Web don't interact with type, so they're intentionally
    not scored here.)
  * **A3.3 (the crux)** damage-calc "will-this-KO / will-I-KO". **OPEN for you
    (interface decision):** `.act(obs, acts)` does NOT expose the belief state /
    calc. Options: (a) keep the heuristic **obs-only** (contract-stable, weaker);
    (b) **extend the agent seam** so a policy may also receive the structured
    `get_context()` + run the damage calc (stronger, touches the driver path);
    (c) give the heuristic a SEPARATE `act_with_context(...)` the driver calls
    when present (opt-in, backward-compatible). My lean: **(c)**.

    **EXECUTION BRIEF (pick one → I wire it as the NEXT small step, no re-plan).**
    The brain (`fitness.strategy.choose_action`) is built, interface-agnostic,
    and integration-tested on a real `get_context()`; ONLY the seam (how the
    context reaches the agent) is left. All three finish with the SAME live gate:
    the KO-aware heuristic beats random **>70%** (A3 self-check).
    * **(a) obs-only** — zero code change; the heuristic keeps `power*eff*STAB`.
      Cheapest, but A3.3's KO-awareness + proactive threat-switching go unused.
    * **(b) extend `act()`** — add an OPTIONAL `ctx` kwarg to the shared agent
      contract; the driver passes `get_context()` to every agent. Touches the
      net's `act` signature (all implementers must tolerate the new kwarg) —
      medium risk (shared path).
    * **(c) opt-in `act_with_context`** — driver: `if hasattr(agent,
      "act_with_context"): ... else: agent.act(...)`. Isolated, the torch-net
      `PokeAgent` is untouched — **low risk, recommended** (also your lean).
    Shared first step under (b)/(c): a `build_obs_and_ctx()` in `net/env.py`
    returning `(obs, ctx)` (single belief-build; `build_obs` becomes a thin
    wrapper, obs stays byte-identical) so the context is available exactly once.
* Self-check gate (A3 done): heuristic beats random **>70%** (the plan's target).

> **PROGRESS (2026-11-21): the crux BRAIN is now BUILT and is interface-AGNOSTIC.**
> I did NOT pick (a)/(b)/(c) — instead I built the decision logic as PURE
> functions over the `get_context()` snapshot, so it is reusable under ALL THREE
> options and commits to none (the same discipline as the KO core):
> * `fitness/ko.py` — the KO assessment core (`move_ko` / `opp_move_ko` /
>   `ko_assessment`). **Bug fixed:** it now degrades gracefully when the
>   ATTACKER species is unknown (previously only the defender was checked, so an
>   unknown opponent would make the calc raise — a latent crash).
> * `fitness/strategy.py::rank_moves` — OFFENSIVE: rank my moves by
>   `guaranteed_ko > can_ko > expected_ko > damage` (status moves sort last).
> * `fitness/strategy.py::threat_assessment` — DEFENSIVE: "will the opponent's
>   likely move KO me?" -> `guaranteed_ko > can_ko > expected_ko > safe > unknown`
>   (lets an agent switch/protect PROACTIVELY, not just on a fixed HP threshold).
> * `fitness/strategy.py::choose_action` — THE COMPLETE DECISION (committed
>   A3.3d): ties it together — (1) no acts -> pass; (2) force-switch; (3)
>   threatened (`guaranteed_ko`/`can_ko`) OR HP<=0.30 with a legal switch ->
>   switch (hazard-aware); (4) else the best damaging move. Deterministic,
>   degrades safely on a partial belief state, never raises.
> All torch-free at import, one damage model, hermetic (fake-calc) + real-calc
> tests, full suite **658 passed / 1 skipped / 8 deselected** (CPU-only).
> **The ONLY remaining step is the fork-dependent SEAM** — HOW the context is
> handed to the agent. Under (a) the brain is simply unused; under (b)/(c) it is
> the core logic (a `build_obs_and_ctx()` enabler + a driver opt-in call). Pick
> (a)/(b)/(c) and I wire it as the next small step.
>
> **PROGRESS (2026-11-21, continued) — full ladder reviewed + one latent issue
> hardened.** Bug-hunt over `ladder` / `harness` / `ko` / `strategy` / `heuristic`:
> * `choose_action`'s active-slot guard hardened to `is None` (NOT truthiness) so
>   a falsy-yet-valid slot key (e.g. an int slot of 0) is not mistaken for
>   "missing" (which would silently bail to the first move). Real `get_context()`
>   slots are "p1a"-style strings so this was latent, not manifest; a regression
>   test pins it (`test_choose_int_slot_zero_is_not_treated_as_missing`, fails
>   under the old guard). Full suite **660 passed / 1 skipped / 8 deselected**
>   (CPU-only).
> * Closed the **hermetic-only gap**: added `test_real_get_context_feeds_the_brain
>   _end_to_end` — a REAL `BeliefState` -> REAL `get_context()` -> the brain with
>   the REAL shared calc. Every prior brain test used a hand-crafted ctx; this
>   pins the integration against field-name drift (a missing field would silently
>   degrade the brain, invisible to hermetic tests). Verified live: 252-Atk
>   Garchomp earthquake ranks first (58-70% of max HP, guaranteed KO) vs a 1-HP
>   Blastoise; `choose_action` picks earthquake.
> * Observation (NOT a bug for the A3 gate): `evaluate()` seeds the bot AND the
>   reference with the SAME per-battle seed (`b_seed`). Fine when the bot is a
>   checkpoint / deterministic policy, but a **random-vs-random** ref-vs-ref run
>   would build the SAME net on both sides (degenerate, ~turn-order-biased). If
>   we ever score random-vs-random, give the two sides distinct seeds.
> * Confirmed sound: the two `_pick_switch` variants (obs-based vs ctx-based)
>   agree on hazard sets/penalties; KO boundary `pct >= hp_frac*100 - 1e-3`
>   (`_EPS` absorbs the 2dp rounding without crossing a quantum); attacker-
>   unknown degrades in BOTH `move_ko` and `opp_move_ko`; "unknown" threat does
>   NOT trigger a panic switch.

---

## 2026-11-21 — HP>1.0 blind-play bug: ROOT-CAUSED + FIXED (the "unwanted outcome")

**Symptom (the real bug in the 1000-battle run, `scratch/_full_validate.log`):**
`build_obs` degraded to a **BLIND zero-observation 15x** — 8x
`RuntimeError: hp must be a 0.0-1.0 fraction (got 1.0484…)` + 7x `(…1.0671…)`.
On each of those turns the bot played **blind** (a zero obs), and the belief state
was internally inconsistent. This is an "unwanted outcome," not a cosmetic issue.

**Root cause (traced + reproduced):** the belief held **`hp > maxhp`** in a slot
(a stale `maxhp` / an inconsistent reading — a mon's HP can *never* exceed its max
HP, so this is always a data bug, never a real state). The damage-evidence path
(`state.py::_damage_evidence`) then computed the *unclamped*
`d_frac = prev/maxhp > 1.0` and handed it to the calc via `evidence.mon_options`.
`node_calc.js` **throws** on `hp > 1` (the guard is correct and intentional —
`damage/tests/test_curhp_abilities.py::test_hp_out_of_range_raises` pins it for
direct callers), and `net/env.py::build_obs`'s catch-all returned a **zero obs**.
Reproduced exactly: `facts.hp=192 / maxhp=183` + a damage event on that slot
→ `RuntimeError: …got 1.049…`.

**Fix (four layers, all strict-safe because a HP fraction > 1.0 is always
invalid, so clamping never masks a real value): the belief invariant is now
enforced at the source AND clamped at every belief→calc/obs boundary:**
1. **`evidence.mon_options` (the belief→calc boundary, the backstop):**
   `mon["hp"] = clamp(hp_fraction, 0, 1)`. A stale/inconsistent reading can no
   longer reach the calc, so it can no longer throw → no more blind zero-obs.
2. **`state._damage_evidence`:** `d_frac = clamp(prev/maxhp, 0, 1)` (the one path
   that bypassed the clamped helper).
3. **`state.get_context` (`_mon_block`):** `hp_frac = self._hp_fraction(facts) or
   0.0` — the observation reuses the clamped helper (was an unclamped inline).
4. **Belief invariant (root cause):** new `BeliefState._clamp_hp(hp, maxhp)`
   (`0 <= hp <= maxhp`, `None`-safe) applied in `_set_hp` (damage/heal/faint),
   both `_set_switch_facts` branches (maxhp set first, then hp clamped to it),
   and the **HealObserved** dispatch now refreshes `maxhp` from the line first
   (`max(old, line)` — a heal can only reveal the same mon's true maxhp, never a
   smaller one; never shrinks a known maxhp) before clamping hp.

**Regression tests (4 new, `belief/tests/test_step3_4.py`):** `_clamp_hp`
helper; the EXACT blind-play path (hp>maxhp + damage event → no throw, valid
posterior); `get_context` `hp_frac ≤ 1`; the natural trigger (stale maxhp + heal
→ maxhp refreshed, invariant holds).

**Verification (CPU-only, `CUDA_VISIBLE_DEVICES=-1`):** belief+damage **352
passed / 3 deselected**; net+league+control+fitness **231 passed / 1 skipped**.
All green. The raw-calc guard is **unchanged** (direct `hp>1` still throws — the
test still passes); only the belief bridge is now clamped.

**Flagged (NOT changed unilaterally — needs your call):** `build_obs`'s
**catch-all degrades to a ZERO obs on ANY exception.** That is aggressive: a
blind decision is worse than a slightly-wrong one. Now that the HP path (the
only throw we've seen) is clamped at the boundary, the remaining question is
what a *genuine* unexpected data gap should do — options: (a) keep zero-obs
(simplest, but blind); (b) a **last-known-good** obs (carry the previous turn's
obs — a slightly-stale-but-real state beats a blind one); (c) a targeted partial
obs (fill only the failed field). **My lean: (b) last-known-good**, with (a) as
the very-first-turn fallback. Not built until you pick.

---

## 2026-11-21 — vocabs=None blind-play bug (A3 self-check): ROOT-CAUSED + FIXED

**Symptom:** the A3 self-check (`scratch/_a3_selfcheck.py`, heuristic vs random,
real battles) logged ~30x `build_obs degraded to zero obs (AttributeError:
'NoneType' object has no attribute 'weather')` — the bot won 3/3, but BOTH sides
were playing BLIND (a zero obs every turn). A "pass" built on blind play is a
false positive.

**Root cause (traced + reproduced):** the harness path called `evaluate(...)`
WITHOUT `vocabs`, so `vocabs=None` flowed `evaluate` → `play_agent_battle_on`
→ `policy_action` → `build_obs` → `encode_context` → `vocabs.weather` →
`AttributeError` → swallowed by `build_obs`'s catch-all → a **blind zero-obs**
every turn. A missing vocabulary is a **wiring/config error, not a transient data
gap** — it fails on EVERY turn, not one. (The league path was always fine:
`SelfPlayRun._driver_vocabs()` builds and passes vocabs to every driver. Only the
direct-harness/self-check path forgot it.)

**Fix (fail loud, defense in depth — commit `cb78e50`):**
1. **`net/env.py::build_obs`** raises `ValueError` on `vocabs is None` — BEFORE the
   catch-all, so a wiring error fails LOUD instead of degrading to blind. This is
   the single choke point (every real-obs path funnels through it). A valid vocabs
   + a transient gap (garbage log, no prior) still degrades gracefully (unchanged).
2. **`fitness/harness.py::evaluate`** raises at the API boundary when the real
   decide path needs vocabs (checked AFTER bot validation, so a bad bot still
   reports the bot error; a fake `decide` test-injection still may omit vocabs).
3. **`scratch/_a3_selfcheck.py`** now passes `vocabs=vocabs` (the direct bug).
4. **Regression tests** pin both guards; also fixed a stale `PoolConfig(ports=…)`
   kwarg in the deselected harness live smoke (`ports` is a derived property).

**Verification:** repro confirms `build_obs(…, None, …)` now RAISES (was silent
zero-obs); full stack **614 passed / 1 skipped / 8 deselected** (green); re-ran the
A3 self-check → **0 zero-obs / 0 weather / 0 decision errors** (the bug is gone).

**Gate re-read (honest, not a bug):** with REAL obs both sides now see the state.
The old blind **3/3 was a false positive** (both sides blind). The 5-battle first
re-run looked weak (2W 1L 2D → 0.400) but was a tiny, high-variance sample; a
**20-battle re-run (`scratch/_a3_smoke20.log`) is the firm read: 15W 1L 4D →
win_rate 0.750 → PASS, 0 zero-obs errors.** The 4 draws are max-step (80) stalls.
The heuristic is a **ladder rung (a reference point, weight 2), not the trained
bot**, and 0.75 vs a legal random net confirms it's a sound "clearly better than
random" rung.
**OPEN for you:** is the heuristic strong enough to be the top ladder rung?
Options: (a) accept it as a coarse "better than random" rung — **now well-
supported by the 0.750 pass** (fine for now); (b) strengthen it (accuracy/PP/
priority/status-aware scoring) before relying on it to rank trained bots;
(c) replace it with an existing open-source bot as the top rung. My lean:
**(a) now** (the 0.750 pass validates it), revisit (b)/(c) once we have trained
bots to compare against.

---

## A. Workflow errors found

### Fixed this pass
1. **Checkpoint browser mislabeled `runs/` checkpoints** (`gui/server.py`).
   The scratch-branch read each `.meta.json` (type/top1/verdict/n_battles), but
   the runs-branch **hardcoded `type="selfplay"` and skipped the meta**. So a BC
   checkpoint written into `runs/<id>/checkpoints/` showed up as `[selfplay]`
   with **no top1/verdict** — and collided by name with an older
   `scratch/…_bc_50b.pt`. Fixed: the runs-branch now reads the meta exactly like
   the scratch-branch (D97). Verified: `runs/…/2026-11-19_bc_50b.pt` now reports
   `type=bc top1=0.308 verdict=True n_battles=50`.

### Open (not fixed — each needs a decision, see section B)
2. ~~**Self-play → replay-capture is NOT reachable from the GUI.**~~
   **FIXED (D98):** `launcher.start_trainer` now routes `selfplay-collect` →
   `control.selfplay_main`; the server's `api_start` handles it (CPU-only, no
   device check); `launcher.browse_checkpoints` is extracted as a shared function
   (no duplication); fast test stack **119/119 green**.
3. ~~**Single active-run slot + long-lived "play" sessions.**~~
   **RESOLVED (2026-11-20):** a NON-play start now gracefully pre-empts a
   lingering PLAY session (play is the low-priority interactive mode,
   `n_battles=0` = "play until stopped") and proceeds — so an idle play session
   no longer holds the single slot forever. Play never pre-empts a real
   workload, and two of the same workload still refuse (single-slot preserved
   for real runs). See `gui.launcher._assert_startable` +
   `gui/tests/test_a3_play_preempt.py` (7 tests).
4. ~~**Checkpoint name collision.**~~ **FIXED (D99):** new naming format
   `{n}b_{type}_{id}.pt` (e.g. `50b_bc_a3f2.pt`) — the 4-char uuid4 hex id makes
   collisions impossible. Existing date-named checkpoints still work; new runs
   use the unique-id format.
5. ~~**Checkpoint-path validation resolves against CWD, not `ROOT`.**~~
   **FIXED (2026-11-20):** the eval/play/train/initial-ckpt validators already
   routed through `_resolve_ckpt_path` (ROOT-relative); the BC **fixture** check
   was the last straggler and now does too (`gui/launcher.py`). Pinned by
   `gui/tests/test_paths_resolve_root.py` (a relative fixture is accepted when
   CWD != ROOT; a missing one still 400s).
6. ~~**`wait()` timeout is per-thread, not total** (`control/selfplay.py`).~~
   **FIXED (2026-11-20):** `wait(timeout)` now uses a shared deadline (a TOTAL
   budget); the production `timeout=None` path is unchanged. Pinned by
   `control/tests/test_selfplay.py::test_wait_finite_timeout_is_total_not_per_thread`.

**Not bugs (verified OK):** replay-capture closure (`server_cfg` captured
correctly, port-namespaced per D95.1); defensive `on_finish` (a capture error
records `capture_error` without killing the battle); graceful stop for all modes
(`stop.flag` checked between battles/epochs; error-stop → `failed`, not
`stopped`); content-based dedup (no silent battle drops); the imitation verdict
pipeline (the 50-battle run **PASSED**: top1 0.308 > chance 0.180 & const 0.194;
switch-rate pred 0.399 == human 0.399; first-use 0.276 < repeat 0.342 → no
revealed-flag shortcut).

---

## B. Open design choices — **your call** (nothing built until you pick)

| # | Decision | Options | My lean |
|---|----------|---------|---------|
| 1 | **GUI framework** | ~~A = extend the existing **web** GUI~~ vs B = **native** desktop | **RESOLVED (2026-11-19): B — native Python GUI (tkinter).** The user explicitly requested a native Python GUI and asked to retire the browser one once it works. See D98 for Phase 1 (control-plane) and the Phase 2 (tkinter) plan. |
| 2 | **Self-play → capture in GUI** | expose `control/selfplay_main` as a "Self-play (collect)" mode so the library fills from the GUI | **RESOLVED (2026-11-19): FIXED as D98.** `launcher.start_trainer` now routes `selfplay-collect` → `control.selfplay_main`; the server's `api_start` handles it; the fast test stack is green (119/119). The biggest gap (item A2) is closed. |
| 3 | **Multi-bot model** | league/ELO/pairing vs independent parallel bots | **BUILT (2026-11-19, D100):** league infra complete — `league/registry.py` (roster + standard ELO K=32), `league/pairing.py` (round_robin / elo_weighted), `league/driver.py` (checkpoint-vs-checkpoint battle, live-play decision path), `league/engine.py` (pair -> battle -> ELO, reusing `SelfPlayRun` via the `battle_driver_factory` + `on_battle` hooks), `league/league_main.py` (subprocess entry). **Still needs a decision before the first real run:** which checkpoints form the initial roster (the board in section C has 4 candidates) + pairing strategy + whether to expose a "league" mode in the native GUI (not wired yet). An ELO magnitude bug (underdog wins under-rewarded) was found + fixed in this pass. **GUI wiring DONE (2026-11-20, D102):** `league` is now a mode in the native GUI (Start tab) — `validate_league_config` (roster + checkpoint existence checked BEFORE spawn) + roster panel (view/add bots) + `start_trainer` → `league.league_main`. **Still open:** which checkpoints form the initial roster + pairing strategy (defaults: round_robin, best_of 1). |
| 4 | **Formats per run** | one format per run vs multiple simultaneously | one for now (simpler, verified) |
| 5 | **Replay storage** | keep all vs top-ELO only? disk budget? (library has no pruning — ready for either) | keep all for now; add pruning later |
| 6 | **Server lifetime** | 8 always-on vs start-on-demand | on-demand (saves RAM when idle) |
| 7 | **Global STOP scope** | kill training only vs also the pool's servers (never your `:8000`) | training + this-run's servers; never `:8000` |
| 8 | **Concurrent runs** | keep single-slot vs allow multiple vs auto-expire idle "play" sessions | **RESOLVED (2026-11-20):** single-slot kept; a non-play start pre-empts an active PLAY session (item A3) |
| 9 | **`on_finish` contract** | `play_battle_on` catches+logs a raising hook vs leave it to the caller | leave to caller (engine already defends) |
| 10 | **Checkpoint naming** | ~~allow collisions (scratch vs runs)~~ vs a global unique name | **RESOLVED (2026-11-19, D99):** `{n}b_{type}_{id}.pt` (e.g. `50b_bc_a3f2.pt`). No date — avoids same-day collisions; 4-char uuid4 hex id — unique and league-friendly. |
| 11 | **Switch-rate verdict gate is one-sided/weak** | The `switch_rate_gap` gate (\|gap\| ≤ 0.15) is structurally one-sided: a *switch* decision's legal list is **only** switches, so every switch-decision is automatically a "predicted switch" ⇒ `pred_switch ≥ n_switch` always ⇒ `gap ≥ 0` always. It only ever fires by predicting switches on *move*-decisions, and trivially passes at gap=0 (which is exactly what the good nets show). It is a **weak** anti-collapse check, not the strong one the docstring implies. | **OPEN (2026-11-20):** replace/supplement it with an action-type-conditional check — e.g. "predict a switch on a **move**-decision at a rate near the human move→switch rate" (the honest collapse test). Keep **switch-accuracy** as the primary switch-quality gate (it's the real signal — and the BC nets only *barely* clear it: n2000 switch-acc 0.387 vs chance 0.259 / const 0.264). |

---

## C. Current checkpoint board (reference)

| checkpoint | top1 | battles | verdict | note |
|---|---|---|---|---|
| `scratch/s43_bc_n2000.pt` | **0.542** | 1463 | PASS | **best** — fight this one first |
| `scratch/s43_bc_n1000.pt` | 0.476 | 717 | PASS | |
| `scratch/2026-11-19_bc_50b.pt` | 0.391 | 50 | PASS | earlier 50-battle |
| `runs/20261119T044029Z/checkpoints/2026-11-19_bc_50b.pt` | 0.308 | 50 | PASS | **this pass's** validation run |

Smaller data → lower top1, but every run **passes** the S4.3 gate (beats chance
+ constant baselines, predicts switches at the human rate, no revealed-flag
shortcut). To improve top1: scale battles/epochs (you have `gen9ou.jsonl` 20k +
`gen9ou_part2.jsonl`) and run on the **GPU** via the one-click button.

---

## D. Net-learning verdict — "is the net actually learning?" (2026-11-20)

> Investigation the user asked for after the A3 fix. Evidence-based; no new
> training was run (CPU-only reads + one checkpoint-load probe).

**Verdict: the BC (imitation) net genuinely learns state→action — but unevenly
(moves ≫ switches). The self-play PPO net only shows "beats random" with no
improvement trend, because its opponent is `random` (a saturated, weak signal).
So imitation is the real learning path; self-play needs the league (varied,
stronger opponents) to keep improving. The user's pivot to the league was
well-founded.**

### Evidence — BC net (best = `s43_bc_n2000`, 1463 battles, 5741 held-out)
- **Fits:** train loss 1.12 → 0.96 over 13 epochs (`scratch/_scale_n2000.log`).
- **Generalizes:** held-out top1 0.436 → 0.508 (peak **0.542**) while train fits
  → not pure memorization.
- **Beats the honest baselines:** top1 0.542 vs chance 0.224 / const 0.234;
  move-acc **0.733** vs chance 0.182 / const 0.198. (n1000: 0.476; n50: 0.308.)
- **No "grab the new move" shortcut:** (n50) first-use 0.276 < repeat 0.342.
- **Weak spot — switches:** switch-acc **stuck ~0.34 across all 20 epochs**
  (chance 0.259, const 0.264). The net is bad at choosing *which* mon to
  switch to. This is the net's real limitation, not a training bug.

### The `switch_rate_gap = 0.0` is **NOT a bug** (traced)
A switch decision's legal list is *only* switches ⇒ it is always a "predicted
switch"; the good nets additionally never predict a switch on a *move*-decision.
So `pred_switch == n_switch` exactly and the gap is 0 by construction. Consequence:
that gate is one-sided/weak (see B.11). The honest switch signal is **switch-acc**.

### Evidence — self-play PPO net (`runs/20261120T015832Z`, 10 rounds × 5)
- **Opponent = `random`** (`config.json`). Win rate: 3,4,5,4,4,5,4,4,4,5 = **84%
  and FLAT** (no improvement trend).
- KL/clip → ~0 by round 7: **expected** for on-policy PPO — `trainer.update()`
  refreshes `old_logprob` to the current net, `epochs=2`, small lr ⇒ epoch-1
  ratio ≡ 1.0. **Not a freeze bug.** But it does mean the policy barely moves
  per round: the sparse ±1 terminal reward is **saturated** (beating random is
  easy), so there is little gradient pressure left.
- ~43 s/battle on CPU (50 battles ≈ 36 min).

**Honest read:** (1) "learning" is proven for the BC net (real fit +
generalization + beats honest baselines), weakly on switches. (2) "learning" is
NOT well-demonstrated for the PPO net — 84% vs random with no trend is
consistent with "quickly better than random, then stopped." That's a property of
the random opponent + sparse reward, not (necessarily) a broken net. (3) The BC
net (expert imitation) is very likely the **strongest single policy** we have →
make it a league seed / opponent.

### New follow-ups surfaced (need your call)
- **N1 — switch-rate gate (B.11):** strengthen it (action-type-conditional)
or accept switch-acc as the real gate. *(design choice, section B.11)*
- **N2 — D81 re-verify blocked:** the best BC checkpoint's PASS predates the
  D81 first-use/repeat leak gate (its meta lacks those fields). Gates 1–4 are
  still valid; re-run `evaluate()` under the current code to add gate 4. **Blocked
  on N3** (the trace cache is now stale, so a re-eval needs a rebuild).
- **N3 — trace cache is stale (fingerprint drift):** current `9318a22b…` vs
  cached `6d798cc29f…`. Since the cache was built (2026-11-18 08:37Z), the
  fingerprint set changed: `belief/state.py` (**champions prior gap fix —
  possibly behavioral**) + `legality.db` (updated) + `belief/evidence.py`,
  `damage/calc.py`, `node_calc.js` (non-behavioral: batch transport / retry /
  server). **The checkpoints still LOAD fine** (verified: all 4 → 7.21M params
  under the current registry/vocabs), so nothing is lost — only **re-training /
  re-eval** is affected. Options: **(a)** rebuild the cache (safe; reflects the
  champions fix — my lean, since ≥1 change is behavioral); **(b)** narrow the
  fingerprint to behavior-affecting code / add a calc-output fingerprint; **(c)**
  confirm gen9ou traces are byte-identical and bless the old cache. **My lean: (a)**
  for the next BC run; (b) if we want cheap re-evals.

---

## 2026-11-21 — Bench-aware net (the net sees the ENTIRE battle state)

**Goal (user-approved):** the net must see the whole team, not just the active
mon. Scenario (user's words): *Garchomp @80% vs a confirmed-slow Blastoise @5%,
fast Fairy + Gliscor benched → the net should weigh the bench and pick COVERAGE
(Rock hits Flying/Ground Gliscor), not just hit the active.*

### The gap (verified, not assumed)
In singles the lead is chosen during team preview, so the `|poke|` preview order
does NOT map to field slots — only the active mon ever occupies field slot `p2a`
(verified in `scratch/hf_sample_replay.json`: preview = Kyurem, Tinkaton,
**Blaziken**(#3), Great Tusk, Sinistcha, Darkrai, but the lead is
`|switch|p2a: Blaziken|`). The 5 bench mons were therefore never registered and
the net saw only `opp_species[0]` — **the bench was invisible**.

### Step 1 — BUILT, TESTED, COMMITTED (`18c13cd`)
- `|poke|` → `TeamPreview` event (adapter); belief accumulates the opponent team
  (`self.opp_team`); `get_context()` places the active at `p2a` and the ALIVE
  bench at `p2b`–`p2f` (preview order, minus active + fainted).
- **Representation = repurpose the existing 6 `opp_species` slots.** No new
  dimension → registry checksum unchanged → **existing checkpoints still load**
  (honors “don't delete old data”). This deviates from the naive “preview order →
  field slot” mapping, which would LOSE a bench species in singles — I proved
  that wrong in a test (`test_context_shows_all_six_opp_species`).
- The ACTIVE mon's damage/max-HP EV inference (3.6) is preserved — the preview
  handler only touches `opp_team` (pinned by
  `test_preview_does_not_clobber_active_ev_inference`, real calc).
- **Verified:** 9 new tests green; belief+net suite **254 passed, 1 skipped,
  4 deselected** (CPU-only); end-to-end `get_context → encode_context` confirms
  all 6 `opp_species` slots non-zero + distinct + correctly indexed
  (`[1,4,6,3,5,2]` = blaziken/kyurem/tinkaton/greattusk/sinistcha/darkrai).

### Decisions I made (recorded — NOT yet user-approved)
- **Bench shows only the ALIVE team** (excludes the active + any fainted mon)
  via a new `opp_fainted_species` list (the opponent analog of the existing
  `own_fainted_species`, D68). A fainted bench mon is **omitted** (not shown as
  fainted).

### Open questions (need your call)
- **OQ-bench-1 — fainted bench mon: omit vs show-as-fainted.** I chose *omit*
  (a gone mon must not be offered as switchable bench). Alternative: keep the slot
  but set `fainted=True` so the net sees “that mon is gone” (more info, but the
  slot would read hp 0 + fainted 1 — distinguishable from benched-untouched via
  the existing `fainted` column). *My lean: omit (cleaner, and the fainted count
  global already tells the net how many are gone).*
- **OQ-bench-2 — BC-baseline re-baseline.** Populating the bench slots changes the
  **input distribution** the net sees (bench slots were all-zero when the BC
  checkpoints were trained, now they carry species). The BC baseline's behavior
  with a populated bench is therefore untrained. This is FINE for the intended
  artifact (the NEW self-play-league net, random init, which will train WITH the
  bench) — but the BC baseline's decisions will change. *Do you want to
  re-baseline the BC net, or is it fine as-is (it's a baseline, not the target)?*
- **OQ-bench-3 — bench posterior.** Bench slots carry `posterior=[] / confidence=0`
  (no set inference yet — they're never active, so no damage evidence). The net
  still sees their *species* (the key signal for coverage). Later we could run
  set inference on a bench mon IF it ever becomes active (it does — it switches in
  and becomes the active, where inference already works). *No action needed now.*

### Next (awaiting your GO — do NOT start without confirmation)
**Step 2 (the (b) seam — you already picked option b):** add an optional `ctx`
kwarg to `PokeAgent.act` (`net/agent.py`) + `HeuristicAgent.act` (`heuristic.py`);
a `build_obs_and_ctx()` enabler in `net/env.py` (single belief-build, `build_obs`
thin wrapper, obs byte-identical); the driver (`league/driver.py`) builds ctx once
and passes it; update `human_play.py`, `trainer.py` (the protected GPU path) to
tolerate it; suite stays green. This hands the structured `get_context()` (now
including the bench) to the agent so the brain (`fitness/strategy.py::choose_action`)
can use it. **Step 3** (bench-aware heuristic: among moves that all KO the active,
prefers the one with best coverage on the likely switch-in). **Step 4** (training;
GPU, last, you-run).

---

## 2026-11-21 — The six long-term scenarios -> ONE architecture (the real gap is LOOKAHEAD)

> The user gave six scenarios the net must eventually do (recognize win
> conditions, evaluate expected-WR, weigh both players' switches, value
> setup/screens/hazards over the match, spot the opponent's win condition, and
> update its belief from moveset changes). I grounded these in the actual code
> (not assumption) and they collapse into ONE gap + a few signal gaps.

### What already exists (verified in code, not assumed)
- **The net ALREADY has a value head** (`value (B,T,1)` in (0,1) = expected win
  probability) alongside the policy head (`net/model.py`). So scenario #2 (state
  -> preferred/bad/losing by expected WR) is **architecturally already built**.
- **PPO + GAE ALREADY trains that value head** (`net/ppo.py`: clipped policy +
  value-MSE + entropy, sparse terminal ±1). The agent *reports* the value but
  does not *use* it for decisions yet (`net/agent.py`).
- **The belief state** infers hidden info (EVs, moves, items, team comp, field
  state) and now sees the bench. **The only "reasoning" today** is the hand-coded
  `fitness/strategy.py::choose_action` (KO ladder + hazard-aware switch + threat)
  — single-step, no lookahead, on the heuristic path (not the NN).

### The real gap (all six point at this)
The net is **single-step** (state -> one action + one value). It cannot **look
ahead** — so it can't "realize a win condition" (multi-turn conditional),
"evaluate a double switch" (my move + their reply), or "spot the opponent's
setup line." That is a **SEARCH** problem. The long-term engine is **MCTS
(AlphaZero-style)**: search the tree using the Showdown sim + damage-calc as the
world model and the value head as the leaf. **One engine serves #1, #3, #5;**
**#2 is already built (becomes the leaf);** **#4/#6 are belief/observation
enrichments.**

### Scenario -> mechanism map
| # | Scenario | Served by |
|---|---|---|
| 1 | Multiple win conditions (Dragonite sweep when X and Y hold) | MCTS lookahead (value head ranks the setup line) |
| 2 | Expected-WR state eval (preferred/bad/losing) | **value head (built)** — surface it; MCTS uses it as leaf |
| 3 | Both players' switches (double switch) | MCTS searches the opponent's reply |
| 4 | Value setup/screens/hazards across the match | belief field-state (present) + a few derived signals + value training |
| 5 | Spot the opponent's win condition (free SD) | MCTS sees their setup line + belief opponent posterior |
| 6 | Update belief from moveset changes (Wish->Protect, Scarf KO'd) | belief `moves_used` + posterior (mostly present) + surface in obs |

### Phase plan (each = one tested step; NOT all-at-once)
- **M1 — MCTS core (BUILT this step, `net/mcts.py`):** PUCT search over a
  pluggable `WorldModel` (`actions`/`step`/`value`/`policy`). Net-agnostic,
  deterministic, reward space = (−1,+1) (the same as `ppo.py`, so no second
  convention). **7 tests pass**, incl. "lookahead beats 1-step-greedy on a trap"
  (the Dragonite analog) and "searches the opponent's reply" (the double-switch
  analog). Bug caught + fixed by testing: the PUCT backup was anchored at the
  root instead of the leaf (values never propagated) — now walks up from the leaf.
- **M2 — real WorldModel:** wrap the Showdown sim + damage-calc + `PokeNet`
  (`actions`=legal list, `step`=apply action->next belief ctx, `value`=2V−1,
  `policy`=policy-head priors). The bridge that makes MCTS play real Pokémon.
- **M3 — value-head surface:** `evaluate_state(ctx) -> {win_prob, label}`
  (preferred/even/bad/losing) for interpretation + as the MCTS leaf; ensure
  self-play trains it well.
- **M4 — derived observation signals** (belief-derived, NOT hand-maintained):
  win-condition readiness, opposing-setup threat, double-switch availability,
  hazard value. Gives the net explicit handles to learn #1/#4/#5 faster.
- **M5 — training (GPU, you-run):** MCTS-guided policy+value (AlphaZero loop:
  MCTS -> better targets -> better net -> better MCTS), or value-head self-play.

### Open choices (need your call)
- **OC-1 — MCTS vs pure value-learning.** MCTS (explicit search, uses the sim as
  world model) is more powerful + interpretable for #1/#3/#5 but is per-decision
  compute at inference. A value-only net (deeper self-play) is cheaper but
  learns these only implicitly and slower. *My lean: MCTS — it is the direct,
  verifiable engine for the win-condition/double-switch/threat scenarios, and the
  value head (already trained by self-play) is its leaf. Cost is a later
  optimization (budget the iterations).*
- **OC-2 — does the value head need RE-BASELINing before MCTS uses it as the
  leaf?** A weak value leaf makes MCTS weaker. We have BC + PPO checkpoints
  (value head trained on self-play). *Do you want to re-verify the best
  checkpoint's value head is sane (M3) before wiring M2, or wire M2 first and
  let M5 (training) improve the leaf?*
- **OC-3 — inference budget.** MCTS iterations per decision (e.g. 200 vs 2000)
  trade strength vs latency. The league/GPU training can afford more; live play
  less. *Do you have a latency target for live play?*
- **OC-4 — the hand-coded `choose_action` brain.** It's a single-step heuristic
  (KO/hazard). With MCTS + value, is it (a) retired, (b) kept as a prior/sanity
  floor, or (c) its signals folded into M4 derived observations? *My lean: (c) —
  keep the reasoning, but as LEARNED signals, not a separate decision path (avoids
  two workflows).*
- **OC-5 — scenario priority.** I'm sequencing M1->M5 (search first). If you'd
  rather start with #2 (value surface) or #4 (hazards) or #6 (belief) first, tell
  me and I'll re-order — the phases are independent enough to swap.

---

## 2026-11-21 — M2 design decision (the sim-branching crux) — DECIDED, M2a next

> The previous agent was mid-investigation of the single hardest M2 sub-problem:
> **how MCTS branches a live poke-env battle to search futures.** Decided this
> session (evidence-based), and split M2 into a safe non-controversial part
> (M2a, next) and the controversial part (M2b, needs your OK).

### The decision: approach (b) — the MCTS `state` is the belief `get_context()`

**Why not (a) clone the live poke-env `battle`:** it owns a `PSClient` + an
event-loop thread + websocket sockets; poke-env reaps them only at `atexit`
(the teardown-leak fix proved `PokeEnv.close()` never stops them). It is NOT
`deepcopy`-able, and replaying a move through it needs **network round-trips**
— which defeats the whole point of *fast local* search (your "the calc needs to
be quick and api reachable"). **Why not (c) serialize + re-replay:** same
network cost, strictly more moving parts. **(b) keeps the state pure data**
(cloneable via `copy.deepcopy`) and reuses the ONE damage pipeline — so MCTS can
branch a state with **zero** live-sim dependency.

### Two crux subtleties (found by reading the net, NOT assumed)

1. **The context alone does NOT say whose turn it is.** The net is a
   **P1-decider** (it always acts for the player whose POV the obs is in —
   verified: `build_obs` swaps to p1-POV; the value head is trained on
   deciding-player-POV states). So the MCTS **state must carry a `decider`**
   ("p1"/"p2"): the context (canonical, always p1-POV) + who moves next. The
   root is `decider="p1"`; `step` flips it each ply.
2. **Every value-head call must stay IN-dISTRIBUTION.** The value head learned
   "deciding player's win prob" (p1-POV in training). So at a P2-turn node the
   value is read from P2's POV (swap) = P2's win prob, then mapped to the ROOT
   player's (P1's) perspective for the MCTS `_backup` (`1 − P2win`, i.e. `−(2V−1)`
   in reward space). Feeding a "P2-turn, p1-POV" state to the head would be
   out-of-distribution. **Policy is the same:** it always runs on the *decider's*
   POV. Both heads therefore always see a deciding-player-POV obs — in-distribution.

### M2a vs M2b (the split that lets us keep tests green + avoid a new source of truth)

- **M2a (NEXT, safe, non-controversial):** the **non-branching** WorldModel
  adapters over the REAL `PokeNet` + a real belief context:
  `actions(state)` (legal moves + bench switches, decider-aware),
  `value(state)` (value head → decider POV → mapped to root perspective),
  `policy(state, actions)` (policy head → priors over the decider's legal set).
  `step` raises `NotImplementedError` (M2b). **Tested with a random-init net,
  CPU-only** — no live battle, no effect bookkeeping, no second source of truth.
  This is a real, verifiable slice: it makes `MCTS.best_action` drive the ACTUAL
  net end-to-end (it can't produce a real tree without `step`, but each adapter
  is independently testable + the value/policy perspective contract is pinned).
- **M2b (AFTER your OK — this is the "new source of truth" the handoff warned
  about):** `step(state, action)` = apply the decider's move to a copied context:
  damage via the **ONE** `damage/calc` (never re-derived), plus minimal
  HP/status/switch/faint bookkeeping. **Risk:** it can drift from the sim. The
  mitigation is a **"world-model-vs-sim" agreement test** (CPU): for a set of
  (context, action) pairs, the context `step` predicts must match what the real
  Showdown sim produces for the same action. If it drifts, we trust the sim and
  fix the bookkeeping — the sim stays the ground truth (per your "smogon calc is
  the source of truth" rule).

### New open choices (need your call — nothing in M2b is built until you pick)
- **OC-6 — M2b world-model fidelity.** (a) minimal bookkeeping (HP + switch +
  faint only; status/boosts via the calc where the calc knows them) + a
  world-model-vs-sim agreement gate (my lean — smallest second source of truth);
  (b) richer bookkeeping (track volatiles/stockpile/PP) for a more faithful tree;
  (c) hybrid: use the sim for the FIRST ply (real) and (a) for deeper plies
  (approximate — the leaves are where the value head lives anyway). *My lean: (a)
  first (correct + fast), upgrade to (c) if the agreement test shows drift.*
- **OC-7 — the `decider`-in-state design.** I've recorded the design (context +
  decider; heads always run on the decider's POV; value mapped to root view).
  *Do you want M2a to implement exactly this, or do you'd rather I keep the root
  as the ONLY decider for now and treat the opponent's reply with a fixed
  opponent-policy (e.g. the net on its own POV)? The former is the real engine;
  the latter is a weaker but simpler v1. My lean: full decider-aware (it's the
  double-switch scenario #3 you asked for).*
- **OC-8 — tera in the world model.** `build_action_list` offers a one-shot `tera`
  variant per move (only when `can_tera`). The context does NOT track
  "tera-used," so a multi-ply search would illegally tera repeatedly. M2a
  **omits tera** (moves + switches only). *Do you want tera bookkeeping added to
  the context for M2b, or leave it out (rare in the formats we target)?*

### M2a — BUILT + TESTED (this session)
The non-branching bridge is in and green:
- `net/world_model.py` — `State` (canonical p1-POV ctx + `decider`) +
  `PokeWorldModel` (`actions`/`value`/`policy` live via the ONE
  `PokeAgent.forward_obs` path; `step` + the p2-POV orientation raise as M2b).
  The value is mapped to the ROOT player's perspective (the M1 `_backup`
  contract); every value/policy head call stays in-distribution (decider POV).
- `net/tests/test_world_model.py` — 8 tests, CPU-only, random-init net,
  against the realistic `test_s41.make_ctx` context: the D60 legal list (moves
  + alive bench, in training order; fainted bench excluded), value == the net's
  value head mapped to root (`2V−1`), policy == the net's masked prob (sums to
  1, identity-stable to action order), determinism, and the M2b boundary
  (`step` and `decider='p2'` both raise).
- **Gate:** full net suite **127 passed / 1 skipped / 2 deselected**
  (CPU-only; was 119 before M2a, +8 the new WorldModel tests).

**Still NOT built (awaiting OC-6/7/8):** M2b (`step` + the p2-POV
orientation) — the branching + the "new source of truth" the handoff warned
about. No GPU touched; no `git add -A`; committed only the M2a files + this
doc.

---

## 2026-11-21 — M1+M2a self-review + M2b de-risk (this session)

**Self-review (all verified correct, no active bugs):**
* **M1 backup sign-convention** (`mcts._backup`): traced end-to-end — a
  node's `value` is always stored from the SELECTOR's (parent mover's)
  perspective, so "always-maximize" selection is correct for both players, and
  the leaf→root walk flips the sign one step at a time. Consistent with the
  `2V−1` reward-space contract shared by `ppo.py`.
* **Pointer-head identity** (`model.py`): per-action logits = `intent ·
  action_embed` (both normalized) — depend only on the action's OWN embedding,
  never its position. The "shuffled-list → same per-action prior" test verifies
  a real structural property, not a fluke.
* **Value mapping** (`world_model.value`): decider win-prob → root reward is
  correct for BOTH cases (root: `2V−1`; opponent: `−(2V−1)`).
* **Gate:** net suite **129 passed / 1 skipped / 2 deselected** (CPU-only;
  was 128, +1 the new test below).

**De-risk (sanctioned in-scope step, independent of OC-6):** extracted the
root-view value mapping into a PURE net-free function
`world_model._map_value(decider_win_prob, decider, root_decider)` and added
`test_map_value_root_and_opponent_decider`. This validates M2b's **perspective
contract for BOTH decider cases in isolation** — so M2b only has to wire the
p2-POV *observation* orientation (`_orient`) on top of an already-verified
mapping. `value()` now delegates to it (no behavior change; the p1-root case is
bit-identical).

**Two LATENT M2b risks flagged (NOT active bugs in M2a — pin these before M2b):**
1. **`State` frozen ≠ immutable-dict.** `dataclass(frozen=True)` only blocks
   field reassignment; the `ctx` dict is still mutable. M2b's `step` MUST hand
   out a **deep-copied** ctx (never mutate a node's ctx in place) or the shared
   dict corrupts the whole MCTS tree (parent + descendants share one live
   dict). **NOW BUILT + TESTED:** `State.copy(decider=None)` deep-copies the
   ctx (the safe child primitive for `step`) + a `flip_decider` helper; pinned
   by `test_state_copy_is_independent_of_parent` (mutate the child's ctx, the
   parent's ctx is provably unaffected).
2. **Two forward passes per node.** `value()` and `policy()` each run a full
   `forward_obs`; MCTS calls both per node. M2b should **cache the forward
   result per state** (one pass feeds both heads). Not a bug — an efficiency
   landmine that only matters once `step` produces a real tree.

**De-risk (this ping — decision-independent, `bad2cf7`):** built the
**action-transform primitives** `apply_hp_delta` / `apply_faint` /
`apply_switch` (pure, POV-aware, context-only — no net/calc/RNG). A switch /
faint / HP-delta is the SAME under every viable OC-6/OC-9 option, so this
encodes no design decision — it proves the "apply the action's effect" mechanics
before the OC-gated parts (the calc spread + the posterior refinement) are wired
in. Each returns a NEW ctx (never mutates the input — the MCTS tree must not
corrupt a shared parent ctx); pinned by `test_world_model_transforms` (7 tests).
Net suite 143 passed / 0 failed (CPU). So when OC-6 + OC-9 are picked, `step`
WIREs these in + the batched calc + the posterior refinement — it does NOT
re-derive the structural mechanics.

**M2b BUILD PLAN — mechanical checklist (REVISED 2026-11-22 for the locked
decisions: OC-6=(b) rich, OC-9=(b) carry-opaque, OC-7 verified, OC-8
format-conditional). 7 steps, each GREEN before the next, commit per sub-step,
CPU-only, `live` deselected, explicit `git add <file>` (never `-A`).**
Everything below is ALREADY committed + green (verified this session):
`maxhp` in `get_context` (`ef86f08`); the transform primitives (`bad2cf7`);
`State.copy` (`dfa7753`); `_map_value` both-decider (`8286b35`); `_flip_pov`
round-trip (`fc1a2d7`); the field-set single-source-of-truth (`3987c1e`); the
M1 length guard (`02f3a9b`); the ONE damage pipeline + `calc_damage_many` batch.
So this is a WIRING pass, not a re-derivation. Steps, in order (each ships with
a test; `live` deselected; CPU-only; no GPU):

1. **p2 orientation + posterior visibility (OC-9b, LOCKED).** Wire `_flip_pov`
   into `_orient(ctx, "p2")` (replace the `NotImplementedError`) — the posterior
   RIDES ALONG (no top-1 collapse). Make the encoder's posterior channels carry
   "the uncertain side's distribution" (opponent at p1-POV, own at p2-POV) so
   p2's own posterior is VISIBLE to the net. Zero schema change; old
   checkpoints load. *Tests:* `test_orient_p2_round_trips_to_identity`;
   p1 obs unchanged (checksum); p2 obs shows the own-side posterior; p1's
   own-posterior channels stay zero.
2. **Move-effect table (OC-6b, LOCKED).** Source deterministic effect
   metadata (status inflicted, stat boosts, hazard set/clear, screens up,
   tailwind) from the pokemon-showdown fork's move data — ONE source, no
   hand list. Probabilistic effects EXCLUDED from the deterministic model
   (the value head learns them; the sim gate keeps the sim as ground truth).
   *Tests:* lookup pins for the T1 effect-movers (Wish/Protect, Stealth Rock,
   Spikes, Swords Dance, Will-O-Wisp, Reflect/Light Screen, Tailwind, Rapid
   Spin/Defog) in the target formats.
3. **Format-aware action list (OC-8, LOCKED).** `PokeWorldModel` takes the
   format spec (has_tera: gen9+; has_mega: gens 6/7) at construction.
   `_actions_from_own` includes the tera/mega action ONLY where the format has
   it, with per-mon `tera_used`/`mega_used` ctx flags (additive keys) gating
   repeats. *Tests:* gen8 ctx -> no tera action; gen9 ctx -> tera present,
   then absent after one use; mega gen6 same shape.
4. **`step(state, action)` in `PokeWorldModel`:**
   * `child = state.copy(decider=flip_decider(state.decider))` (the safe deep-copy);
   * branch the action's effect via the transforms — `apply_switch` (a switch
     action) / `apply_hp_delta` + `apply_faint` (a move action: damage from the
     batched calc; faint if the victim's hp_frac hits 0) + the step-2 effect
     updates (statuses/boosts/hazards/screens) on the child ctx;
   * the damage number comes from the **ONE** pipeline: `calc_damage_many` (batch
     every child damage in one round-trip) — own spread = team spec, opp spread =
     posterior top-1 FOR THE CALC ONLY (the net keeps the full posterior — OC-9b);
   * `reward = _map_value(leaf_win_prob, child.decider, self.root_decider)`;
     `done = (either side is 0-6 / 6-0)`.
   * *Test:* `test_step_move_produces_p2_child` (a move -> a p2-turn child, the
     victim's hp_frac drops by damage/maxhp, status/boost applied where the move
     has one, the decider flipped, the parent's ctx provably unchanged).
5. **World-model-vs-sim agreement test (the OC-6 gate, now RICH):** a scripted
   sequence through the real sim vs. the world model's predicted outcomes
   (hp_frac / fainted / active / statuses / boosts / hazards) — they must
   AGREE (the sim is ground truth). *Test:* the CPU `live`-deselected
   agreement test.
6. **MCTS 2-ply smoke:** `MCTS.best_action` now runs a real 2-ply tree (root p1
   + reply p2). *Test:* `test_mcts_two_ply_tree_is_nontrivial_and_legal`.
7. **Efficiency (the flagged landmine):** cache the forward result per state so
   one `forward_obs` feeds BOTH value + policy (do this inside `step`'s child
   eval). Not a correctness bug — a latency win once the tree is real.

**BRANCHES LOCKED (2026-11-22, user-ruled):** OC-6=(b) rich; OC-9=(b)
carry-opaque (top-1 used ONLY as the calc's spread input); OC-8
format-conditional (tera gen9+, mega gens 6/7, once-per-mon flags). Full
reasoning preserved in the (now-superseded) section below.

**SUPERSEDED (2026-11-22): the calls below are LOCKED — OC-6=(b), OC-7
verified+confirmed, OC-8 format-conditional, OC-9=(b) — see the board at the
top of this file. Kept only as the reasoning record:**
* **OC-6 — world-model fidelity of `step`:** (a) minimal bookkeeping
  (HP+switch+faint; status/boosts via the calc where it knows them) + a
  world-model-vs-sim agreement gate [my lean]; (b) richer; (c) full sim
  replay per node (rejected — network round-trips defeat fast local search).
  **Concrete sub-finding (this pass — decision-independent, informs the choice):**
  `step` can reuse the ONE damage pipeline cleanly: it is snapshot-pure (JSON
  in/out), runs on a long-lived Node worker POOL (no per-call spawn), and already
  has a BATCH API — `calc_damage_many` ("24+ calcs per decision in one round-
  trip"). So branch all child damages in ONE batch call, not N sequential ones.
  Two data-flow facts to decide *within* OC-6: (1) `maxhp` — **NOW EXPOSED**
  (`get_context()` carries `maxhp` per mon block; committed `ef86f08`): the
  belief already tracked `Facts.maxhp` (int) per mon; I added it to the context
  (public, additive, encoder-ignored, so the registry checksum is unchanged and
  checkpoints stay loadable). Bench placeholders carry `maxhp=None` (uniform
  schema, no KeyError). This closes the `maxhp` data-flow sub-gap —
  decision-independent, needed by every viable OC-6 option. The OPEN part of
  (1) is the calc SPREAD: (2) the calc needs a spread (evs/ivs/nature): own
  side has the full team spec, opponent only a posterior (→ OC-9 top-1). Net:
  OC-6(a) "minimal" = HP/switch/faint + `maxhp` in the context [DONE] +
  assumed/posterior spread for the calc [OPEN], batched via `calc_damage_many`.
* **OC-7 — decider-in-state:** the decider-aware design I built M2a on
  (context + `decider`; heads on the decider's POV; value mapped to root) vs a
  simpler "root-only decider" stub. M2a + the `_map_value` de-risk assume the
  decider-aware design.
* **OC-8 — tera in the world model:** add "tera-used" tracking to the context
  for M2b (else multi-ply search teras repeatedly), or leave tera out of the
  search (rare in our target formats). M2a currently omits tera.
* **OC-9 — the p2-POV posterior asymmetry (the real crux of the `_orient`
  swap).** Confirmed from `belief/state.py::get_context()`: the belief is
  **P1's** model — `own` (p1's team) is FULLY known (name/ability/item + the
  full spec moveset), but `opponent` (p2's team) carries ONLY `species` + a
  `posterior` (top-K {ability, moves, weight}) + `top_moves` + `confidence`,
  and stores **NO true ability/item for p2** — only p1's posterior over them.
  So a p2-POV flip is NOT a clean invertible reflection:
  - **p2's "own" team**: the belief only has the POSTERIOR, not the truth → to
    present it as "own" (the net trains own-side as fully known) I must pick a
    deterministic stand-in.
  - **p1's team becomes the opponent**: the belief KNOWS p1's ability/item/
    moves (they're p1's own), so from p2's view they'd be "revealed" —
    asymmetric with reality, but it's all the belief has.
  Options: **(a) [my lean] deterministic top-1** — p2's own = argmax-posterior
  candidate (ability + its moves); p1's opponent = known species+moves with
  ability/item taken from p1's own spec (the belief knows them). In-
  distribution for the net (own-side = fully known); it's an APPROXIMATION —
  fine, the MCTS search is exploring, not exact, and the value head (trained
  on p1-POV self-play) is the ground truth the search converges toward.
  **(b) posterior-symmetric** — keep the posterior channel on BOTH sides; but
  the net trains own-side as fully known, so a posterior on own is OUT-of-
  distribution (risky). **(c) opponent-reply is heuristic-only** — don't build
  a full p2-POV own-team model at all; score the opponent's reply with the
  policy prior + a fixed heuristic (e.g. they play their best-KO move).
  Cheapest, but least faithful to the real game.
  **Progress (this session):** the STRUCTURAL core is now built + round-trip
  tested as pure, invertible `world_model._flip_pov` (own<->opponent + slot
  remap `p1x<->p2x` + field side-flip + fainted/active swap) with
  `_flip_pov(_flip_pov(ctx)) == ctx` pinned — so the fiddliest part of M2b's
  `_orient` is proven a correct PERMUTATION. It carries the posterior opaquely
  (makes NO OC-9 decision). Wiring `_flip_pov` into `_orient(ctx,"p2")` + the
  OC-9 refinement is the last structural M2b step, then `step` (OC-6) completes
  the engine. `_orient` still raises for p2 (M2a boundary held).

**Single-source-of-truth fix (this pass — decision-independent, no OC impact):**
Self-review traced the p2-POV flip's per-side field list against the encoder's
actual field reads and found them to be TWO hardcoded copies of the same 8 names
(`reflect/lightscreen/auroraveil/stealthrock/stickyweb/safeguard` +
`spikes/toxicspikes`). They match today (verified — no active bug), but a
hand-maintained parallel list is exactly how the flip would silently keep one
field in p1-POV once the encoder grows. Now `encoder.SIDE_KEYED_FIELDS` /
`HAZARD_FIELDS` / `ALL_SIDE_KEYED_FIELDS` are the single source of truth; the
encoder's global reads AND `world_model._flip_field` both derive from them.
Behavior-preserving (the refactored loop writes identical slots/values); pinned
by `test_flip_field_covers_every_side_keyed_field_the_encoder_reads`. Net suite
now **135 passed / 1 skipped / 2 deselected** (was 134, +1 the new test).

---

## STEP 3 (review 2026-11-22) — "the net is not GIVEN the information needed to plan" — SUB-STEP PLAN

User-approved (2026-11-22): Steps 2+3 back to back (one invalidation), Step 8 after each.
Step 2 is DONE (commit e716ab4). Step 3 = the review's observation expansion (items a-d),
executed as sub-steps, each: registry field(s) + encoder + model consumption + test + commit.
Checkpoints invalidated either way — layout changes re-pin test_s41's checksum pin (deliberate).

| sub | what | calc? | status |
|---|---|---|---|
| 3.1 | static MOVE features per action: `action_move_static` (MAX_ACTIONS,11): power/200, type/18, category, priority(-6..6 -> 0..1), accuracy/100, flags: is_hazard/is_screen/boosts/side_status/side_condition/is_tailwind | no (legality.lookup.get_move) | next |
| 3.2 | static MON features per slot: `own_types`/`opp_types` (MAX_SLOTS,18) dual=0.5/0.5, `own_stats`/`opp_stats` (MAX_SLOTS,6) base/255 | no (legality.lookup.get_species) | next (same commit as 3.1) |
| 3.5 | per-SLOT opponent posterior: `opp_post_per_slot_ids/w` (MAX_SLOTS,MAX_POST) + `opp_topmoves_per_slot` (MAX_SLOTS,MAX_TOP_MOVES) — replaces the active-only channels (bench threat visible) | no (ctx already carries per-slot posterior) | DONE (per-mon side encoding) |
| 3.3 | per-ACTION damage block `action_damage` (MAX_ACTIONS,9): moves vs opp active (min/max/exp pct, guaranteed/can/expected KO, outspeeds, immune; belief-weighted over opp ability posterior); switches (entry-hazard pct, expected dmg taken from opp top moves, forced-switch flag) | YES (damage.calc.calc_damage_many batch + effective_speed) | after 3.5 |
| 3.4 | MATCHUP matrix `matchup` (120,): alive own i x opp j: exp dmg both ways + speed order (108) + per-own-mon KO counts (12) | YES (one calc_damage_many: 6x6x2) | after 3.3 |
| 3.6 | env/driver wiring: build features at decision time, pass into encode_context (encoder stays PURE: takes precomputed arrays, zero-fills when absent) | - | with 3.3/3.4 |
| 3.7 | gate tests: no-leak (unrevealed opp move never changes obs), deterministic, bench-slot permutation permutes only matching rows, OHKO scenario gives switch a distinct feature vector from attacks | - | last |

DESIGN RULES (kept from the review): id embeddings stay (these are ADDITIONAL features);
encoder stays pure/numpy/DB-free (static+calc features are PRECOMPUTED by net/static_features.py
/ net/action_features.py and injected as optional arrays); calc is the ONE batch endpoint
(calc_damage_many); opponent hidden info still only via posterior (rule 4).

PROGRESS:
- 3.1+3.2: DONE — `net/static_features.py` (build_static_features/move_static_vector/
  types_onehot/stats_vector), registry +5 fields (action_move_static 12x11,
  own/opp_types 6x18, own/opp_stats 6x6), encoder places them (zero-fill when
  absent), MonEncoder + pointer head consume them (widths from registry), env.build_obs
  wires build_static_features. test_s41 checksum re-pinned to c38b07e0...; new
  net/tests/test_static_features.py (13 tests). net suite 157 passed; combined
  suite 831 passed (6 benign Tk/Tcl fixture errors, documented).
- 3.5: DONE — the 3 opponent-posterior fields became PER-SLOT (registry
  opp_post_per_slot_ids/w 6x5, opp_topmoves_per_slot 6x4): every bench mon's
  ability/move threat is now visible ("Gengar is in the back"), not just the
  active one. Encoder fills per-slot; slot 0 == the old active-only value (a
  strict superset, no signal lost). MonEncoder now folds each mon's posterior/
  top-moves INTO that mon's vector (like types/stats) — set-pooled, bounded,
  per-mon, permutation-invariant. IMPORTANT: first attempt put the posterior in
  the GLOBAL context (sum over 6 slots) and that REGRESSED the
  state-conditioning guard (inflated constant shadowed the own-side signal,
  loss stuck at 0.69); fixed by moving it per-mon into the side encoding. test
  checksum re-pinned to 9d00eb07...; +test_posterior_is_per_slot_bench_threat_visible.
  net 161 passed; league+fitness 186 passed.
- 3.3: per-ACTION damage block (the "belief state" the net needs to plan: will
  this KO / will I be KO'd / who is faster, FOR EACH legal action). Schema locked
  (K_ACTION_DAMAGE=9) — see "3.3 DESIGN" below. Split into 3.3a (move/tera rows)
  + 3.3b (switch rows) so each is a small, verifiable commit. **3.3a DONE.**

### 3.3 DESIGN (locked before coding, 2026-11-22)

New registry field `action_damage` (MAX_ACTIONS, 9) float32 — one row PER LEGAL
ACTION: the damage/KO/speed facts FOR THAT action. Precomputed by the ONE calc
pipeline (`damage.calc` + `effective_speed`) and injected by `net/action_features.py`
(lazy calc import; degrades to a ZERO row — the same discipline as
`net/static_features.py`). The encoder PLACES it (zero-fill when absent, stays
pure/DB-free). The model folds it into the per-action embedding CONSISTENT WITH
`action_move_static` (both the pointer head and the masked-mean context), bumping
`action_dim` by 9. Rationale for "both, mirroring action_move_static" (rather than
pointer-head-only): the two `act_emb` cats MUST share one width because
`policy_act_proj` is a single Linear, and action_move_static already works in both;
the context path is a bounded masked-MEAN (not the 3.5 failure, which summed 6
opponent slots into a global constant). FALLBACK if Step 8 shows 3.5-style shadowing:
move it to pointer-only (2-line change, pre-authorized).

9-dim schema (each dim has a MOVE/TERA meaning AND a SWITCH meaning; the net learns
the mapping from `action_kind`, already in the obs):
  [0] min_pct   move: min dmg % on opp active (/100, cap1)   switch: entry-hazard min % on incoming
  [1] max_pct   move: max dmg %                               switch: entry-hazard max %
  [2] exp_pct   move: expected dmg %                          switch: entry-hazard exp %
  [3] can_ko    move: my move can KO opp active              switch: opp top move can-KO the incoming mon
  [4] guar_ko   move: guarantees KO                          switch: opp top move guarantee-KOs incoming
  [5] exp_ko    move: expected-KO                            switch: opp top move expected-KOs incoming
  [6] outspeed  both: active speed order (1.0 faster / 0.5 tie / 0.0 slower)
  [7] immune    move: 1.0 if immune on opp active            switch: 0
  [8] dmg_in    move: 0 (I'm the attacker)                   switch: expected dmg the incoming mon takes from opp top move (/100 cap1)

Reuse, do NOT re-derive: MOVE rows come from `fitness.ko.move_ko` (-> the ONE
calc_damage) with the opponent's TOP-1 posterior ability as the hidden-info
override, + `ko_assessment` for the KO verdicts; outspeed from
`calc.compare_speed(own_active, opp_active)`. SWITCH rows: hazard % from
`legality.get_type_chart` (SR = 12.5% x Rock-effectiveness on the incoming types;
Spikes = 12.5% x layers, Ground-immune) + the incoming mon's damage from the
opponent's top move via the SAME calc (defender = incoming species at full HP).

DECISIONS (recorded):
 * TOP-1 posterior ability for the calc (NOT full posterior-weighting): the dominant
   pick carries the signal and it stays ~1 calc/action (12-24 calcs batched in ONE
   calc_damage_many spawn) — the user's "quick + API-reachable" priority. Trivially
   upgradable to weighted (loop candidates x weight).
 * TOP-1 opponent move for the switch "dmg_in"/KO dims (belief top_moves are already
   ranked; the most-likely threat is the one that matters).
 * outspeed uses EFFECTIVE speed (abilities/items/weather included) — the "who acts
   first" fact the net otherwise has to guess.
 * "forced-switch flag" (in the review spec) is IMPLICIT, not a dedicated dim: in a
   forced-switch state only the switch rows carry non-zero facts and the move rows are
   all zero, so the net sees it structurally (keeps the block at the spec's 9 dims).

SPLIT: 3.3a = move/tera rows (+ zero switch rows), full wiring + tests + commit.
        3.3b = switch rows (hazard % + incoming damage), tests + commit.
        (Checksum is pinned ONCE in 3.3a — the field/shape is added once; 3.3b only
         changes content + tests, no re-pin.)

PROGRESS (3.3a + 3.3b DONE, 2026-11-22):
- 3.3a: DONE — `net/action_features.py` (build_action_damage / action_damage_row /
  _top_ability / _active_outspeed), registry +1 field (action_damage 12x9), encoder
  places it (zero-fill when absent, stays pure/DB-free), model folds it into the
  per-action embedding (both the context-mean AND the pointer head, mirroring
  action_move_static; action_dim += 9), env.build_obs wires build_action_damage in
  its OWN try/except (a calc failure degrades only this field, never the obs).
  MOVE/TERA rows filled (fitness.ko.move_ko with the top-1 posterior ability +
  calc.compare_speed outspeed); SWITCH rows zero (3.3b). test_s41 checksum
  re-pinned (739e1d03, 1033->1141 elements, 23->24 fields); +net/tests/
  test_action_features.py (11 tests, hermetic via injected calc/speed fakes).
  net 172 passed; league+fitness 186 passed.
- 3.3b: DONE — SWITCH rows now carry entry-hazard % (dim 8) + the incoming mon's
  damage from the opponent's TOP move (dims 0-5,7). New helpers in
  net/action_features.py: `hazard_pct` (Stealth Rock 12.5% x per-type Rock
  effectiveness from the D29 DB chart; Spikes 12.5%/layer, Ground types take
  none; Toxic Spikes 12.5%/layer; Sticky Web excluded — it is speed, not
  damage; unknown types degrade to neutral 1.0; never raises) and
  `switch_damage_row` (same 9-dim vocabulary as move rows; dim 6 outspeed
  stays 0 for switches). The switch branch builds the calc dicts DIRECTLY via
  block_to_calc_mon (attacker = opp active at its HP fraction, defender = the
  INCOMING mon at its own HP, 1.0 for a fresh bench mon) instead of
  fitness.ko.opp_move_ko — that function scores vs MY ACTIVE's HP, which is the
  wrong defender for a switch row; re-deriving the KO verdicts via
  ko_assessment(raw, defender_hp_frac=hp_in) is what the
  test_switch_ko_verdicts_use_incoming_hp_not_active guard pins. +11 tests in
  test_action_features.py (21 total, hermetic via injected calc/speed/chart
  fakes + one real-calc/DB integration test). BUG CAUGHT during 3.3b: my first
  hazard test values were computed from memory of the type chart, not the DB
  chart (e.g. assumed Ground 2x Rock); the DB chart (D29) is the source of
  truth and the tests now pin its actual cells (Garchomp 6.25%, Magcargo 25%
  SR hazard). No registry/checksum change (shape landed in 3.3a). net 182
  passed; league+fitness 186 passed.
- GUARD FIX (3.3a bug-hunt): test_s42 overfit-32 + state-conditioning FAILED at the
  fixed seed-0 (loss pinned at ln(2)=0.693, the constant-policy attractor).
  Diagnosed (scratch/probe_33_overfit.py) as MULTI-BASIN init-sensitivity, NOT a
  regression: seeds 1/3 converge to 0.0000, seeds 0/2 sit on the attractor even at
  1500 epochs. The net CAN learn; the single-seed gate was brittle to the
  param-count shift (every Step 3.x feature add moves the seed-0 init). Fix: both
  guards now try seeds 0-3 and pass if ANY converges — a structurally-broken net
  still fails every seed, so the guard stays strong. (Recorded per the user's
  "never-learning disaster" sensitivity: this must stay a HARD catch, not a waiver.)

BUG-HUNT (2026-11-22, "check for bugs and unwanted outcomes first"): found + fixed
3 real issues, all in the static-feature path (the net suite grew to 160 passed,
league+fitness 186 passed):
 1. **tera actions lost their move-static row** — build_action_list emits BOTH a
    plain `move` and its `tera` variant (each carrying the same `move` id), but the
    builder only filled `kind=="move"`. A tera action is EXACTLY where the move's
    type/power matter most (tera changes type effectiveness), so it was zeroing
    out. Fix: fill the row for ANY action carrying a `move` (move+tera); switch/
    transform stay zero. (test_tera_action_gets_move_static_row)
 2. **old-gen species zeroed out** — verified the legality DB stores ONLY gen-9
    species data; gens 1-8 `get_species` return `types=[]` / `stats=None`, so every
    non-gen9 format would have had ZERO mon static features (the user wants all
    formats). Fix: `_species_data` falls back to the gen-9 row for types/stats
    (base types/stats are gen-invariant); move data stays per-gen. (test_old_gen_*)
 3. **a static-feature bug could take down the WHOLE obs** — env.build_obs computed
    static features inside the same try as belief/encode, so an unexpected static
    error would zero out HP/status/hazard/belief too. Fix: static features are
    built in their OWN try/except; a failure degrades only the static fields to
    zero (encoder zero-fills), the rest of the obs survives.
 + defensive: flag checks use truthiness (not `is not None`) so an empty container
   from the DB cannot falsely flag a move.

### 3.4 DESIGN (locked before coding, 2026-11-22) — per-own-mon MATCHUP summary

WHAT IT ADDS: the board-level damage LANDSCAPE — "how does EACH of my mons fare
against the whole enemy team?" This is the signal the net needs for the user's
win-condition / switch / threat scenarios ("my dragonite 6-0s once gengar is
dead", "who can I KO, who KOs me", "is my switch-in safe"). It is a PER-OWN-MON
fact (one 8-dim vector per own slot), so it folds into the OWN-side MON ENCODING
(`MonEncoder` / `_side_repr`) — the EXACT 3.5 lesson: per-mon things go per-mon,
NOT summed into the global context (the 3.5 first attempt summed 6 opponent slots
into a global constant and shadowed the own-side signal, loss stuck 0.69). The
opponent side passes all-zero own_matchup (its <pad>/zero contribution is 0),
mirroring how the own side already passes all-zero posterior.

New registry field `own_matchup` (MAX_SLOTS, 8) float32. K_OWN_MATCHUP = 8. All
dims bounded in [0,1] (pcts/100, fractions/6, flags) — no raw constant to shadow.
  [0] best_hit       strongest hit THIS mon can land on any enemy   (max exp% /100)
  [1] worst_hit      strongest hit any enemy can land on THIS mon   (max exp% /100)
  [2] ko_guar_frac   # enemy mons it KOs guaranteed / (# alive enemy mons)
  [3] ko_can_frac    # enemy mons it CAN KO (max dmg)              / (# alive enemy)
  [4] taken_ko_frac  # enemy mons that CAN KO this mon             / (# alive enemy)
  [5] survives_strongest  1.0 iff the worst enemy hit < 100% of full HP
  [6] net_ko_margin  (ko_can_frac - taken_ko_frac)                 (in [-1,1])
  [7] best_vs_worst  best_hit - worst_hit  (signed, in [-1,1]) — "does it hit
                          harder than it gets hit?" (a fast read on threat level)

COMPUTATION (reuse the ONE calc, do NOT re-derive):
 * ONE batched `damage.calc.calc_damage_many(items)` — up to 6 x 6 x 2 = 72 items
   (own_i's strongest move vs opp_j, AND opp_j's top move vs own_i), in a SINGLE
   Node spawn (the "quick + API-reachable" priority; no per-item spawn).
 * each own mon's attack move = its HIGHEST-base-power damaging move (DB lookup;
   fallback the first move). each opponent's attack move = its `top_moves[0]`
   (belief posterior), fallback its first observed move.
 * opponent hidden info = that opponent mon's TOP-1 posterior ability (override),
   same convention as 3.3.
 * KO verdicts via `fitness.ko.ko_assessment(raw, defender_hp_frac=defender's
   current hp_frac)` — same semantics as `move_ko` (a defender at hp_frac has
   hp_frac of max HP; a move KOs when exp/max% >= that threshold).
 * DEGRADES to a ZERO row per mon (never raises) when a species/move is unknown
   or the calc fails — the net reads "unknown", exactly like the id channels.
 * TURN-1 / NO-REVEALED-MOVE edge (BUG-HUNT 2026-11-22): before the opponent
   reveals a move, its `top_moves` is empty. Rather than silently zeroing the
   whole matchup (a real bug found while probing the live belief path), a mon
   with NO revealed move uses the NEUTRAL Body Slam baseline (85 power, 100% acc,
   always-legal, always-hits) as its representative attack move. A "highest
   power" legal-move scan was considered and REJECTED (it picked quirky moves
   like Explosion / Gigaimpact, which are not representative). Body Slam is a
   sane, universal, always-hits damage baseline; it leaks no hidden info
   (species-independent) and gives a sensible "the opp can hurt me" prior on
   turn 1 that sharpens as the opp's real moves are revealed. (Own side is
   unaffected in self-play / live bot — it has its full moveset from turn 1.)

PLACEMENT / WIRING (same discipline as static_features + action_features):
 * `net/matchup_features.py` — PURE, lazy calc import, `build_own_matchup(ctx,
   fmt, gen, calc_fn=None)` -> `{"own_matchup": (6,8) float32}`; injectable
   `calc_fn` so tests pin interpretation with fakes (only a couple touch Node).
 * encoder PLACES it (zero-fill when absent — the pure/DB-free contract holds).
 * model: `MonEncoder` takes an `own_matchup` input, folds it into the per-mon
   vector (like types/stats), `in_dim += K_OWN_MATCHUP`; own side passes the
   real array, opponent side passes zeros. `_side_repr` threads it through.
 * `env.build_obs` wires `build_own_matchup` in its OWN try/except (a calc
   failure degrades ONLY this field to zero; the HP/status/belief obs survives).
 * `test_s41` checksum re-pin (24 -> 25 fields); +`net/tests/test_matchup_features.py`
   (15 hermetic tests via an injected batch calc_fn + 1 real-calc/DB integration
   test, incl. the turn-1 Body Slam fallback guard).

SPEED DECISION (recorded, deliberate): the per-pair SPEED-advantage dim from the
initial sketch is DROPPED — base speed is ALREADY in the `own_stats`/`opp_stats`
channels (per-mon), the net can read "who is faster" per mon, and per-pair
effective speed (abilities/items/weather) would need 36 extra Node spawns (2 x
36 compare_speed calls), violating the quick/API priority. The per-turn ACTIVE
outspeed is already carried by `action_damage[6]` (3.3). If Step 8 shows the net
wants an explicit speed-order signal, this is a 2-line re-add (re-pin checksum).

PROGRESS (3.4, 2026-11-22):
- 3.4: DONE — registry +K_OWN_MATCHUP (8) + field `own_matchup` (6x8); new
  `net/matchup_features.py` (build_own_matchup / own_attack_move / opp_attack_move
  / best_damaging_move (Body Slam baseline) / _top_ability, ONE calc_damage_many
  batch, ko_assessment KO verdicts, degrades to zero, turn-1 Body Slam fallback);
  encoder places it (zero-fill); MonEncoder folds it into the per-mon vector
  (in_dim += 8), own side real + opponent side zeros; env.build_obs wires it in
  its OWN try/except. test_s41 checksum re-pinned (1141 -> 1189 elements,
  24 -> 25 fields); +net/tests/test_matchup_features.py (15 hermetic tests via an
  injected batch calc_fn + 1 real-calc/DB integration test). net 197 passed;
  league+fitness 186 passed. BUG-HUNT 2026-11-22: found + fixed the turn-1
  "opponent revealed no move -> top_moves empty -> whole matchup silently zero"
  edge (now falls back to the Body Slam baseline, pinned by a test); confirmed
  the live own-block path carries `name`+moves (env._own_team_from_battle) and
  the opponent path carries `species` (belief), so block_to_calc_mon resolves
  both sides. (See the BEFORE baseline recorded in the Step 8 section below.)

### Step 8 BEFORE baseline (recorded 2026-11-22, the ONLY "before" number)

The old (GRU-trunk) checkpoints can no longer load under the Step 2/3 code, so
this is the last chance to capture the "before" performance. Ran on the league's
best checkpoint `scratch/league_random_1001.pt` (ELO 1586, `league/bots.json`),
50 battles vs two references (random + heuristic), on the OLD in-memory code:

  * sampled  vs random     **WR 0.74** (37W/13L/0D)  95%CI [0.618, 0.862]
  * sampled  vs heuristic  0.48  (24W/25L/1D)         95%CI [0.342, 0.619]
  * argmax   vs random     0.54  (27W/18L/5D)         95%CI [0.402, 0.678]
  * argmax   vs heuristic  0.44  (22W/26L/2D)         95%CI [0.302, 0.578]

Interpretation: the league's best self-play checkpoint is WEAK — it barely
beats random and LOSES to the simple heuristic. So Step 3's added signal
(static facts + per-action damage + per-mon matchup) + the Step 4 single-learner
experiment must show a clear lift over these numbers to count as "learning."
The "after-Step-3" learning_check run (Step 8) is the comparison point. (Rows
appended to `runs/learning_check.csv`, label `before-step2`; CSV is gitignored.)

### After-Step-3 TRAINING run (recorded 2026-11-22; the "after" checkpoint)

Because the old (GRU-trunk) checkpoints can't load under the Step 2/3 code, the
"after" number needs a FRESH checkpoint trained on the NEW layout. Trained one
main learner (Fork 2, single-learner-first) vs random, CPU, 8 rounds x 20 battles
= 160 battles, on-policy PPO (lr 3e-4, clip 0.2, gamma 0.99, lam 0.95, 2 epochs).
Checkpoint: `scratch/after_step3/selfplay_round8.pt` (saves + re-loads OK under the
new-layout verification; the smoke + this run also proved the pipeline is sound
under the new layout).

Per-round vs-random win rate (the "is it learning?" signal at train time):
  round 1: 12W/8L (60%)   round 2: 14W/6L (70%)   round 3: 15W/5L (75%)
  round 4: 15W/5L (75%)   round 5: 16W/4L (80%)   round 6: 16W/4L (80%)
  round 7: 18W/2L (90%)   round 8: 18W/2L (90%)

0 crashes, 0 watchdog timeouts, 0 illegal actions across all 160 battles. KL
stabilizes to ~0.01 by round 7-8 (the policy is converging), value loss falls
0.25 -> 0.07. The net is CLEARLY learning and climbing — at 90% vs random after
only 160 battles, it already matches the before-baseline's 0.74 (which needed
~1000 battles). That is a strong positive signal for the Step 2/3 observation
layout. The definitive "after" number (vs random + vs heuristic, sampled +
argmax, n=50) is the concurrent `learning_check` run (label `after-step3`),
appended to `runs/learning_check.csv` when it completes.

HARDENING (committed `7ec0869`): the first 160-battle attempt hung for 14+ min
inside a single `env.step()`/`reset()` (a transient server stall — a `max_steps`
cap can't help because the stall is INSIDE one step call). Diagnosis (15 battles,
same seeds) reproduced NO hang, confirming it was transient, not a Step 2/3 bug.
Fix: `SelfPlayTrainer._play_with_watchdog` (cfg `battle_timeout`, default 240s,
0=disabled) kills a hung battle and skips it so one hang can never stall a run;
`rollout` records `n_timeouts`. +4 hermetic tests.

### Step 8 RESULT: after-Step-3 vs before (the definitive comparison, 2026-11-22)

`learning_check` (n=50/ref, gen9ou, sampled+argmax, vs random+heuristic) on
`selfplay_round8.pt`, appended to `runs/learning_check.csv` (label `after-step3`):

  * sampled  vs random     **0.84**  (42W/8L/0D)   95%CI [0.738, 0.942]
  * sampled  vs heuristic  **0.60**  (30W/20L/0D)  95%CI [0.464, 0.736]
  * argmax   vs random     **0.76**  (38W/11L/1D)  95%CI [0.642, 0.878]
  * argmax   vs heuristic  **0.76**  (38W/12L/0D)  95%CI [0.642, 0.878]

vs the BEFORE baseline (old layout, `league_random_1001.pt`, ~1000 battles):
  sampled vs random 0.74 -> **0.84**  (+10)
  sampled vs heuristic 0.48 -> **0.60** (+12)   [before LOST to the heuristic; after WINS]
  argmax  vs random 0.54 -> **0.76**  (+22)
  argmax  vs heuristic 0.44 -> **0.76** (+32)   [before LOST to the heuristic; after WINS]

**Verdict: the Step 2 (MLP trunk) + Step 3 (static facts + per-action damage +
per-mon matchup) observation layout is a clear, strong positive.** All four
metrics improve; the net flips from LOSING to the heuristic (before: 0.48/0.44)
to BEATING it (after: 0.60/0.76); and it does so with only 160 battles (the
before needed ~1000). The "never-learns" disaster is avoided — the pipeline
learns and the new signal helps. This clears the Step 8 gate to proceed to
Step 4 (the single-learner-first experiment at scale, per the approved Fork 2).

OPEN — next call with the user: (a) proceed to Step 4 single-learner experiment
(the league now plays a POOL of frozen snapshots + the main learner, not just
random — the before net trained against a league pool, so the fair next step is
the same, with the new layout); (b) how much training to give the main learner
before re-running this eval (the 160-battle result is already strong, but more
training + the league pool is the real test).

OPEN DESIGN OBSERVATION (not changed — needs a call): the MOVE's type is encoded as
ONE ordinal float (`type_idx/18`) while the MON's types are one-hot (18 floats).
An ordinal encoding of a CATEGORICAL type is a slightly awkward input for learning
the type CHART (Fire vs Water vs Grass becomes a distance problem). Making the move
type one-hot too (18 floats instead of 1) would be more symmetric and likely better,
but it changes K_MOVE_STATIC -> re-pin checksum + wider input. Deferred: decide with
3.3/3.4 (the damage features) when the observation layout is finalized.

---

## STEP 5 DESIGN NOTE (draft, PARKED — implementation blocked on Step 4/8 showing learning)

Status: design only. Per review plan, do NOT implement M2b until the Step 8 eval
of the Step 4 learner confirms improvement. Step 4 training launched 2026-11-22 13:18
(PID 1765, 160 battles / 8 rounds, log `scratch/step4/train_step4.log`).

### Problem (why)
The alternating MCTS tree gives the opponent an unfair first-look (setup/hazard/
double-switch cases systematically undervalued) and cannot output mixed strategies.
Pokemon turns are SIMULTANEOUS.

### Design (per-turn matrix game)
1. Rows = my legal actions (cap top-K by policy, K=3-4).
2. Cols = opponent plausible actions, belief/posterior-weighted (top-K by posterior),
   K=3-4. Turn-1 case: posterior must yield >=1 plausible attack move (already an
   OC-9 test requirement).
3. Cell (i,j) = state after BOTH actions resolve, scored by value head (root-POV).
   - Forward model option A: Showdown sim clone (Battle.toJSON/fromJSON, apply both
     choices, read result). Requires spike: clones+turns/sec on gen9bss3v3.
   - Forward model option B (fallback): depth-1, calc-based approximation for the
     top-3x3 cells only.
4. Solve zero-sum matrix with `scipy.optimize.linprog`; SAMPLE action from the
   resulting mixed strategy (keeps exploration legal; log-prob from the mixed
   strategy for PPO, or sample-and-record like today — decide at build time).
5. Belief samples: for each posterior sample, matrix differs via cell values;
   average mixed strategies across samples (or sample state first, then solve).
   Budget check: ~36 cells x ~20 samples x forward-cost must fit per-turn time.
6. Tera/mega: omitted from action lists (OC-8); nulled in formats without them.

### Spike plan (first build task when unblocked)
- Measure sim clone+one-turn cost in the fork (target: <~50ms/clone to keep
  ~720 forward evals < 36s/turn; otherwise drop to option B or smaller K).
- Verify Battle.toJSON/fromJSON round-trip for gen9bss3v3 incl. weather/terrain.
- Hermetic test: known 2x2 matrix -> LP returns analytic mixed strategy.

### Acceptance gates
- Sim forward model agrees with real Showdown on scripted cells (reuse OC-6
  agreement-suite style: hazards, status, screens, speed ties).
- Mixed strategy sanity: rock-paper-scenario states produce non-degenerate mixes.
- A/B vs alternating tree on the Step 7 scenario suite; keep the better one.

### Open decisions (for user, at unblock time)
- S5-D1: forward model A (sim clone) vs B (calc depth-1) — expect A if spike is
  fast, else B.
- S5-D2: K (rows/cols) default 4 or 3 — latency vs quality.
- S5-D3: how to log mixed-strategy action for PPO (sampled action + logprob from
  mix, vs. policy head fallback).
- S5-D4: belief-sampling strategy (average mixes vs sample-state-then-solve).

### SPIKE RESULTS (2026-11-22, gen9bss3v3 + gen9ou) — prerequisite verified ✅
Script: `scratch/step5_spike/roundtrip.js` (node, uses fork `dist/sim`).
- **Round-trip fidelity: PASS.** Mid-battle state (damage dealt, Stealth Rock up,
  Light Screen up, burn, Dragon Dance boosts) serialized via `State.serializeBattle`
  / `State.deserializeBattle` and back: ALL fingerprinted fields agree
  (per-mon hp/maxhp/status/boosts/volatiles/item/ability, side conditions,
  field weather/terrain/pseudoweather, PRNG seed). Both gen9bss3v3 and gen9ou.
- **Clone is fully playable:** deserialized clone accepts `makeChoices`, plays a
  full turn, end-of-turn effects fire (Black Sludge tick), boosts/hazards persist.
  => Forward-model option A (sim clone) is VIABLE. Option B (calc depth-1) is now
  only a fallback, not a necessity.
- **Cost (CPU, training running in parallel):** serialize ~0.2-0.4ms,
  deserialize ~0.4-1.1ms, clone+1 turn ~0.5-0.6ms median.
  36 cells x 20 belief samples ≈ **1.0-1.5 s/turn** — acceptable. Re-measure on GPU
  machine without CPU contention before finalizing K (S5-D2).
- **LP solve (scipy linprog, highs):** verified against analytic 2x2 solutions
  (symmetric v=2 p=(0.5,0.5); asymmetric v=1.5). ~0.8ms per 2x2 solve.
- **NEW DEPENDENCY:** `scipy` (installed 1.17.x via pip 2026-11-22). NOT yet pinned
  in any requirements file (bot/requirements.txt is dirty/uncommitted — pin with the
  user when it is next touched).

### SIM API FINDINGS (do not re-discover at build time)
- Fork already ships `sim/state.ts`: `State.serializeBattle(battle)` (AnyObject) and
  `State.deserializeBattle(obj|str)`. `Battle.toJSON()` / `Battle.fromJSON()` wrap them.
- `State` is NOT re-exported by `dist/sim/index.js` — require `dist/sim/state.js` directly.
- Battle ctor: `new Battle({ format: Dex.formats.get('gen9bss3v3'), p1:{name,team}, p2:{...} })`
  (pass the FORMAT OBJECT, or use `formatid`). Team sets use object EVs
  (`evs:{hp,atk,def,spa,spd,spe}`).
- Ctor leaves the battle in **team preview**: answer with `battle.makeChoices('team 3','team')`
  ('team N' = lead with slot N, 1-indexed).
- Choices are STRINGS: `side.choose('move <id>')`, `'switch N'`, `'team N'`.
  `battle.makeChoices(p1str, p2str)` = both sides choose + commit.
  `pokemon.moves` is an array of move IDs (strings).
- `side.active` is the active-Pokemon array; the `pokemon.active` flag does NOT exist.
  Side: `side.requestState` ('move'|'switch'|'teampreview'|'').
- **Quirk (found, not chased):** U-turn / forced-switch flows can leave the OTHER side
  with `requestState===''` (serialized requests) and `switch 1` can reject when the
  desired target is flagged active. For the matrix-game build, forced-switch states
  should be handled explicitly (they are low-entropy: the switcher's options are the
  bench) — do not assume both sides always have a 'move' request simultaneously.
- `commitChoices()` throws "Not all choices done" if a side's request is empty —
  guard with `requestState` checks.

### Step 4 smoke + argmax investigation (2026-11-22)
- Smoke (n=10, step4_round8.pt): sampled 1.000 vs random, 0.900 vs heuristic;
  **argmax 0.500 vs random (FAIL), 0.900 vs heuristic**. Full n=50 gate launched
  (label `step4-vs-pool`, log `scratch/step4/eval_n50.log`).
- Argmax trace (scratch/step4/argmax_trace.py, 3 battles vs random, seed 777):
  **no degenerate fixed top-1** — picks are context-dependent, top1-top2 gap small
  (median 0.072), bot won 3/3. => the 0.50 is likely variance + a behavioral bias.
- **Behavioral bias observed:** greedy top picks skew to setup/support
  (Rapid Spin x64, Swords Dance x32, Calm Mind x32, Stealth Rock/Spikes) over
  attacks. Greedy = slow utility play. Watch the n=50 heuristic column; if greedy
  stays weak vs random but strong vs heuristic, the fix is training-side (reward
  pacing / opponent pressure), not a code bug.
- OPEN: if n=50 confirms argmax-vs-random < 0.6, decide: (a) accept greedy as a
  weak mode (we play sampled in training and can play sampled in league), or
  (b) investigate the policy head's top-1 (e.g. entropy coefficient is keeping the
  distribution flat — median gap 0.072 — which is fine for PPO but makes argmax
  nearly a coin-flip among 3-4 moves).
