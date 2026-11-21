# Workflow Review + Open Design Choices (2026-11-19)

> Purpose: a single scannable place for (A) the workflow errors found in the
> end-to-end pass, (B) the design choices that still need **your** call, and
> (C) the current checkpoint board. Review this when you're back — nothing in
> section B is built until you pick.

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
