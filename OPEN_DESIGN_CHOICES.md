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

**Gate:** league+net+gui 211 passed / 1 skipped (CUDA-only) / 5 deselected
(live); `league/tests/test_learner.py` 13/13; three real checkpoints load
through the single path. CPU-only throughout.

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
