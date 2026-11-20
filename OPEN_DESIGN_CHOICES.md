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
