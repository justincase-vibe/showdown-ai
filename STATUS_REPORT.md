# PROJECT STATUS REPORT — 2026-11-24 (late night)

> For the next agent: read this first, then read the Claude handout
> (`pokemon_ai_master_handoff (1).md` in the scratchpad) for the full
> strategic context. This report is the operational snapshot.
>
> **2026-11-24 ~00:30 UPDATE (second stuck-run incident + thread-leak fix):**
> the 23:37 learnability relaunch (PID 11036) was a SECOND hang in the round-0
> baseline eval — same symptom, different root cause: NOT orphaned node
> processes (all 60 were legitimate children of that run) but RAM exhaustion
> (free RAM 0.5 GB, page file active, node servers swapped out and idle while
> python spun ~8 cores in retry loops). User reference point: 8 servers x 2
> envs ran fine last week at ~70% RAM, so the 4-server footprint is not the
> limit — the LLM's KV-cache growth is what squeezed the box that night.
> User restarted the conversation/llama-server to free system RAM. A smoke run
> (PID 10428, 1 server, ~7.5 GB free) then completed round-0 eval with
> 0 crashes (sampled WR 0.150 / greedy 0.250, baseline) and round-1 PPO
> update cleanly — pipeline verified working when RAM is available.
> Meanwhile py-spy forensics found a real per-battle LEAK: ~2-3 live threads
> per battle (the PsEnv loop thread poke-env never reaps + its default-executor
> thread). Fixed in net.env (`_stop_env_loop`, commit 3b51c0d, 6 hermetic
> tests; net suite 235 passed). See INCIDENT #2 below.
>
> **2026-11-25 ~01:30 UPDATE (smoke verdict + full run relaunched):** the smoke
> run finished (PID 10428, 4513 s): **pipeline VERIFIED** — 0 crashes across all
> 6 eval phases, PPO rows both rounds, WR climbing (sampled 0.150→0.700,
> greedy 0.250→0.350→0.600), entropy healthy (1.62→1.58), KL early-stop works.
> Verdict line says FAIL only because the 0.85 criterion is defined for 15
> rounds, not 2 — expected. Smoke killed, ports verified free (49990/49991/
> 50006/50007), **FULL run relaunched: PID 13028**, run dir
> `runs/learnability_20261125T001817Z`, logs `runs/learnability_full6.{out,err}.log`,
> PID file `runs/learnability_full6.pid`. Post-launch checks: thread count
> FLAT at 61 (fix working — old code reached 166 in ~90 battles), no real
> errors (err log = poke-env `CancelledError intercepted` CRITICAL teardown
> noise only, expected with the new loop-stop path), free RAM ~3.5-6.6 GB,
> pagefile inactive (no incident-#2 signature). NOTE: with this conversation's
> llama-server (~18 GB) + 60 node workers, free RAM is tighter than the
> ~5 GB rule — user may restart the conversation for overnight headroom (the
> run is detached and does not need the LLM).
>
> **2026-11-25 ~03:00 UPDATE (user: "continue bug fixing, tear down as needed, ASAP"):**
> Process audit came back CLEAN — every node proc belongs to the current run
> (2 servers × battle/plugin workers + 8 calc workers), exactly one python
> (PID 13028), zero orphans, nothing from the old project. Run verified
> progressing (round 2/15, 0 crashes, threads flat 51, no swap). No live bug:
> battle loop has no artificial delays; pace = battle length × parallelism.
> DECISION PENDING (user asked for asap): **PATH A** = keep 2×2 running
> (verdict ~17:00, safe). **PATH B (fast)** = user restarts the conversation
> to free ~10 GB, next session tears down 13028+node and relaunches with
> `--n-servers 8` (user's proven last-week config; verdict ~08:00–10:00).
> Scripts prepped in `scratch/`: `kill_run13028.ps1` (kill python 13028 + all
> node, verify 49990-49997 + 50006-50013 free) and `launch_full8.ps1`
> (preflight: 0 node/python, free RAM ≥ 8 GB, 16 ports free; launches
> `python -m tools.learnability_test --rounds 15 --battles-per-round 50
> --n-servers 8`, logs `runs/learnability_full8.{out,err}.log`,
> PID `runs/learnability_full8.pid`). If PATH B is chosen, run the kill
> script FIRST, then the launch script, from
> `C:\Users\JustinCase\Documents\Projects\showdown AI-imitation`.
>
> **2026-11-25 ~04:45 UPDATE (unattended pipeline FINAL — user cannot watch, everything automated):**
> `scratch/unattended_pipeline.ps1` (parser-verified, ASCII-only) +
> `scratch/START_UNATTENDED.bat` (hidden window) + `scratch/STOP_ALL_TRAINING.bat`
> (NEW, calls `stop_all.ps1`). Sequence: wait free RAM (probe >=5 GB / runs >=8 GB,
> 45-min gate, then REFUSE - incident-#2 lesson) -> **yardstick probe** (CPU 2x2,
> ports 49960-49961, 10 h timeout) -> auto GREEN/RED (C4a/C4b mirror |wr-0.5|<=0.15,
> C3 heuristic-vs-strategy <=0.55, C5 fallbacks <= max(5, 0.5% of decides), 0
> crashed) -> **gate Run 1** controls (CUDA 6x2, ports 49980-49985, `--out
> runs\gate_run1`) -> auto GREEN/RED (`completed_ok` + `validity.yardstick_invalid
> == false`) -> **gate Run 2** (r70+r80, `--out runs\gate_run2`) -> verdicts -> DONE
> flag. In parallel a **15-min watchdog** over learnability PID 13028: died ->
> scoped node-kill (`Kill-NodeUnder` dead-pid chain only, other stages' node
> procs untouched) + relaunch (<=2 retries); incident-#2 signature (log stale
> >120 min + free <1.5 GB) -> same recovery; stale but RAM OK -> leave it;
> `LEARNABILITY VERDICT` line -> stop watching. RED at any stage -> STOP +
> `runs\unattended\PIPELINE.flag` (no burning the 8 h controls). Log:
> `runs\unattended\unattended.log`. **Review of the pre-written pipeline caught
> 6 real bugs via small functional tests (all fixed, all 19 checks PASS):**
> (1) `Get-RowNum` required `[Hashtable]` but `ConvertFrom-Json` yields
> PSCustomObject -> every counter 0 -> probe would ALWAYS flag RED;
> (2) stages 2+3 missing `--out` -> result.json in default ts dir -> Run 1 would
> ALWAYS flag RED and Run 2 verdicts unread; (3) `Log` used `Write-Output` ->
> polluted `Invoke-Stage`'s return stream (log lines + code array);
> (4) C3's unparenthesized `Get-RowNum a + Get-RowNum b` chain evaluates to the
> FIRST call only in command context (empirically 30, not 100) -> C3 always
> computed wr=1 -> always RED; (5) watchdog recovery killed ALL node procs ->
> would have killed a concurrently RUNNING gate stage's servers; (6)
> `Test-PortsBusy` matched TIME_WAIT -> would falsely abort Run 2 reusing Run 1's
> ports (now LISTENING-only). Also: probe timeout 5->10 h, `launch_full8.ps1`
> port check + em-dash fixed, all files ASCII-only (PS 5.1 cp1252 trap).
> **Learnability 13028 at 04:31: round 2/15, 0 crashes, r2 sampled 0.490,
> threads flat, pagefile ~0.6 GB (no incident-#2 signature).**
> USER ACTION: close the Bionic conversation (frees ~10 GB), then double-click
> `scratch\START_UNATTENDED.bat` (or paste its one line). Walk away.
>
> **2026-11-26 RESULTS (all runs COMPLETE — league ran on CUDA after the user freed RAM):**
> **M1 GATE = PASS** (`runs/gate_run2/result.json`, r70+r80; controls in `runs/gate_run1`; in-distribution only — same 25-team pool as training, NOT a generalization claim):
> - **r70 PASS** — pooled WR vs strategy **0.6525** [95% CI 0.545–0.757]; vs strategy_posterior **0.7225** [0.616–0.825].
> - **r80 PASS** — pooled WR vs strategy **0.6375** [0.530–0.742]; vs strategy_posterior **0.695** [0.588–0.798].
> - Control random-init: 0.2675 [0.165–0.374] → STOP (validity floor, expected). `yardstick_invalid: false`, C2 null, C1 upper-95% 0.374 < 0.40.
> - **C3 WARNING (open):** legacy `heuristic` still beats the `strategy` yardstick (~0.68; was 0.60 in the RED run) → yardstick weaker than the baseline it is meant to replace; quantify with the league table.
> **2000-battle self-play league = COMPLETE** (`runs/selfplay_20261126/summary.json`, CUDA, 0 errors, 2000/2000 saved, replays `runs/replays/gen9ou.jsonl` = 3896): net WR over ~1111 battles each → **r80 0.6526 > r70 0.6403 > learn_r15 0.4608 > random 0.0526**. Head-to-head: r80 edges r70 (r70 wins only 0.462); r70/r80 beat learn_r15 0.785 / 0.761; learn_r15 0.856 vs random; r70/r80 0.955 / 0.964 vs random. Mirrors ~0.50 (r70 0.5045, r80 0.4595, learn_r15 0.5315) → no mirror bias.
> **Learnability (learn_r15) = FAIL vs the 0.85 bar** but clearly learning (`runs/learnability_20261125T001817Z/report.json`): greedy 0.41→0.74, sampled 0.53→0.67 (best r14), 0 crashes, ~17 h.
> **r80 is the strongest NN** (best league WR + gate PASS) — treat it as the “latest NN.”
> **Fix shipped this session:** the p1 harness `act()` contract — the yardstick probe had been crashing 200/400 battles on `StrategyAgent has no attribute 'act'` (C4a/C4b mirror controls). `net/env.py` thread-local ctx stash (`set_last_ctx`) + `fitness/strategy_agent.py.act()` + tests pin it; the 0-crash gate runs are a direct result.

---

## WHERE WE ARE

**Active macro step: M1 GATE (single-learner experiment)**
Per the Claude handout §4, the M1 gate is the active experiment. M2b/M3/M4/M5
are PARKED until the gate result is in.

**Work order (user-approved, from `GATE_CONTROLS_C1_C5.md`):**
1. **(b)** Run gate NOW with controls C1–C5 → **DONE 2026-11-26: r70 + r80 PASS**
2. **(a)** PPO fix pack + learnability test → **IN PROGRESS** (see below)
3. **(c)** BC warm start (Arm B) → pipeline built, collection pending
4. **Compare** fixed-PPO vs BC+PPO, ≥400 battles per comparison → PENDING

---

## WHAT IS RUNNING (as of this report)

| Item | Status | PID |
|------|--------|-----|
| Learnability SMOKE (a) | **DONE** — verdict in (pipeline verified; 0 crashes; WR 0.15→0.70; FAIL vs 0.85 on 2 rounds is expected) | 10428 (exited) |
| Learnability FULL (a) | **RUNNING** — round 2/15 at 04:31, 0 crashes, r2 sampled 0.490; watchdog covers it | 13028 |
| Unattended pipeline | **READY, not launched** — `scratch\START_UNATTENDED.bat`; waits for RAM, then probe → Run 1 → Run 2 with auto decisions | — (not started) |
| GPU gate (b) | Now driven BY the unattended pipeline (Run 1/Run 2, `--device cuda`, user-launched bat) | — |
| BC collection (c) | Pipeline ready, not started | — |

### ⚠️ INCIDENT #2: RAM exhaustion hang (2026-11-24 ~23:37–00:10) + thread leak

The 223712Z relaunch (PID 11036) hung in the round-0 baseline eval again.
**This time the cause was NOT orphaned processes** — all 60 node processes
(4 servers x 12 subprocess workers + 8 calc workers) were legitimate children
of the run. The cause was RAM:

| signal | value |
|---|---|
| free RAM | **0.5 GB** (oscillating 0.5→4 GB = thrash) |
| page file | peak 3.6 GB and climbing (active swap) |
| node CPU | 0.3 s per 5 s (servers swapped out / idle) |
| python CPU | ~8 cores (poke-env retry spin) |
| err log | `Task was destroyed but it is pending!` growing (9→30) |

Budget: llama-server ~23 GB + 4 servers ~5.2 GB + python ~1 GB ≈ 31 GB of
32 GB. Killed at ~00:10 (python 11036 + 52 node procs; ports verified free;
free RAM recovered to ~7.5 GB). User then restarted the conversation
(llama-server KV cache) — free RAM now ~10 GB at idle.

**User reference point (IMPORTANT for sizing):** 8 servers x 2 envs ran fine
last week at ~70% RAM. The server footprint is NOT the limiting factor; LLM
RAM growth is. Do NOT "fix" this by shrinking server counts — free the LLM
side instead (and keep an eye on free RAM during long runs: kill/restart the
conversation before free RAM drops below ~5 GB).

**Thread leak found during forensics (FIXED, commit 3b51c0d):** every battle
leaked its PsEnv loop thread (poke-env spawns `Thread(target=self._loop.run
_forever, daemon=True)` anonymously and only reaps the shared POKE_LOOP at
atexit) + the loop's default-executor thread (`asyncio_N` names) — ~2-3 live
threads per battle for the life of the process (~5-7k threads + handles over
a full run). `net.env._stop_env_loop()` now stops it in `ShowdownEnv.close()`
(last, after the PS clients whose tasks run on that loop). Includes the
**Windows proactor quirk** (verified): cancelling a task suspended on a timer
does NOT wake a sleeping loop — an explicit `call_soon_threadsafe` kick is
required or the cancellation sits unprocessed (this also explains the
"Task was destroyed but it is pending!" noise: tasks GC'd while still
pending). 6 hermetic tests in `net/tests/test_env_teardown.py`; net/tests
235 passed, 2 skipped.

### ⚠️ INCIDENT #1: 58 orphaned node processes (2026-11-24 ~23:10–23:35)

The first relaunch (PID 20116, started 23:10) got stuck in round 0 eval.
Diagnosis: **58 node processes** running (expected ~28 for 4 servers × 4 sim
workers + 8 calc workers). Root cause: orphaned node processes from previous
crashed runs accumulated — each killed run left its sim workers behind. The
websocket connections were overwhelmed, battles couldn't complete, and the
asyncio tasks were destroyed pending.

**Fix applied:** killed all python + node processes, verified 0 remaining,
relaunched clean (PID 11036, run dir `runs/learnability_20261124T223712Z/`).

**Prevention for next agent:** before launching any battle run, check for
orphaned node processes:
```
powershell -NoProfile -Command 'Get-Process | Where-Object { $_.ProcessName -eq "node" } | Measure-Object | Select-Object Count'
```
If count > 30, kill them all first:
```
powershell -NoProfile -Command 'Get-Process | Where-Object { $_.ProcessName -eq "node" } | Stop-Process -Force -ErrorAction SilentlyContinue'
```

**Learnability test details (as of 00:30):**
- Full run: **NOT RUNNING** (killed at ~00:10, incident #2). Relaunched with
  the thread-leak fix once the smoke run (PID 10428, run dir
  `runs/learnability_20261124T225458Z_smoke/`, log
  `runs/investigate_smoke.out.log`) finishes — it was in round-1 eval at 00:35.
- Command: `python -m tools.learnability_test --rounds 15 --battles-per-round 50`
- Gate: WR vs random ≥ 0.85 within 15 rounds
- Baselines (init-vs-init mismatch, varies per init, not a bug): smoke run
  223712Z-smoke gave **0.150 sampled / 0.250 greedy**; the earlier
  "0.740 sampled / 0.620 greedy" was a different init. Either way round 0 is
  not ~50/50 — the learning requirement is climbing to 0.85 within 15 rounds.
- **Check progress:** `cat <run_dir>/metrics.jsonl`
- **Pace (measured, 1 server x 2):** ~45 s/battle; a full 15-round run is
  ~10 h on 2x2 servers. Launch at night, check round-0 + round-2 rows, then
  leave it. Free-RAM watch: keep ≥ ~5 GB free (see incident #2).

**If the user launches the GPU gate:**
- Kill the learnability test first: `powershell -NoProfile -Command 'Stop-Process -Id 11036 -Force'`
- Also kill all node processes: `powershell -NoProfile -Command 'Get-Process | Where-Object { $_.ProcessName -eq "node" } | Stop-Process -Force'`
- Gate paste is in `GATE_CONTROLS_C1_C5.md` §7
- The LLM (llama-server) uses ~23 GB CPU RAM — the user is aware and considering config changes

---

## WHAT IS DONE (this session + prior sessions)

### Production fixes (committed)
0. **Per-battle thread leak (poke-env loop + executor threads)** —
   `net/env.py::_stop_env_loop` + call in `ShowdownEnv.close()`, 6 hermetic
   tests in `net/tests/test_env_teardown.py`, Windows proactor cancel-wake
   quirk handled. Commit `3b51c0d` (branch `bionic/ppo-fix-pack`).
   net/tests: 235 passed, 2 skipped.
1. **B1 all-format calc-vs-sim regression** — `tools/tests/b1_calc_vs_sim_all_formats.js` + pytest driver. 504 pass / 28 warn / 0 FAIL across all 50 formats. Ground truth = @smogon/calc library.
2. **Doubles gameType production bug** — @smogon/calc defaulted `Field.gameType` to 'Singles', giving wrong screen ratio (0.5 instead of 2/3) in ALL doubles formats. Fixed at single injection point in `damage/calc.py` (`_game_type` + `_with_game_type`). 6 new tests in `damage/tests/test_game_type_doubles.py`.
3. **Pre-existing test bugs fixed** — two positional-arg bugs in `test_audit_fixes.py`, VGC level test using a spread move.
4. **Ping-pong `min_run` 3 → 8** — user ruling: 3 was too low for valid stall strategies. Updated in `tools/bc_collect.py`.
5. **Windows cp1252 stdout trap** — fixed with `tools/stdio_guard.safe_stdio()` at all 5 headless entry points.
6. **Fork safeWrite concurrency** — two-phase fix (ENOENT + EPERM) pinned by regression test.
7. **strict_ports for headless runs** — aborts loudly instead of silently attaching to another run's servers.

### Infrastructure (built, committed)
- **C1: M1 pool** — `fitness/m1_pool.py`, opponent pool = {random, heuristic, strategy, strategy_posterior} + frozen snapshots
- **C1b: play_vs_checkpoint** — drives both opponent interfaces (strategy via `.decide`, net via `.act`)
- **BC pipeline (c)** — `tools/bc_collect.py` (ping-pong filter, progress signature, npz save) + `net/bc_warm.py` (deterministic NLL/acc)
- **PPO fix pack (a)** — advantage normalization + minibatch PPO (256, 4 epochs, approx-KL > 0.03 stop) on branch `bionic/ppo-fix-pack`
- **Learnability test** — `tools/learnability_test.py`, 15 rounds × 50 battles, eval every 2 rounds
- **Gate eval** — `tools/gate_eval.py` with C1–C5 controls, Wilson bounds, verdict rules
- **B2: decision_view** — `tools/decision_view.py` for inspecting bot decisions

### Test suites (all green)
- `damage/tests`: 226 passed
- `tools/tests/test_b1_all_formats.py`: 5 passed (504/28/0)
- `fitness/tests`: all green
- `net/tests`: all green
- `pool/tests`: all green (562 total hermetic)
- `gui/tests`: 66 passed

---

## WHAT IS PENDING (next agent's tasks)

### Immediate (in order)
0. **UNATTENDED PIPELINE (final design ~04:45 — user cannot watch, so no pastes):**
   (a) close the Bionic conversation (frees ~10 GB), (b) double-click
   `scratch\START_UNATTENDED.bat` (or paste its one line), (c) walk away.
   The pipeline waits for RAM (probe >=5 GB, runs >=8 GB; 45-min gate then
   REFUSES), then: yardstick probe (CPU 2×2, 49960/49961) → auto GREEN/RED →
   gate Run 1 controls (CUDA 6×2, 49980-49985) → auto GREEN/RED → gate Run 2
   (r70+r80) → verdicts. RED at any stage → STOP + `runs\unattended\PIPELINE.flag`
   (read it first; probe RED = fix the yardstick in `fitness/strategy_agent.py`,
   which is NOT frozen). Log: `runs\unattended\unattended.log`.
   STOP at any time: `scratch\STOP_ALL_TRAINING.bat`.
   Probe RED reading guide (2026-11-25, spec-aligned): HARD gates =
   C4a/C4b mirror ~50% (±15), C5 fallbacks <= max(5, 0.5% of decides),
   0 crashed; C3 heuristic-vs-strategy >0.55 is now a WARNING (the canonical
   spec GATE_CONTROLS_C1_C5.md lists C3 as a control ROW — validity = C1/C2
   only) — review it with the Run-1 table.
   NOTE: the pipeline also WATCHES learnability 13028 (15-min cadence, <=2
   auto-relaunches) — see task 2.
   Manual fallback (still valid): the old YARDSTICK PROBE paste in git history /
   `GATE_CONTROLS_C1_C5.md` §7.
0a. **SPEED DECISION (superseded by 0):** PATH A keep 2×2 (PID 13028) running →
   verdict ~17:00. (PATH B 8×2 relaunch still available:
   `scratch\kill_run13028.ps1` then `scratch\launch_full8.ps1`.)
1. ~~Finish the smoke run~~ **DONE 01:22** — verdict: pipeline verified (0 crashes,
   PPO rows, WR 0.15→0.70 sampled / 0.25→0.60 greedy, ent 1.62→1.58, KL early-stop
   OK). `runs/learnability_20261124T225458Z_smoke/report.json`. Smoke's final
   thread count with OLD code: 166 (leak confirmed, ~2/battle).
2. **Monitor the FULL learnability run** (PID 13028, launched 01:18 with the fix)
   — NOW COVERED BY THE PIPELINE WATCHDOG (15-min cadence; died/incident-#2 →
   scoped node-kill + relaunch, <=2 retries; verdict line → done):
   - **Check progress:** `cat runs/learnability_full6.out.log` +
     `runs/learnability_20261125T001817Z/metrics.jsonl`
   - **Verified so far:** thread count FLAT (61) across battles = fix working;
     err log only has poke-env `CancelledError intercepted` CRITICAL teardown noise
     (harmless; new loop-stop path cancels the message-handler task on close —
     poke-env logs it at CRITICAL, see `ps_client.py:226`).
   - **Still to verify:** round-0 baseline row + round-1 train row land in
     metrics.jsonl (expected ~02:00-02:20); free RAM holds (was ~3.5-6.6 GB,
     pagefile inactive). Then let it run ~10 h; check periodically.
   - Gate: WR vs random ≥ 0.85 within 15 rounds. PASS → proceed to gate. FAIL → diagnose.
   - If it hangs (round 0 stuck > ~20 min): check free RAM FIRST (incident #2
     signature: node idle + python spinning + "Task was destroyed" growing).
   - RAM headroom: this conversation's llama-server (~18 GB) + 60 node workers
     leaves ~3.5-6 GB free. Restarting the conversation (user action) frees the
     KV cache — the full run is detached and does not need the LLM overnight.

3. **Launch GPU gate (b)** — needs user to run (GPU). Paste in `GATE_CONTROLS_C1_C5.md` §7.
   - Controls C1–C5 MUST run before trusting any gate result
   - Validity rules: if C2 (random bot) beats strategy yardstick >25%, or C1 random-init net's upper 95% bound ≥ 0.40 → YARDSTICK_INVALID, fix first
   - Report per (opponent × mode) W/L/D with Wilson bounds, pooled bounds, all controls
   - Always add: "in-distribution only (same 25-team pool as training); not a generalization claim"

4. **BC warm start (c)** — run collection when servers are free:
   ```
   python -m tools.bc_collect --battles 200 --teacher strategy
   ```
   Then train: `python -m net.bc_warm --dataset <path> --steps 500`

5. **Compare (a) vs (c)** — fixed-PPO from scratch vs BC+PPO, ≥400 battles per comparison, same controls C1–C5.

### After gate result (per Claude handout)
- **PASS (≥40% lower bound vs strategy):** propose M2b (search layer)
- **25–40%:** keep training, report which Step 8 checks lag
- **<25%:** stop, run feature ablation + entropy/action-mix diagnostics

### DP-A (diversity/anti-overfitting, from Claude handout §6)
- A1: Holdout 20% teams + 1-2 scripted bots (never used in training)
- A2: Symmetry augmentation (bench permutation, p1/p2 balance)
- A3: Behavior signature per checkpoint (switch rate, hazard rate, etc.)
- Enable AFTER M1 gate result

---

## KEY FILES

| File | Purpose |
|------|---------|
| `GATE_CONTROLS_C1_C5.md` | Gate rules, C1–C5 spec, verdict rules, §7 = GPU paste |
| `OPEN_DESIGN_CHOICES.md` | All design decisions, OC rulings, workflow log (very long) |
| `PLAN.md` | Original build plan (steps 1-8) |
| `damage/calc.py` | Production damage calc (gameType fix here) |
| `damage/node_calc.js` | Node worker for @smogon/calc |
| `belief/evidence.py` | Routes all calcs through calc.py |
| `belief/state.py` | Belief state (observation space) |
| `fitness/m1_pool.py` | M1 opponent pool |
| `fitness/strategy_agent.py` | Strategy bot (yardstick) |
| `tools/learnability_test.py` | (a) learnability gate |
| `tools/gate_eval.py` | (b) gate with controls |
| `tools/bc_collect.py` | (c) BC data collection |
| `net/bc_warm.py` | (c) BC warm-start trainer |
| `net/model.py` | Neural net (PokeNet) |
| `net/env.py` | Battle env (ShowdownEnv) + `_stop_env_loop` teardown fix (3b51c0d) |
| `net/tests/test_env_teardown.py` | Hermetic tests for the loop/executor thread reaping |
| `legality/formats.py` | 50 target formats |
| `legality/lookup.py` | Level rules, type charts |
| `pool/server_pool.py` | Showdown server pool management |

---

## 2026-11-25 — YARDSTICK PROBE CRASH: ROOT CAUSE + FIX

**Symptom:** unattended probe RED — 200/400 battles crashed, all with
`AttributeError: 'StrategyAgent' object has no attribute 'act'`. Only the
C4a/C4b mirror rows (strategy / strategy_posterior as the **p1 subject**)
crashed; the C3 rows (heuristic p1 vs strategy p2) played fine.

**Root cause (contract gap, not RAM/env):** the frozen eval loop
(`gui/m1_gate_main._play_one_battle`) drives the p1 SUBJECT through
`agent.act(obs, acts)` for every subject (wrapped in `_Greedy`), while the
p2 side dispatches `decides_from_battle` brains through `.decide()`. `HeuristicAgent`
satisfies `.act`; `StrategyAgent` only had `.decide(battle, ...)` — so it
crashed as p1. The gate spec says "ANY pool member can be a p1 subject" —
the harness simply lacked the p1 dispatch, and both harness files are frozen.

**Fix (non-frozen files only, one decision pipeline):**
- `net/env.py` (NOT frozen): `build_obs` now stashes the EXACT ctx snapshot
  it encodes, per-side, thread-local (`set_last_ctx` / `last_ctx`). The obs
  contract is unchanged (pinned by `net/tests/test_env_obs.py`).
- `fitness/strategy_agent.py` (NOT frozen): decision core factored into
  `_choose_from_ctx(ctx, acts, fmt)` — `decide()` (live battle) and the NEW
  `act(obs, acts, deterministic)` (p1 harness contract) BOTH converge on the
  SAME `choose_action` path (same 6.6a posterior overrides, D2 hysteresis,
  C5 counters). `act` returns a torch-free `Decision`-shaped object (the
  exact `fitness/heuristic.py` precedent).

**Verification:**
- 91/91 pass: `fitness/tests/test_strategy_agent.py` (6 new act tests incl.
  "act matches decide on the same ctx"), `test_heuristic.py`,
  `net/tests/test_env_obs.py` (obs contract unchanged),
  `league/tests/test_driver.py` (ctx-brain hook).
- 8-battle e2e smoke THROUGH THE FROZEN HARNESS (the exact crash path):
  `control:strategy` 4/4 played 0 crashed; `control:strategy_posterior` 4/4
  played 0 crashed; C5 decides counted, 0 fallbacks.
  `runs/yardstick_actfix_smoke/result.json`.
- Old RED evidence preserved (not deleted): `runs/yardstick_probe_*RED_060709*`,
  `runs/yardstick_probe_20261125_RED/`.

**Probe decision (spec-aligned):** C3 "heuristic beats strategy" (observed
wr 0.60 in the RED run) is now a WARNING, not a failure — the canonical spec
lists C3 as a control row (validity = C1/C2 only). Hard gates remain: crashed
= 0, C4a/C4b mirror ±15 of 0.5, C5 fallbacks <= max(5, 0.5% of decides).

**State after fix:** learnability run healthy (PID 13028, round 4/15, 0
crashes, entropy ~1.5 stable). Pipeline RELAUNCHED 2026-11-25 06:35:57
(powershell PID 3348, alive-verified; probe CPU 2×2 → Run 1 CUDA → Run 2
CUDA). It is RAM-gated: waits for ≥5 GB free (i.e. until the Bionic
conversation is closed), then auto-proceeds; if RAM never frees in 45 min it
REFUSES to launch into a squeeze. Learnability is under its 15-min watchdog
(auto-relaunch, max 2 retries). Log `runs/unattended/unattended.log`, flag
`runs/unattended/PIPELINE.flag`. Stop-everything:
`scratch/STOP_ALL_TRAINING.bat`.

---

## CONSTRAINTS (binding)

- **CPU-only for anything the agent runs** (agent is the LLM on the GPU)
- **Do NOT edit frozen files:** `gui/m1_gate_main.py`, `tools/gate_eval.py`
- **Do NOT `git add -A`;** do NOT commit pre-existing dirty files:
  `bot/requirements.txt`, `bot/team_pools.json`, `check_battles.ps1`,
  `start_selfplay.bat`, `start_servers_10.bat`, `scratch_learnset_probe.js`, `scratch_team.json`
- **Do NOT delete old checkpoints/replays**
- **Ground truth for damage = @smogon/calc library, NOT the fork**
- **Step-by-step, commit per sub-step, hermetic tests after each piece**
- **No GPU training from the agent** (user runs GPU)
- **RAM:** keep free ≥ ~5 GB (llama-server uses ~23 GB of 32 GB total)
- **Do NOT quote round-70 0.72** (winner's curse: best of 8 noisy evals)
- **Do NOT loosen B1 damage-band logic** to hide failures

---

## OPEN ITEMS (need user decision)

| # | Item | Status |
|---|------|--------|
| 1 | GPU gate launch | User to trigger. NOTE: needs the learnability full run to be done/decided first (same box) |
| 2 | llama-server RAM config | **ACTIVE** — root cause of incident #2 (KV growth squeezed the run). User restarted the conversation 2026-11-24 night; keep ≥ ~5 GB free during runs, restart conversation before that |
| 3 | Shaping (PPO reward shaping) | Separate later commit, not yet built |
| 4 | DP-A/B/C/D (diversity) | After M1 gate result |
| 5 | Second format (singles across gens) | After gate PASS + DP-A |
| 6 | Doubles (VGC/champions) | Not yet, per user |

---

## GIT STATE

- Branch: `bionic/ppo-fix-pack`
- Remote: `justincase-vibe/showdown-ai`
- Latest commit: `3b51c0d` (net.env per-battle thread-leak fix + tests)
- `3b51c0d` is LOCAL — push it (project convention: all work pushed; no large files)
- Pre-existing dirty files remain untouched (bot/requirements.txt, bot/team_pools.json,
  check_battles.ps1, start_selfplay.bat, start_servers_10.bat, scratch_learnset_probe.js,
  scratch_team.json) — do NOT commit them
