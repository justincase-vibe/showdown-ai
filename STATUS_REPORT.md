# PROJECT STATUS REPORT — 2026-11-24 (evening)

> For the next agent: read this first, then read the Claude handout
> (`pokemon_ai_master_handoff (1).md` in the scratchpad) for the full
> strategic context. This report is the operational snapshot.

---

## WHERE WE ARE

**Active macro step: M1 GATE (single-learner experiment)**
Per the Claude handout §4, the M1 gate is the active experiment. M2b/M3/M4/M5
are PARKED until the gate result is in.

**Work order (user-approved, from `GATE_CONTROLS_C1_C5.md`):**
1. **(b)** Run gate NOW with controls C1–C5 → **PENDING** (needs GPU)
2. **(a)** PPO fix pack + learnability test → **IN PROGRESS** (see below)
3. **(c)** BC warm start (Arm B) → pipeline built, collection pending
4. **Compare** fixed-PPO vs BC+PPO, ≥400 battles per comparison → PENDING

---

## WHAT IS RUNNING (as of this report)

| Item | Status | PID |
|------|--------|-----|
| Learnability test (a) | **RUNNING** — relaunched after incident (see below) | 11036 |
| GPU gate (b) | **NOT LAUNCHED** — waiting for user | — |
| BC collection (c) | Pipeline ready, not started | — |

### ⚠️ INCIDENT: 58 orphaned node processes (2026-11-24 ~23:10–23:35)

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

**Learnability test details:**
- Command: `python -m tools.learnability_test --rounds 15 --battles-per-round 50`
- Run dir: `runs/learnability_20261124T223712Z/` (current, PID 11036)
- Prior attempt: `runs/learnability_20261124T221036Z/` (stuck, killed)
- Gate: WR vs random ≥ 0.85 within 15 rounds
- Baseline (from prior completed run): 0.740 sampled / 0.620 greedy (init-vs-init mismatch, not a bug)
- Log: `runs/learnability_full5.out.log` / `.err.log`
- **Check progress:** `cat runs/learnability_20261124T223712Z/metrics.jsonl`
- **Kill if needed:** `powershell -NoProfile -Command 'Stop-Process -Id 11036 -Force'`

**If the user launches the GPU gate:**
- Kill the learnability test first: `powershell -NoProfile -Command 'Stop-Process -Id 11036 -Force'`
- Also kill all node processes: `powershell -NoProfile -Command 'Get-Process | Where-Object { $_.ProcessName -eq "node" } | Stop-Process -Force'`
- Gate paste is in `GATE_CONTROLS_C1_C5.md` §7
- The LLM (llama-server) uses ~23 GB CPU RAM — the user is aware and considering config changes

---

## WHAT IS DONE (this session + prior sessions)

### Production fixes (committed + pushed)
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
1. **Check learnability run** (PID 20116): `cat runs/learnability_20261124T221036Z/metrics.jsonl`
   - If it completed: report the numbers (baseline → final WR, entropy, switch share, KL)
   - If it died: diagnose from `runs/learnability_full4.err.log`, fix, relaunch
   - Gate: WR vs random ≥ 0.85 within 15 rounds. If PASS → proceed to gate. If FAIL → diagnose.

2. **Launch GPU gate (b)** — needs user to run (GPU). Paste in `GATE_CONTROLS_C1_C5.md` §7.
   - Controls C1–C5 MUST run before trusting any gate result
   - Validity rules: if C2 (random bot) beats strategy yardstick >25%, or C1 random-init net's upper 95% bound ≥ 0.40 → YARDSTICK_INVALID, fix first
   - Report per (opponent × mode) W/L/D with Wilson bounds, pooled bounds, all controls
   - Always add: "in-distribution only (same 25-team pool as training); not a generalization claim"

3. **BC warm start (c)** — run collection when servers are free:
   ```
   python -m tools.bc_collect --battles 200 --teacher strategy
   ```
   Then train: `python -m net.bc_warm --dataset <path> --steps 500`

4. **Compare (a) vs (c)** — fixed-PPO from scratch vs BC+PPO, ≥400 battles per comparison, same controls C1–C5.

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
| `legality/formats.py` | 50 target formats |
| `legality/lookup.py` | Level rules, type charts |
| `pool/server_pool.py` | Showdown server pool management |

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
| 1 | GPU gate launch | User to trigger (RAM is tight at 8.4 GB free) |
| 2 | llama-server RAM config | User considering `--parallel 1` or removing `--kv-offload` |
| 3 | Shaping (PPO reward shaping) | Separate later commit, not yet built |
| 4 | DP-A/B/C/D (diversity) | After M1 gate result |
| 5 | Second format (singles across gens) | After gate PASS + DP-A |
| 6 | Doubles (VGC/champions) | Not yet, per user |

---

## GIT STATE

- Branch: `bionic/ppo-fix-pack`
- Remote: `justincase-vibe/showdown-ai`
- Latest commit: `a28b73a` (BC collector min_run 3→8)
- All work pushed to GitHub (no large files)
