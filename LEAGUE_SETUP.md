# League Self-Play — Setup Guide (status 2026-11-19)

**TL;DR:** "League self-play" has two layers.
- **P4 — single-agent self-play (PPO vs random): ✅ BUILT & READY.** You can launch it on your GPU right now via the GUI.
- **P5 — the actual league (population + ELO + cull/refill + champion): ❌ NOT BUILT YET.** This is the next build. Plan + first step are in §4.

The browser client hang is **fixed** (see §5).

---

## 1. Launch what's ready NOW — P4 single-agent self-play

This trains **one** agent against `opponent="random"`. It is the foundation the league will stand on.

**Option A — GUI (recommended):**
1. Make sure the fork (Showdown server) is up on port 8000 → double-click `start_play_server.bat` (or `node pokemon-showdown` from `pokemon-showdown/`).
2. Double-click `start_gui.bat` (or `python -m gui`). Dashboard: **http://127.0.0.1:8700**.
3. In the **🎓 Train** panel pick mode **Self-play (PPO)**, `device = cuda`.
4. Keep defaults (or tune below), click **Start**.
5. Watch live metrics on the dashboard; **Stop** any time. Closing the window does **not** stop training (it's a detached subprocess).

**Option B — direct CLI (no GUI):**
```
python -m gui.trainer_main --config <run>/config.json --run-dir <run>
```
(config.json is written by the GUI; the GUI path is simpler.)

**Starting hyperparameters (already the defaults in `net/trainer.py` `TrainConfig`):**

| Field | Default | Note |
|---|---|---|
| `battle_format` | `gen9ou` | change if you want another format |
| `battles_per_round` | `50` | you said "start with 50, scale up slowly" |
| `n_rounds` | `10` | a GUI run does this many rounds then stops |
| `max_steps` / battle | `500` | step cap per battle |
| `epochs` | `2` | on-policy (fresh rollouts) |
| `lr` / `gamma` / `lam` / `clip_eps` | `3e-4` / `0.99` / `0.95` / `0.2` | PPO defaults — tune later |
| `save_every` | `1` | checkpoint every round |
| `resume_from` | (empty) | point at a prior `.pt` to continue |

**Checkpoints** land in `runs/<run-id>/checkpoints/selfplay_roundN.pt`. Browse them from the dashboard (🎮 Play / ⚔️ Eval panels list every `.pt`).

**⚠️ GPU rule:** *you* run `cuda` (I run the LLM on the same GPU and must not). I only ever launch things with `CUDA_VISIBLE_DEVICES=-1`.

---

## 2. Play / eval against a checkpoint (works now)
- **🎮 Play in browser:** pick a `.pt`, set bot username (default `checkpointbot`), **Start & open browser** → log in at `http://localhost:8000` → type `/challenge checkpointbot`.
- **⚔️ Eval vs random:** pick a `.pt` + N battles → get a win-rate sanity check.

---

## 3. What "the league" will add (P5) — not built yet
Per PLAN.md §3c / D73 / D76 / D77, league v1 = **population 8→32 agents + ELO + clones/prune**:
- Keep a pool of N agents (start 8, grow toward 32).
- **ELO** ratings; pair higher-elo vs lower-elo (or round-robin per generation).
- **Cull** weakest ~20% each generation; **refill** with mutants (ES mutation of the champion, and/or BC-reseed).
- Track the **champion**; optional **champion transfer** (advantage-weighted delta with a sign-agreement gate, per D58).
- **Gates:** >60% vs max-damage and rising; ≥400 battles per comparison; 100 self-play games with 0 crashes; 5-generation stability; switch/setup rate stays non-zero (alarm <3%).

**First build step (proposed, step-by-step):** a `league/` manager that (a) holds a population of `PokeNet` weights, (b) schedules P4-style battles between drawn pairs, (c) updates ELO and persists a small `league.json` — **reusing the existing `SelfPlayTrainer` + env**, no new PPO math. Nothing is wired to the GUI yet until it's verified.

**Open decisions to lock before I build it:**
1. Population size start (8? 16?) and max (32?).
2. Pairing: ELO-weighted random vs full round-robin per generation.
3. Cull fraction (20%?) and refill source (champion-mutation vs BC-reseed vs both).
4. Do you want **champion transfer** in v1, or defer to v2?
5. Parallelism: how many simultaneous battles (GPU batch vs sequential)?

---

## 4. This session's fix (browser "Loading…" → fixed)
- **Root cause:** `pokemon-showdown/client/config/config.js` was a copy of the **Node** config (`'use strict'; exports.*`), so in the browser the first line threw `ReferenceError: exports is not defined` → `Config` never created → client hung at "Loading…". (The poke-env bot is unaffected — it doesn't load this file.)
- **Fix:** rewrote `client/config/config.js` as a proper **browser** config (`window.Config = …`) pointing at `http://localhost:8000` → `ws://localhost:8000/showdown/websocket`. Bumped the cache-buster to `?v=local3` in `client/index-new.html`.
- **Verified:** config now served browser-valid; all client scripts 200; `/showdown/info` websocket endpoint up. **You confirmed it opened.**
- ⚠️ A `node build` can re-copy the Node config over it — if the client hangs again after a build, restore the browser config (the file has a comment explaining this).
