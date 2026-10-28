# Showdown AI

A local Pokemon Showdown server plus a Python (poke-env) toolkit for training a
**self-play reinforcement-learning bot** that plays Pokemon.

Design goal: one shared policy that plays across multiple generations and tiers
(target: **Gen 6–9, OU / UU / RU**), learning from self-play. Reward is +1 win / −1 loss.

> **📋 Full plan, decisions & status → [PLAN.md](PLAN.md)** — the imitation → self-play recipe, state design, opponent-inference approach, measured throughput, and next steps.

## Repository layout

| Path | What it is |
|---|---|
| `pokemon-showdown/` | Official [smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown) server, runs locally on **http://localhost:8000**. |
| `bot/env.py` | `ShowdownEnv` (a concrete poke-env battle env) + `TraineeEnv` (single-agent wrapper). Unified 26-action space, 328-dim observation, +1/−1 reward. |
| `bot/obs.py` | `embed_battle(battle)` → fixed 328-dim feature vector (`OBS_DIM=328`). |
| `bot/policy.py` | `MaskedPPO` — masked actor-critic PPO (action masking as a first-class citizen). |
| `bot/train.py` | Self-play training loop (drives N envs, PPO updates). |
| `bot/teams.py` | Team builders that draw teams from a pool (`team_pools.json`). |
| `bot/teamgen.py` | Builds **server-verified** team pools (uses the server's own TeamValidator as the legality oracle). |
| `bot/verify_env.py` | Torch-free end-to-end pipeline check (obs shape, masks, terminal reward, battle completion). |
| `bot/bot.py` | Minimal random-vs-random self-play harness (quick smoke test). |
| `bot/replays/` | HTML replays of finished battles (open in a browser to watch them). |

## Architecture — a two-part system

We split the problem into two decoupled parts. **Part 1 (Playing) is the current focus.**

### Part 1 — PLAYING
Given a fixed team of 6 legal Pokemon, decide each turn: which move to use, which
Pokemon to switch to, and (generation-dependent) whether to trigger Mega / Z-Move /
Dynamax / Tera.
- Trained with masked PPO against a self-play opponent (our own policy, `frozen` or `live`).
- **The AI does NOT choose the team, moves, items, or IVs in this part** — those are
  fixed by the team definition (real Smogon sets). It only controls in-battle decisions.
- One shared policy spans Gen 6–9 because the legal actions for any generation are a
  *prefix* of the 26-action unified space (gen6 = 14 valid, gen7 = 18, …, gen9 = 26),
  so a single action mask makes one policy exact across generations.

### Part 2 — DRAFTING (later)
Choose *which* 6-Pokemon team (and their sets) to use.
- **Design:** a *value head* `V(team) → predicted win-rate` that **predicts how good a
  random team is**.
- **Why this shape:** it lets us **reward the drafter without it playing** — reward = how
  close `V(team)` is to the team's *actual* battle outcome. So the drafter learns to
  *evaluate* team strength from a clean signal, even on random teams.
- **Later use:** we can *use* `V(team)` to search over candidate teams and pick the
  strongest one — i.e. the drafter becomes a learned team selector.
- This is the "learned helplessness" fix from the reference paper: the drafter is
  rewarded for accurate judgment (not just for winning), which is exactly what lets it
  improve on random teams.

### What the AI can / can't do (answers to the design questions)
- **Can it pick bad picks like Ralts?**
  - *Playing part:* No — it plays the 6 mons it's given and can only choose moves/switches
    among them. Ralts never appears because we use strong OU/UU/RU teams.
  - *Drafting part (if built to draft from the full legal pool):* Yes, it could pick Ralts
    early on — and that's precisely the signal that teaches it to stop. `V(team)` will be
    low for Ralts-heavy teams, so the drafter learns to deprioritize weak picks.
- **Can it choose moves / items / IVs?**
  - *Not today.* Moves, items, IVs, EVs, and natures are fixed by the team (from Smogon).
    Extending the AI to *build* sets (choose moves/items/IVs) is a much larger action space
    and is normally a separate model. **Recommendation:** keep sets fixed (real Smogon sets)
    and let the AI learn to *play* them well first; treat team/set composition as the
    drafting layer.

## Team sources & plan

Current working pools (`bot/team_pools.json`, all server-verified): gen6ou (20), gen7ou (21).
These are enough to *play-test*, but the real plan is **real Smogon forum teams**:
- **Target formats:** Gen 6–9 × {OU, UU, RU} →
  `gen6ou, gen6uu, gen6ru, gen7ou, gen7uu, gen7ru, gen8ou, gen8uu, gen8ru, gen9ou, gen9uu, gen9ru`.  (12 formats; all confirmed present on this server.)
- **Two reference documents to build (TODO — later, not yet):**
  1. Per-Pokemon **moveset tables + legality by generation** (so we can generate valid
     sets and reason about what is legal where).
  2. **Pre-built teams by generation** scraped from the Smogon strategy forums (realistic
     sets per tier, so we can start working immediately).
- `teamgen.py`'s server-validator stays useful as the **legality checker** for scraped teams.

## Server & logistics (single machine, CPU-friendly)

- **Throttling stays ON** (per your request — "do not remove throttling").
  `config/config.js` has `nothrottle = false`. The server still caps load (per-IP/per-user gates,
  raised so they no longer cap us, but still **bounded**) — which is a *feature*: it bounds CPU
  load. Cost: "thousands of games" becomes a multi-hour run (fine overnight).
- **Concurrency (target ~10 battles/min):** **4 worker processes / 16 concurrent battles**
  (`simulator: 16`, `config.js`) — bounded, ≤ the 16 cores, CPU-safe. Measured ceiling at 16
  concurrent is ~12–13/min; 20/min would need >16 concurrent → core thrash (we avoid frying the
  CPU). Full details in [PLAN.md](PLAN.md) §6.
- **GPU:** Part 1 PPO can train on the RTX 5090, but **only when you're not using it** —
  start it explicitly (not while I'm working).

### Running the server
```bash
cd pokemon-showdown
node pokemon-showdown start                       # foreground
# or background (survives the shell closing):
nohup node pokemon-showdown start > ../server.log 2>&1 &
```
Stop it: `taskkill //F //T //PID <pid>`  (find the PID via `netstat -ano | grep :8000`).
> Git Bash note: use the `//`-escaped flags for `taskkill`, else `/F` is mangled to a path.

### Quick tests
```bash
# pipeline check (no GPU/torch), 1 battle per format:
python bot/verify_env.py --formats gen6ou,gen7ou,gen7randombattle --battles 1

# minimal random self-play (poke-env Player):
python bot/bot.py gen7ou
```

## Current status

> **Authoritative plan, decisions & status → [PLAN.md](PLAN.md).** Snapshot:

- ✅ Server up (port 8000), throttle **ON** (bounded), simulator = 16.
- ✅ Random-vs-random reliable (gen6ou / gen7ou / gen7randombattle); team pools server-verified.
- ✅ Env + PPO scaffolding in place; throughput measured; design locked (see PLAN.md).
- ⬜ **Next:** `obs.py` upgrade → datamine replays (RU→OU + random) → imitation → self-play → eval.

## Installed tooling

| Tool | Purpose |
|---|---|
| Git | version control |
| Node.js (LTS) + npm | running the Pokemon Showdown server |
| Python 3.12 + pip | the bot |
| [poke-env](https://poke-env.readthedocs.io/) | Python framework for Showdown bots (from PyPI) |
| PyTorch (CUDA) | PPO policy, RTX 5090 |

## Reference

- Chen & Lin, *"Gotta Train 'Em All"* (Stanford CS230, 2018): PPO + hand-crafted features;
  self-play to avoid "learned helplessness."
