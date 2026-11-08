# Showdown AI

A local Pokemon Showdown server plus a Python toolkit for building **the greatest Pokemon
AI ever** — a bot that plays *mixed strategy* (sometimes switches, sets
hazards/screens/boosts, anticipates danger) instead of always clicking its strongest move.

**Target:** **Gen 9, `[Gen 9] BSS 3v3`** (id `gen9bss3v3`) — fixed-team 3v3 singles
(bring 6, pick 3, level 100), defined in this project's server fork.

> **📋 Plan, decisions & status → [PLAN.md](PLAN.md)** — the 10-step roadmap, decision log,
> and verified facts. The verbatim step specs live in
> [build_plan_10step.txt](build_plan_10step.txt).
> Legality-module detail → [legality/PLAN.md](legality/PLAN.md).

This folder is the **new build** (2026-11-07). The previous project, `showdown AI`
(one folder up), is kept **read-only as a reference** — its environment wrapper,
battle loop and PPO training loop are reused by later steps.

## Repository layout

| Path | What it is |
|---|---|
| `pokemon-showdown/` | [smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown) server fork, runs locally on **http://localhost:8000**. Custom: `[Gen 9] BSS 3v3` format in `config/formats.ts`. |
| `legality/` | **Step 1 (done):** SQLite lookup layer + legality oracle over the server's own data — moves, species, abilities, items, learnsets, type chart, per-format legality, Smogon common sets. |
| `legality/lookup.py` | The Step 1 API: `get_move`, `get_species`, `legal_moves`, `get_common_sets`, `get_type_chart`. |
| `legality/extract_gen9.js` | Node extractor: server `Dex` → JSON payload (Dex is the ground truth). |
| `legality/ingest_movesets.py` | Smogon common sets (`bot/data/movesets.json`) → `strategy` tables, legality-filtered. |
| `legality/tests/` | `test_step1.py` — 21 tests, the Step 1 gate. |
| `bot/` | poke-env wrapper, PPO training loop, team pools (from the old project; reused by steps 4–7), Smogon sets data in `bot/data/movesets.json`. |

## Quick start

```powershell
# from the project root (Python 3.12, Node 24 available)

# 1. check the legality DB (built: schema v3, 50 formats, 477 Smogon sets)
python -c "from legality import api; import json; print(json.dumps(api.status(), indent=2))"

# 2. run the Step 1 gate
python -m pytest legality/tests -v

# 3. poke around the lookup API
python -m legality.lookup
```

**Running the Showdown server** (needed for battles, not for the legality DB):

```bash
cd pokemon-showdown
node pokemon-showdown start                       # foreground
# or background (survives the shell closing):
nohup node pokemon-showdown start > ../server.log 2>&1 &
```

## Where we are

- **Step 1 — Move/species database: ✅ DONE (2026-11-07).** 21/21 tests green.
  Lookup layer over the server's Gen 9 data + Smogon common sets, with per-format
  legality and the type chart (verified cell-by-cell against this build's sim).
- **Steps 2–10:** not started (damage calc → belief sampling → structured observation →
  network → behavior cloning → PPO+league → depth-1 payoff matrix → Nash solver →
  integration & eval). See the roadmap table in [PLAN.md](PLAN.md).

**One step at a time** — step N+1 starts only when step N's tests are green.
