# Mission Control GUI — Side-Project MAP (planned, NOT built)

> Goal: **one Python GUI** that runs the whole pipeline and keeps you in control:
> (1) self-play training — adjustable **games / bots / formats**, (2) fetch **last** or **top-ELO X** replays, (3) **one-click imitation** training on X replays (auto-movesets), (4) **play vs a checkpoint**, (5) live **error log**, (6) **global STOP**.
>
> This is a **map**. Nothing is built until you say go.

---

## Status — what already exists vs what's new

Most of the **backend** is already there and verified. The GUI is mostly **consolidation + 3 real new pieces**: the **resource pool** (8 servers / 24 envs), the **league (multi-bot) mode**, and **replay index+fetch / one-click BC / unified log / global stop**.

| # | Feature | Today | New work | Reuse |
|---|---|---|---|---|
| 1 | Self-play: games / bots / formats | ✅ games + 1 format, **1 bot vs random** (P4). ❌ multi-bot | league (multi-bot) mode — **design pending our discussion**; multi-format | `net/trainer.py`, `gui/launcher.py` |
| 2 | Fetch last / top-ELO X replays | ⚠️ env supports `save_replays` but P4 **doesn't enable it**; no index | enable save + record ELO/winner/ts/bots + **index** + fetch endpoint | `net/env.py`, `gui/server.py` |
| 3 | One-click imitation (auto-movesets) | ✅ BC core **and** D83 moveset-fill are **already automatic** | replay picker + one-click preset + scale cache (wire `corpus/cache.py`) | `net/bc.py`, `net/batch.py` |
| 4 | Play vs a checkpoint | ✅ **done** (browser fixed today) | just a button + open-browser | `net/human_play.py` |
| 5 | Error log | ⚠️ per-run log + runstate | **unified live log** across all subsystems, error-highlighted | `gui/runstate.py` |
| 6 | Global STOP | ⚠️ per-run stop | **kill-all** (trainers + env pool + optional servers) | `gui/launcher.py` |

---

## Hard constraints (design must obey)

- **RAM ceiling: 8 Showdown servers, ≤ 24 concurrent environments** (≈3/server, tested last project). The pool **must enforce** this and back-pressure (queue) when full.
- **You run GPU; I stay CPU** (`CUDA_VISIBLE_DEVICES=-1`). I only CPU-verify.
- **Step-by-step**, each phase has a **gate** before the next.
- **Reuse, don't duplicate** the existing control plane (launcher / runstate / net).
- **Held off:** league *mechanics* (population / pairing / cull) — pending our discussion, per your note.

---

## The resource pool (new infra — Phase 0, the backbone)

- `servers/pool.py` — start/stop/health-check **8** `node pokemon-showdown` on ports **8000–8007**; cap **3 concurrent battles per server** (24 total).
- `envs/pool.py` — **24** acquire/release slots, round-robin across servers; **async** (poke-env `POKE_LOOP`) so 24 battles can be in flight at once (I/O-bound, RAM is the real limit).
- GUI shows live: servers up, **slots free/in-use**, per-server load.
- Global STOP tears the pool down.

---

## GUI framework — DECISION NEEDED

- **A (recommended):** **extend the existing web GUI** (FastAPI + browser @ `:8700`). It already has start/stop/status/metrics/log + the 4 modes (selfplay/bc/eval/play). Lowest risk, least code, already GPU-ready.
- **B:** a **new native desktop GUI** (Tkinter / PyQt / CustomTkinter) wrapping the **same** backend. More code + divergence, but a real desktop window.
- Either way the **backend is reused**; only the front differs.

---

## Phased build (each gate-verifiable, one at a time)

- **Phase 0 — Resource pool + GUI shell.** 8 servers, 24 slots, health, pool-status panel, global-STOP.
  *Gate: 24 concurrent battles run, then clean shutdown; STOP kills all.*
- **Phase 1 — League self-play (multi-bot).** League manager (N bots + ELO + pairing + cull) **reusing** `SelfPlayTrainer`; knobs for games / bots / formats. **Design pending our discussion** (don't lock yet).
  *Gate: N bots train across ≤24 slots, ELO updates, 0 crashes / 100 battles.*
- **Phase 2 — Replay library + fetch.** Save replays (JSONL, ELO/winner/format/ts/bots) + an index; **"last X"** / **"top-ELO X"** endpoints + GUI list/download/view.
  *Gate: fetch returns the correct top/last set.*
- **Phase 3 — One-click imitation.** Replay picker + one-click BC (auto-movesets already work) + wire the scale cache for N≥1000.
  *Gate: one-click BC on X replays → checkpoint + verdict.*
- **Phase 4 — Play + unified log + global STOP.** Play button (opens browser); unified live error log; global STOP.
  *Gate: STOP cleanly kills everything; a forced error shows in the log.*

---

## Open decisions (lock before I build)

1. **GUI framework:** A (extend the web GUI — my rec.) or B (native desktop)?
2. **Multi-bot model:** league / ELO (bots interact & rank) vs. independent parallel bots? *(you said "top-ELO" → leans league; mechanics still deferred)*
3. **Formats:** one format per run vs. multiple simultaneously (separate bot pool per format)?
4. **Replay storage:** save **all** self-play replays (disk grows fast — 26k files historically) vs. keep **only top-ELO**? Disk budget?
5. **Server lifetime:** 8 servers **always-on** vs. **start-on-demand**?
6. **Global STOP scope:** kill training only, or **also** the 8 Showdown servers?

## Risks

- **RAM:** 24 is a hard cap → pool must enforce + back-pressure, and the GUI must show headroom.
- **Replay disk:** HTML replays accumulate fast → need a size cap / pruning policy.
- **Concurrency:** must confirm battles run **async-concurrent** (not sequentially stepped) or 24 slots won't help wall-time.
- **GPU contention:** you own the GPU; I only CPU-test.
- **League mechanics** not yet designed → Phase 1 is gated on our discussion.
