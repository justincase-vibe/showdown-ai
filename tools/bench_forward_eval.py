"""B3 — ONE clone+advance cost + parallelism feasibility (M1-gate plan).

The approved spec (SMALL ADDITIONS B3):

  > Report the exact "~1–1.5 s/turn" sim-clone cost: time for ONE
  > clone+advance, and whether it can run in parallel across worker
  > processes. Numbers please.

The 2026-11-22 spike measured the SIM-side clone (fork
``State.serializeBattle`` round-trip: ~0.5–0.6 ms/clone+turn; the "1.0–1.5
s/turn" was the WHOLE search, 36 cells x 20 samples). This bench re-measures
the CURRENT belief-side forward-eval unit end to end (the M2b ``step`` will
use exactly these pieces):

  * **clone**   — ``State.copy()`` (deepcopy of the belief ctx);
  * **advance** — feed ONE turn's real log lines through the belief pipeline
                  (``LogAdapter`` + ``BeliefState.update`` + ``get_context``);
  * **encode**  — ``encode_context`` (ctx -> registry tensors);
  * **forward** — ``PokeAgent.forward_obs`` (B=1 net forward, CPU).

  per cell = clone + advance + encode + forward
  search budget = cells x samples x per-cell

plus a **parallelism** measurement: the same encode+forward workload in 1 vs
P spawned processes (torch frees the GIL per process; the question is
whether P workers actually buy ~Px wall-time).

CPU-only (the GPU is the LLM's). Random-init net (architecture-identical to
the trained nets — the cost is weight-count-bound, not value-bound).

Usage (from the repo root):
    python -m tools.bench_forward_eval --log runs/replays/gen9ou.jsonl \
        --turn 3 --n 100 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _read_log(log_path: str) -> str:
    """Accept a raw protocol log OR a league jsonl (first record's ``log``)."""
    if log_path.endswith(".jsonl"):
        with open(log_path, encoding="utf-8") as f:
            rec = json.loads(f.readline())
        return rec["log"]
    with open(log_path, encoding="utf-8") as f:
        return f.read()


def _cut_turn(log_text: str, turn: int):
    """(before_lines, turn_lines) — everything before the turn marker, and the
    marker's own lines up to the next turn (the ONE-turn advance workload)."""
    lines = log_text.splitlines()
    before: List[str] = []
    turn_lines: List[str] = []
    mode = "before"
    for ln in lines:
        s = ln.strip()
        if s == f"|turn|{turn}":
            mode = "turn"
            turn_lines.append(ln)
            continue
        if mode == "turn" and s.startswith("|turn|"):
            break
        (before if mode == "before" else turn_lines).append(ln)
    return before, turn_lines


def _ms(xs: List[float]) -> str:
    med = statistics.median(xs) * 1000.0
    hi = max(xs) * 1000.0
    return f"{med:8.3f} ms (max {hi:9.3f})"


def _worker_forward(ctx: Dict, actions: List[Dict], n_cells: int,
                    seed: int = 0) -> float:
    """Run ``n_cells`` encode+forward evals in THIS process, return wall-seconds.

    Module-level (NOT nested) so ``ProcessPoolExecutor`` / ``spawn`` can pickle
    it — a local function raises ``PicklingError`` on submit. Each worker builds
    its own net (process-safe; the M2b design requires per-process Node pools +
    torch anyway). Encode runs once up front; the timed loop is pure B=1
    forwards — the per-eval unit M2b search will repeat.
    """
    from net.vocab_db import build_vocabs
    from net.agent import make_random_net, PokeAgent
    from net.encoder import encode_context

    v = build_vocabs()
    agent = PokeAgent(make_random_net(v, device="cpu", seed=seed), seed=seed)
    obs = encode_context(ctx, v, actions)
    agent.forward_obs(obs)  # warm — excluded from timing
    t0 = time.perf_counter()
    for _ in range(n_cells):
        agent.forward_obs(obs)
    return time.perf_counter() - t0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--log", required=True)
    ap.add_argument("--fmt", default="gen9ou")
    ap.add_argument("--turn", type=int, default=3)
    ap.add_argument("--n", type=int, default=100, help="timing repetitions")
    ap.add_argument("--workers", type=int, default=4,
                    help="processes for the parallelism check")
    ap.add_argument("--cells", type=int, default=36)
    ap.add_argument("--samples", type=int, default=20)
    ap.add_argument("--parallel-iterations", type=int, default=120,
                    help="forward-eval cells for the parallel wall-time check")
    args = ap.parse_args(argv)

    log_text = _read_log(args.log)
    before, turn_lines = _cut_turn(log_text, args.turn)
    if not turn_lines:
        print(f"ERROR: turn {args.turn} not found in the log")
        return 2
    before_text = "\n".join(before)

    # ── build the mid-battle belief (the state the search would branch from) ──
    from belief.log_events import LogAdapter, Unknown
    from belief.state import BeliefState

    def build_belief() -> BeliefState:
        belief = BeliefState(args.fmt, own_team=None)
        adapter = LogAdapter()
        for line in before_text.splitlines():
            if not line.strip():
                continue
            ev = adapter.feed_line(line)
            if ev is None or isinstance(ev, Unknown):
                continue
            belief.update(ev)
        return belief

    belief0 = build_belief()
    ctx0 = belief0.get_context()
    n_lines = len(turn_lines)
    print(f"[bench] fmt={args.fmt} turn={args.turn} advance workload = "
          f"{n_lines} log lines; ctx own-slots={len(ctx0.get('own') or {})} "
          f"opp-slots={len(ctx0.get('opponent') or {})}")

    from net.world_model import State
    st = State(ctx0)

    # ── 1. clone ──────────────────────────────────────────────────────
    ts_clone = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        st.copy()
        ts_clone.append(time.perf_counter() - t0)
    med_clone = statistics.median(ts_clone)
    print(f"[bench] clone   (State.copy deepcopy)   : {_ms(ts_clone)} x{args.n}")

    # ── 2. advance (belief pipeline, one real turn) ───────────────────
    ts_adv = []
    for _ in range(args.n):
        belief = build_belief()
        adapter = LogAdapter()
        t0 = time.perf_counter()
        for line in turn_lines:
            ev = adapter.feed_line(line)
            if ev is None or isinstance(ev, Unknown):
                continue
            belief.update(ev)
        belief.get_context()
        ts_adv.append(time.perf_counter() - t0)
    med_adv = statistics.median(ts_adv)
    print(f"[bench] advance (feed {n_lines} lines + ctx) : {_ms(ts_adv)} x{args.n}")

    # ── 3. encode + 4. forward (the net pieces) ───────────────────────
    from net.vocab_db import build_vocabs
    from net.agent import make_random_net, PokeAgent
    from net.encoder import encode_context

    vocabs = build_vocabs()
    net = make_random_net(vocabs, device="cpu", seed=0)
    agent = PokeAgent(net, seed=0)
    actions = [{"kind": "move", "move": "earthquake"},
               {"kind": "tera", "move": "earthquake"},
               {"kind": "switch", "species": "sylveon"}]
    obs = encode_context(ctx0, vocabs, actions)
    # warm-up (module lazy inits) — excluded from timing
    agent.forward_obs(obs)

    ts_enc, ts_fwd = [], []
    for _ in range(args.n):
        t0 = time.perf_counter()
        o = encode_context(ctx0, vocabs, actions)
        t1 = time.perf_counter()
        agent.forward_obs(o)
        ts_enc.append(t1 - t0)
        ts_fwd.append(time.perf_counter() - t1)
    print(f"[bench] encode  (ctx -> tensors)        : {_ms(ts_enc)} x{args.n}")
    print(f"[bench] forward (B=1 net, CPU)          : {_ms(ts_fwd)} x{args.n}")

    med_enc = statistics.median(ts_enc)
    med_fwd = statistics.median(ts_fwd)
    per_cell = med_clone + med_adv + med_enc + med_fwd
    cells, samples = args.cells, args.samples
    print(f"[bench] PER CELL (clone+advance+encode+forward) ~ {per_cell*1000:.3f} ms")
    print(f"[bench] SEARCH BUDGET {cells} cells x {samples} samples = "
          f"{cells*samples} forward evals ~ {cells*samples*per_cell:.2f} s/turn "
          f"(single process, CPU, this machine's load)")

    # ── 5. parallelism: same workload, 1 vs P processes ───────────────
    print(f"[bench] PARALLELISM: {args.parallel_iterations} forward evals, "
          f"1 vs {args.workers} spawned processes ...")
    import concurrent.futures as cf
    import multiprocessing as mp

    t1 = _worker_forward(ctx0, actions, args.parallel_iterations)
    with cf.ProcessPoolExecutor(max_workers=args.workers,
                                mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_worker_forward, ctx0, actions,
                          args.parallel_iterations // args.workers)
                for _ in range(args.workers)]
        tpar = max(f.result() for f in futs)  # wall = slowest worker
    print(f"[bench]   1 proc  : {t1:.3f} s for {args.parallel_iterations} evals "
          f"({t1/args.parallel_iterations*1000:.3f} ms/eval)")
    print(f"[bench]   {args.workers} procs: {tpar:.3f} s wall for "
          f"{args.parallel_iterations} evals "
          f"({tpar/args.parallel_iterations*1000:.3f} ms/eval-eq), "
          f"speedup {t1/tpar:.2f}x vs single-process wall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
