"""C2 — a thread-safe POOL of long-lived Node calc workers.

The league crash (0xC0000409) was per-request `node` SPAWN churn under 8
concurrent battle threads (each `calc` did `subprocess.run([node, node_calc.js])`
— a fresh process per request). This pool keeps N node processes ALIVE
(`node node_calc.js --server`, the C1 mode) and routes each calc line to an
available worker — NO spawning in steady state.

Contract:
  * ``request(obj)`` where ``obj`` is a single request dict OR an array of them
    (batch). Returns the parsed response (dict, or list for a batch).
  * If a worker DIES mid-request, it is respawned and the request is retried
    ONCE on the fresh worker. A second failure raises ``WorkerError`` so the
    caller (``damage.calc``) can fall back to the one-shot spawn.
  * A calc ERROR (e.g. an illegal species) is NOT a worker death — it is raised
    as ``RuntimeError(msg)`` (the exact one-shot contract), so ``calc.py``'s
    existing error handling is unchanged.
  * The pool is LAZY (created on first ``request``) so importing ``damage``
    spawns nothing. ``shutdown_pool()`` + ``atexit`` reap the node processes on
    clean exit / run STOP.

Env knobs:
  * ``POKE_CALC_POOL``    — worker count (default 8 = the max battle concurrency).
  * ``POKE_CALC_ONESHOT`` — if set (any non-false value), ``request`` raises
    ``WorkerError`` immediately so the caller uses the one-shot path (debugging).
  * ``POKE_CALC_NODE``    — the node executable (default ``node``).
"""
from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, List, Optional

_NODE = os.environ.get("POKE_CALC_NODE", "node")
_SCRIPT = str(Path(os.path.dirname(os.path.abspath(__file__))) / "node_calc.js")
_DEFAULT_SIZE = 8


def _truthy(v: Optional[str]) -> bool:
    return v not in (None, "", "0", "false", "False", "no", "No")


def _pool_size() -> int:
    try:
        return max(1, int(os.environ.get("POKE_CALC_POOL", _DEFAULT_SIZE)))
    except (TypeError, ValueError):
        return _DEFAULT_SIZE


class WorkerError(RuntimeError):
    """A worker died or returned garbage (NOT a calc error — those are RuntimeError)."""


class NodeWorker:
    """One long-lived ``node --server`` process; one request/response at a time.

    ``request`` is synchronous and single-use at a time: write the line, flush,
    read exactly one response line. Any IO failure / EOF / non-JSON means the
    worker is gone -> ``WorkerError`` (the pool respawns + retries)."""

    def __init__(self, node: str = _NODE, script: str = _SCRIPT):
        self.p = subprocess.Popen(
            [node, script, "--server"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)

    def alive(self) -> bool:
        return self.p.poll() is None

    def request(self, obj: Any) -> Any:
        try:
            self.p.stdin.write(json.dumps(obj) + "\n")
            self.p.stdin.flush()
            line = self.p.stdout.readline()
        except (OSError, ValueError) as e:
            raise WorkerError(f"worker IO error: {e}")
        if line == "":
            raise WorkerError("worker closed stdout (died?)")
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            raise WorkerError(f"worker returned non-JSON: {line!r}")
        # A top-level {error} on a SINGLE (non-batch) request is a calc error —
        # the exact one-shot contract (calc.py expects RuntimeError here).
        if isinstance(obj, dict) and isinstance(data, dict) and "error" in data:
            raise RuntimeError(data["error"])
        return data

    def close(self) -> None:
        try:
            self.p.stdin.close()
        except Exception:
            pass
        try:
            self.p.wait(timeout=3)
        except Exception:
            try:
                self.p.kill()
            except Exception:
                pass


class WorkerPool:
    """A fixed-size pool of NodeWorkers, shared across threads.

    The queue is the availability set (always exactly ``size`` workers, alive or
    dead). Each ``request`` checks out ONE worker, uses it, checks it back. A
    dead worker is killed and a fresh one spawned in its place (net-zero size),
    then the request is retried once on the fresh worker.
    """

    def __init__(self, size: int):
        self._size = max(1, int(size))
        self._q: "queue.Queue[NodeWorker]" = queue.Queue()
        self._lock = threading.Lock()
        self._workers: List[NodeWorker] = []
        for _ in range(self._size):
            self._add(NodeWorker())

    def _add(self, w: NodeWorker) -> None:
        self._workers.append(w)
        self._q.put(w)

    def request(self, obj: Any) -> Any:
        w = self._q.get()
        try:
            result = w.request(obj)
        except WorkerError:
            # w is gone: replace it, then retry once on the fresh worker.
            self._replace(w)
            fresh = self._q.get()
            try:
                result = fresh.request(obj)
            finally:
                self._q.put(fresh)
        else:
            self._q.put(w)
            return result
        # reached only if the retry succeeded (a failing retry raises above)
        return result

    def _replace(self, dead: NodeWorker) -> None:
        with self._lock:
            dead.close()
            if dead in self._workers:
                self._workers.remove(dead)
            self._add(NodeWorker())

    def shutdown(self) -> None:
        with self._lock:
            for w in self._workers:
                w.close()
            self._workers.clear()
            while not self._q.empty():
                try:
                    self._q.get_nowait()
                except queue.Empty:
                    break


# ── process-global singleton (lazy) ──────────────────────────────────────────
_POOL: Optional[WorkerPool] = None
_POOL_LOCK = threading.Lock()


def get_pool() -> WorkerPool:
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = WorkerPool(_pool_size())
        return _POOL


def pool_disabled() -> bool:
    """True when POKE_CALC_ONESHOT is set — ``request`` then raises so the
    caller uses the one-shot path (a debugging escape hatch)."""
    return _truthy(os.environ.get("POKE_CALC_ONESHOT"))


def request(obj: Any) -> Any:
    """Route one calc request (object or array) through the pool.

    Raises ``WorkerError`` if the pool is disabled (POKE_CALC_ONESHOT) or keeps
    failing — the caller (``damage.calc``) then falls back to the one-shot spawn.
    Raises ``RuntimeError(msg)`` for a calc error (the one-shot contract)."""
    if pool_disabled():
        raise WorkerError("pool disabled by POKE_CALC_ONESHOT")
    return get_pool().request(obj)


def shutdown_pool() -> None:
    global _POOL
    with _POOL_LOCK:
        if _POOL is not None:
            _POOL.shutdown()
            _POOL = None


atexit.register(shutdown_pool)
