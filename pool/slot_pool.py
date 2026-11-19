"""Bounded pool of concurrent battle SLOTS (thread-based core).

Each slot is bound to exactly one Showdown server (a ``(port, index)`` pair), so
the **per-server RAM cap is enforced by construction** -- there simply are only
``per_server`` slots per port. When every slot is in use, ``acquire`` **blocks**
(back-pressure), so the trainer can never over-subscribe the machine.

Design notes (long-term):
  * **Thread-based core** (``threading.Condition`` + deque). The natural consumer
    is a **thread-per-battle trainer** -- and that is exactly poke-env's design:
    each ``SinglesEnv`` owns its own event loop in a daemon thread, and its sync
    ``reset()``/``step()`` API is built on ``run_coroutine_threadsafe`` from an
    *arbitrary caller thread*. A shared-asyncio pool would instead bind
    ``asyncio.Condition`` to ONE event loop and break the moment a second loop
    or thread touched it -- so the core is threads, by design, not by accident.
  * The **async GUI** wraps the blocking acquire/release in worker threads
    (``ResourcePool.battle`` uses ``asyncio.to_thread``) -- one core, two fronts,
    zero loop coupling.
  * ``with pool.slot()`` guarantees a slot is released even if the battle
    raises (no leaked slots).
  * ``acquire(timeout=...)`` raises ``TimeoutError`` -- the GUI's "pool is
    saturated" signal instead of an infinite hang.
"""
from __future__ import annotations

import threading
from collections import deque
from contextlib import contextmanager
from typing import Deque, Dict, List, Optional, Set, Tuple

# A slot is (server_port, slot_index_within_that_server).
Slot = Tuple[int, int]


class EnvSlotPool:
    def __init__(self, server_ports, per_server: int):
        self._ports: List[int] = list(server_ports)
        self._per_server = int(per_server)
        if self._per_server < 1:
            raise ValueError("per_server must be >= 1")
        self._all: List[Slot] = [(p, i) for p in self._ports for i in range(self._per_server)]
        self._free: "Deque[Slot]" = deque(self._all)
        self._in_use: Set[Slot] = set()
        self._cond = threading.Condition()
        self.max_envs = len(self._all)
        self.per_server = self._per_server

    # --- acquire / release (blocking, thread-safe) --------------------------
    def acquire(self, timeout: Optional[float] = None) -> Slot:
        """Block until a slot is free and return it. ``timeout=None`` waits
        forever (the trainer's back-pressure); a finite ``timeout`` raises
        ``TimeoutError`` when the pool stays saturated (the GUI's signal)."""
        with self._cond:
            got = self._cond.wait_for(lambda: len(self._free) > 0, timeout=timeout)
            if not got:
                raise TimeoutError("EnvSlotPool.acquire timed out -- the battle-slot pool is saturated")
            slot = self._free.popleft()
            self._in_use.add(slot)
            return slot

    def release(self, slot: Slot) -> None:
        with self._cond:
            if slot in self._in_use:            # idempotent: releasing twice is a no-op
                self._in_use.discard(slot)
                self._free.append(slot)
                self._cond.notify()

    @contextmanager
    def slot(self):
        """Context manager: acquire, yield the slot, ALWAYS release (even on raise)."""
        s = self.acquire()
        try:
            yield s
        finally:
            self.release(s)

    # --- introspection (for the GUI) ----------------------------------------
    @property
    def free(self) -> int:
        return len(self._free)

    @property
    def in_use(self) -> int:
        return len(self._in_use)

    @property
    def utilization(self) -> float:
        return 0.0 if self.max_envs == 0 else self.in_use / self.max_envs

    def per_server_usage(self) -> Dict[int, Dict[str, int]]:
        out: Dict[int, Dict[str, int]] = {p: {"used": 0, "cap": self._per_server} for p in self._ports}
        for (p, _i) in self._in_use:
            if p in out:
                out[p]["used"] += 1
        return out

    def reset(self) -> None:
        """Test/teardown helper: return every slot to the free set."""
        with self._cond:
            self._free = deque(self._all)
            self._in_use = set()
