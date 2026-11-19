"""Bounded pool of concurrent battle SLOTS.

Each slot is bound to exactly one Showdown server (a ``(port, index)`` pair), so
the **per-server RAM cap is enforced by construction** -- there simply are only
``per_server`` slots per port. When every slot is in use, ``acquire`` **blocks**
(back-pressure), so the trainer can never over-subscribe the machine.

Design notes (long-term):
  * Pure logic + asyncio -- **no process I/O** -- so it is fast and
    deterministic to unit-test with fake ports.
  * ``async with pool.slot()`` guarantees a slot is released even if the battle
    raises (no leaked slots).
  * ``asyncio.Condition`` binds to the running loop lazily, so the object can be
    constructed outside a loop and used inside one (poke-env's POKE_LOOP).
"""
from __future__ import annotations

import asyncio
from collections import deque
from contextlib import asynccontextmanager
from typing import Deque, Dict, List, Set, Tuple

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
        self._cond = asyncio.Condition()
        self.max_envs = len(self._all)
        self.per_server = self._per_server

    # --- acquire / release ---------------------------------------------------
    async def acquire(self) -> Slot:
        async with self._cond:
            await self._cond.wait_for(lambda: len(self._free) > 0)
            slot = self._free.popleft()
            self._in_use.add(slot)
            return slot

    async def release(self, slot: Slot) -> None:
        async with self._cond:
            if slot in self._in_use:            # idempotent: releasing twice is a no-op
                self._in_use.discard(slot)
                self._free.append(slot)
                self._cond.notify()

    @asynccontextmanager
    async def slot(self):
        """Async context manager: acquire, yield the slot, ALWAYS release."""
        s = await self.acquire()
        try:
            yield s
        finally:
            await self.release(s)

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
        self._free = deque(self._all)
        self._in_use = set()
