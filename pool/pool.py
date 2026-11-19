"""Top-level resource pool: server processes + bounded battle slots.

This is the **single object** the trainer and the GUI talk to:

    from pool import PoolConfig, ResourcePool
    pool = ResourcePool(PoolConfig(n_servers=4, per_server=2))
    pool.start()                                   # sync: spawn + wait healthy
    with pool.battle_slot() as server_cfg:         # sync front (thread-per-battle)
        env = ShowdownEnv(server_configuration=server_cfg, ...)
        ...run exactly one battle on that server...
    async with pool.battle() as server_cfg: ...     # async front (the GUI's loop)
    pool.status()                                  # dict for the GUI
    pool.shutdown()                                # sync: tree-kill all servers

Long-term separation of concerns:
  * lifecycle (start/shutdown) is **sync + blocking** (process mgmt, belongs in a
    thread / the setup path);
  * the slot core is **thread-based** (poke-env's sync ``step`` API is built for
    thread-per-battle; see ``slot_pool.py``) -- ``battle_slot()`` is that front;
  * ``battle()`` is the async front over the SAME core (``asyncio.to_thread``),
    so the GUI's loop is never blocked and nothing binds to one event loop.
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from typing import Any, Dict, Optional

from .config import PoolConfig
from .server_pool import ServerPool
from .slot_pool import EnvSlotPool

try:  # poke_env is installed in the real env; guard so pure-logic tests can run without it
    from poke_env.ps_client.server_configuration import ServerConfiguration
except Exception:  # pragma: no cover
    ServerConfiguration = None  # type: ignore


class ResourcePool:
    def __init__(self, cfg: Optional[PoolConfig] = None, log=None):
        self.cfg = cfg or PoolConfig()
        self.log = log or (lambda m: None)
        self.servers = ServerPool(self.cfg, log=self.log)
        self.slots = EnvSlotPool(self.cfg.ports, self.cfg.per_server)

    # --- lifecycle (sync) ----------------------------------------------------
    def start(self, wait_healthy: bool = True) -> Dict[int, bool]:
        status = self.servers.start_all(wait_healthy=wait_healthy)
        if not all(status.values()):
            self.log(f"[pool] WARNING: not all servers healthy: {status}")
        self.servers.start_monitor()
        return status

    def shutdown(self) -> None:
        self.servers.stop_all()

    # --- battle slot (sync: the thread-per-battle trainer, P4/P5) -----------
    @contextmanager
    def battle_slot(self):
        """Reserve one slot (BLOCKS while the pool is full -- back-pressure) and
        yield the ready-to-use ``ServerConfiguration`` for the bound server.
        Released automatically on exit, even if the battle raises.

        This is the front for a thread-per-battle trainer: each worker thread
        takes a slot, plays exactly one battle on its bound server, releases.
        poke-env's sync ``reset()``/``step()`` API is designed for exactly this
        (each env owns its own loop + daemon thread)."""
        with self.slots.slot() as (port, _idx):
            yield self._server_cfg(port)

    # --- battle slot (async: the GUI) -----------------------------------------
    @asynccontextmanager
    async def battle(self):
        """The async front over the SAME thread-based core: the blocking
        acquire/release run in worker threads (``asyncio.to_thread``), so the
        GUI's event loop is never blocked and no ``asyncio.Condition`` is bound
        to one loop. One core, two fronts."""
        slot = await asyncio.to_thread(self.slots.acquire)
        try:
            yield self._server_cfg(slot[0])
        finally:
            await asyncio.to_thread(self.slots.release, slot)

    def _server_cfg(self, port: int):
        url = f"ws://{self.cfg.host}:{port}/showdown/websocket"
        if ServerConfiguration is None:
            return {"websocket_url": url, "authentication_url": self.cfg.auth_url}
        return ServerConfiguration(websocket_url=url, authentication_url=self.cfg.auth_url)

    # --- introspection (for the GUI) ----------------------------------------
    def status(self, check_health: bool = True) -> Dict[str, Any]:
        return {
            "servers": self.servers.health_all() if check_health else {p: None for p in self.cfg.ports},
            "max_envs": self.slots.max_envs,
            "free_slots": self.slots.free,
            "in_use_slots": self.slots.in_use,
            "utilization": round(self.slots.utilization, 3),
            "per_server": self.slots.per_server_usage(),
            "config": self.cfg.to_dict(),
        }
