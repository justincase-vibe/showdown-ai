"""Top-level resource pool: server processes + bounded battle slots.

This is the **single object** the trainer and the GUI talk to:

    from pool import PoolConfig, ResourcePool
    pool = ResourcePool(PoolConfig(n_servers=4, per_server=2))
    pool.start()                                   # sync: spawn + wait healthy
    async with pool.battle() as server_cfg:        # grabs one of the 24 slots
        env = ShowdownEnv(server_configuration=server_cfg, ...)
        ...run exactly one battle on that server...
    pool.status()                                  # dict for the GUI
    pool.shutdown()                                # sync: tree-kill all servers

Long-term separation of concerns:
  * lifecycle (start/shutdown) is **sync + blocking** (process mgmt, belongs in a
    thread / the setup path);
  * slot acquisition is **async** (composes with poke-env's POKE_LOOP);
  * the GUI calls the blocking parts via ``asyncio.to_thread``.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
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

    # --- battle slot (async) -------------------------------------------------
    @asynccontextmanager
    async def battle(self):
        """Reserve one slot (blocking if the pool is full) and yield the
        ready-to-use ``ServerConfiguration`` for the bound server. Released
        automatically on exit, even if the battle raises."""
        async with self.slots.slot() as (port, _idx):
            yield self._server_cfg(port)

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
