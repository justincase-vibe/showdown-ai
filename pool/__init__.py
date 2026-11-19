"""Resource pool -- the backbone of the Mission Control GUI (Phase 0).

A bounded pool of Showdown **server processes** + **concurrent battle slots**,
enforcing the machine's RAM ceiling (default: 8 servers x 3 battles/server =
24 envs). Framework-agnostic (web GUI or native desktop GUI). Design in
``MISSION_CONTROL_GUI_MAP.md`` (Phase 0).

    from pool import PoolConfig, ResourcePool
    pool = ResourcePool(PoolConfig(n_servers=4, per_server=2))
    pool.start()
    async with pool.battle() as server_cfg:
        env = ShowdownEnv(server_configuration=server_cfg, ...)
    pool.shutdown()
"""
from .config import PoolConfig
from .slot_pool import EnvSlotPool
from .server_pool import ServerPool
from .pool import ResourcePool

__all__ = ["PoolConfig", "EnvSlotPool", "ServerPool", "ResourcePool"]
