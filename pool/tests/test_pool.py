"""Phase 0 FAST tests -- no real servers, no GPU, no live network (only a dead-port probe).

Run:  python -m pytest pool/tests/test_pool.py -q
"""
import asyncio

import pytest

from pool import EnvSlotPool, PoolConfig, ResourcePool, ServerPool


# --- PoolConfig -------------------------------------------------------------
def test_config_derivation():
    cfg = PoolConfig(n_servers=8, port_base=8000, per_server=3)
    assert cfg.ports == [8000, 8001, 8002, 8003, 8004, 8005, 8006, 8007]
    assert cfg.max_envs == 24


def test_config_validation():
    with pytest.raises(ValueError):
        PoolConfig(n_servers=0)
    with pytest.raises(ValueError):
        PoolConfig(per_server=0)


def test_config_to_dict_exposes_budget():
    d = PoolConfig(n_servers=4, per_server=2).to_dict()
    assert d["max_envs"] == 8
    assert d["ports"] == [8000, 8001, 8002, 8003]


# --- EnvSlotPool (the critical budget / back-pressure logic) ----------------
def test_slotpool_initial_state():
    p = EnvSlotPool([100, 110], 3)
    assert p.max_envs == 6
    assert p.free == 6
    assert p.in_use == 0
    assert p.utilization == 0.0


def test_slotpool_acquire_release():
    async def main():
        p = EnvSlotPool([100, 110], 3)
        s1 = await p.acquire()
        assert p.in_use == 1 and p.free == 5
        await p.release(s1)
        assert p.in_use == 0 and p.free == 6
    asyncio.run(main())


def test_slotpool_per_server_cap_by_construction():
    async def main():
        p = EnvSlotPool([100, 110], 3)          # exactly 3 slots per port
        held = [await p.acquire() for _ in range(6)]
        usage = {}
        for port, _i in held:
            usage[port] = usage.get(port, 0) + 1
        assert usage == {100: 3, 110: 3}
    asyncio.run(main())


def test_slotpool_backpressure_blocks_when_full():
    async def main():
        p = EnvSlotPool([100, 110], 2)          # max_envs = 4
        held = [await p.acquire() for _ in range(4)]
        assert p.free == 0

        got = {}

        async def fifth():
            got["slot"] = await p.acquire()

        task = asyncio.ensure_future(fifth())
        await asyncio.sleep(0.05)
        assert "slot" not in got                 # still blocked (pool is full)

        await p.release(held[0])                 # free exactly one slot
        await asyncio.wait_for(task, timeout=1.0)
        assert got["slot"] in p._in_use          # it acquired a real slot
    asyncio.run(main())


def test_slotpool_context_manager_releases_on_exception():
    async def main():
        p = EnvSlotPool([100], 3)
        assert p.free == 3
        try:
            async with p.slot():
                assert p.free == 2
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert p.free == 3                       # released despite the exception
    asyncio.run(main())


def test_slotpool_release_is_idempotent():
    async def main():
        p = EnvSlotPool([100], 2)
        s = await p.acquire()
        await p.release(s)
        await p.release(s)                       # second release must be a no-op
        assert p.free == 2 and p.in_use == 0
    asyncio.run(main())


def test_slotpool_per_server_usage_report():
    async def main():
        p = EnvSlotPool([100, 110], 3)
        for _ in range(3):
            await p.acquire()
        usage = p.per_server_usage()
        assert set(usage.keys()) == {100, 110}
        assert sum(v["used"] for v in usage.values()) == 3
        assert all(v["cap"] == 3 for v in usage.values())
    asyncio.run(main())


# --- ServerPool (command / health; no real spawn) ---------------------------
def test_serverpool_cmd_includes_skip_build():
    sp = ServerPool(PoolConfig(skip_build=True))
    cmd = sp._cmd()
    assert "pokemon-showdown" in cmd and "--skip-build" in cmd


def test_serverpool_cmd_without_skip_build():
    sp = ServerPool(PoolConfig(skip_build=False))
    assert "--skip-build" not in sp._cmd()


def test_serverpool_health_dead_port_is_false():
    sp = ServerPool(PoolConfig(host="localhost"))
    assert sp.health(49999) is False             # nothing listening -> False, fast


def test_serverpool_health_all_shape():
    cfg = PoolConfig(n_servers=3, port_base=49990)
    st = ServerPool(cfg).health_all()
    assert set(st.keys()) == {49990, 49991, 49992}
    assert all(v is False for v in st.values())


def test_serverpool_revive_refuses_after_stop():
    """Regression: a revive racing stop_all must never spawn a new process."""
    sp = ServerPool(PoolConfig(n_servers=1, port_base=49899, auto_restart=True))
    sp._stop.set()                        # simulate shutdown in progress
    sp._revive(49899)
    assert 49899 not in sp._procs         # nothing was spawned
    assert sp._restarts[49899] == 0


# --- ResourcePool (wiring) --------------------------------------------------
def test_resourcepool_status_shape():
    rp = ResourcePool(PoolConfig(n_servers=2, port_base=49980, per_server=2))
    st = rp.status(check_health=False)
    assert st["max_envs"] == 4
    assert st["free_slots"] == 4
    assert st["config"]["ports"] == [49980, 49981]
    assert set(st["per_server"].keys()) == {49980, 49981}


def test_resourcepool_server_cfg_points_at_slot_port():
    cfg = ResourcePool(PoolConfig(host="localhost"))._server_cfg(49995)
    url = cfg.websocket_url if hasattr(cfg, "websocket_url") else cfg["websocket_url"]
    assert ":49995/showdown/websocket" in url


def test_resourcepool_battle_yields_cfg_and_frees_slot():
    async def main():
        rp = ResourcePool(PoolConfig(n_servers=1, port_base=49970, per_server=2))
        async with rp.battle() as cfg:
            url = cfg.websocket_url if hasattr(cfg, "websocket_url") else cfg["websocket_url"]
            assert ":49970/showdown/websocket" in url
            assert rp.slots.in_use == 1
        assert rp.slots.in_use == 0              # released after the block
    asyncio.run(main())
