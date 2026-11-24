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
    p = EnvSlotPool([100, 110], 3)
    s1 = p.acquire()
    assert p.in_use == 1 and p.free == 5
    p.release(s1)
    assert p.in_use == 0 and p.free == 6


def test_slotpool_per_server_cap_by_construction():
    p = EnvSlotPool([100, 110], 3)          # exactly 3 slots per port
    held = [p.acquire() for _ in range(6)]
    usage = {}
    for port, _i in held:
        usage[port] = usage.get(port, 0) + 1
    assert usage == {100: 3, 110: 3}


def test_slotpool_backpressure_blocks_when_full():
    import threading
    import time
    p = EnvSlotPool([100, 110], 2)          # max_envs = 4
    held = [p.acquire() for _ in range(4)]
    assert p.free == 0

    got, errs = {}, []

    def fifth():
        try:
            got["slot"] = p.acquire(timeout=5.0)
        except Exception as e:              # noqa: BLE001
            errs.append(e)

    t = threading.Thread(target=fifth, daemon=True)
    t.start()
    time.sleep(0.2)
    assert "slot" not in got and not errs   # still blocked (pool is full)

    p.release(held[0])                      # free exactly one slot
    t.join(timeout=2.0)
    assert not t.is_alive() and not errs
    assert got["slot"] in p._in_use         # it acquired a real slot


def test_slotpool_acquire_timeout_raises():
    p = EnvSlotPool([100], 1)
    held = p.acquire()
    try:
        with pytest.raises(TimeoutError):
            p.acquire(timeout=0.2)
    finally:
        p.release(held)
    assert p.free == 1 and p.in_use == 0     # timed-out acquire leaked nothing; release is clean


def test_slotpool_context_manager_releases_on_exception():
    p = EnvSlotPool([100], 3)
    assert p.free == 3
    try:
        with p.slot():
            assert p.free == 2
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert p.free == 3                       # released despite the exception


def test_slotpool_release_is_idempotent():
    p = EnvSlotPool([100], 2)
    s = p.acquire()
    p.release(s)
    p.release(s)                             # second release must be a no-op
    assert p.free == 2 and p.in_use == 0


def test_slotpool_per_server_usage_report():
    p = EnvSlotPool([100, 110], 3)
    for _ in range(3):
        p.acquire()
    usage = p.per_server_usage()
    assert set(usage.keys()) == {100, 110}
    assert sum(v["used"] for v in usage.values()) == 3
    assert all(v["cap"] == 3 for v in usage.values())


# --- ServerPool (command / health; no real spawn) ---------------------------
def test_serverpool_cmd_includes_skip_build():
    sp = ServerPool(PoolConfig(skip_build=True))
    cmd = sp._cmd()
    assert "pokemon-showdown" in cmd and "--skip-build" in cmd


def test_serverpool_cmd_without_skip_build():
    sp = ServerPool(PoolConfig(skip_build=False))
    assert "--skip-build" not in sp._cmd()


def _free_tcp_port() -> int:
    """Ask the OS for a currently-free TCP port (bind :0, read it back, free it).

    The old tests hardcoded 49999 / 49990 and asserted 'nothing listening' --
    an ENVIRONMENT assumption that broke on 2026-11-24 when the learnability
    run (train ports 49990/49991) was up: test_serverpool_health_all_shape
    failed because the probe found a LIVE server. A dynamically allocated
    port keeps the test's meaning (shape; dead port -> False) without the
    assumption.
    """
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_serverpool_health_dead_port_is_false():
    sp = ServerPool(PoolConfig(host="localhost"))
    assert sp.health(_free_tcp_port()) is False   # nothing listening -> False, fast


def test_serverpool_health_all_shape():
    base = _free_tcp_port()
    cfg = PoolConfig(n_servers=3, port_base=base)
    st = ServerPool(cfg).health_all()
    assert set(st.keys()) == {base, base + 1, base + 2}
    # values are booleans (dead -> False; the 'dead' case itself is covered by
    # test_serverpool_health_dead_port_is_false with a guaranteed-free port)
    assert all(isinstance(v, bool) for v in st.values())


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


def test_resourcepool_battle_slot_yields_cfg_and_frees():
    rp = ResourcePool(PoolConfig(n_servers=1, port_base=49970, per_server=2))
    with rp.battle_slot() as cfg:
        url = cfg.websocket_url if hasattr(cfg, "websocket_url") else cfg["websocket_url"]
        assert ":49970/showdown/websocket" in url
        assert rp.slots.in_use == 1
    assert rp.slots.in_use == 0              # released after the block


def test_resourcepool_battle_slot_releases_on_exception():
    rp = ResourcePool(PoolConfig(n_servers=1, port_base=49972, per_server=2))
    try:
        with rp.battle_slot():
            raise RuntimeError("battle crashed")
    except RuntimeError:
        pass
    assert rp.slots.in_use == 0


def test_resourcepool_async_battle_front():
    """The GUI's async front (asyncio.to_thread over the thread-based core)."""
    async def main():
        rp = ResourcePool(PoolConfig(n_servers=1, port_base=49971, per_server=2))
        async with rp.battle() as cfg:
            url = cfg.websocket_url if hasattr(cfg, "websocket_url") else cfg["websocket_url"]
            assert ":49971/showdown/websocket" in url
            assert rp.slots.in_use == 1
        assert rp.slots.in_use == 0
    asyncio.run(main())
