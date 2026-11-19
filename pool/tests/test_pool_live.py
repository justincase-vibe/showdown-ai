"""Phase 0 LIVE test: spawn ONE real Showdown server, verify health, use a slot, stop.

Deselectable by name (``test_live_*``). Needs `node` + the fork; ~15-45s.
Uses port **49900** (far from the default 8000-8007) so it never clobbers a live
server, and shuts the server down again.

Run explicitly:  python -m pytest pool/tests/test_pool_live.py -q -s
"""
from pool import PoolConfig, ResourcePool


def test_live_one_server_smoke():
    cfg = PoolConfig(n_servers=1, port_base=49900, per_server=1, boot_timeout=90)
    rp = ResourcePool(cfg, log=print)
    try:
        status = rp.start(wait_healthy=True)
        assert status.get(49900) is True, f"server on :49900 did not become healthy: {status}"

        st = rp.status(check_health=False)
        assert st["max_envs"] == 1 and st["free_slots"] == 1

        cfg2 = rp._server_cfg(49900)
        url = cfg2.websocket_url if hasattr(cfg2, "websocket_url") else cfg2["websocket_url"]
        assert ":49900/showdown/websocket" in url

        # A slot can be taken and released against the real server config.
        import asyncio
        async def use_slot():
            async with rp.battle() as bcfg:
                bu = bcfg.websocket_url if hasattr(bcfg, "websocket_url") else bcfg["websocket_url"]
                assert ":49900/showdown/websocket" in bu
        asyncio.run(use_slot())
    finally:
        rp.shutdown()
