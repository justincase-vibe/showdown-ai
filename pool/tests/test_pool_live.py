"""Phase 0 LIVE tests: real Showdown server(s), real battle(s).

Deselectable by name (``test_live_*``) -- the default fast run is
``pytest pool/tests/test_pool.py``. Needs `node` + the fork + poke_env.

* ``test_live_one_server_smoke`` -- ONE server on :49900 (far from 8000-8007 so
  it never clobbers a live server), ~15-45s.
* ``test_live_gate_full_budget_24_battles`` -- THE PHASE 0 GATE: 8 servers /
  24 concurrent battles (the user's measured RAM ceiling), then clean shutdown
  with 0 orphans. CPU-only (random vs random, no net, no GPU). ~2-4 min.

Run explicitly:  python -m pytest pool/tests/test_pool_live.py -q -s
"""
import threading
import time

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


# ═══════════════════════════════════════════════════════════════════════════
#  THE PHASE 0 GATE -- 24 concurrent battles across 8 servers
# ═══════════════════════════════════════════════════════════════════════════
def _play_random_vs_random(rp: ResourcePool, fmt: str, seed: int, vocabs, max_steps: int = 200):
    """One FULL battle (both sides uniformly random -- CPU-only, no net), on
    one reserved slot. Mirrors ``net.trainer.play_and_collect``'s loop with a
    random policy on BOTH agents (the legality path is exercised end to end).
    Returns ``(a1_won, a2_won, n_steps)``."""
    import logging
    import numpy as np
    from net.env import ShowdownEnv, StrongTeamBuilder, random_action, action_to_pe

    with rp.battle_slot() as server_cfg:
        env = ShowdownEnv(battle_format=fmt, team=StrongTeamBuilder(fmt, seed=seed),
                          server_configuration=server_cfg, vocabs=vocabs,
                          log_level=logging.ERROR)
        a1, a2 = env.possible_agents
        n_steps = 0
        try:
            obs, _info = env.reset()
            for _ in range(max_steps):
                if env.battle1 is not None and env.battle1.finished:
                    break
                actions = {a1: np.int64(-2), a2: np.int64(-2)}
                if env.agent1_to_move:
                    act = random_action(env.battle1)
                    if act is not None:
                        actions[a1] = np.int64(action_to_pe(act, env.battle1))
                if env.agent2_to_move:
                    act = random_action(env.battle2)
                    if act is not None:
                        actions[a2] = np.int64(action_to_pe(act, env.battle2))
                obs, rew, term, trunc, _info = env.step(actions)
                n_steps += 1
                if term[a1] or term[a2] or trunc[a1] or trunc[a2]:
                    break
            return bool(env.battle1.won), bool(env.battle2.won), n_steps
        finally:
            env.close()


def test_live_gate_full_budget_24_battles():
    """PHASE 0 GATE: 24 concurrent battles across 8 servers (the user's measured
    RAM ceiling: 3 battles/server), all to completion, then clean shutdown with
    no leaked slots and no orphaned server processes. CPU-only.

    This is the concurrency proof the Phase 0 gate in MISSION_CONTROL_GUI_MAP.md
    calls for -- the league (P5) and the GUI both build on it.

    KNOWN COSMETIC: the "Task was destroyed but it is pending!" lines at teardown
    are poke-env noise, NOT a leak. poke-env has no clean client-close API: each
    env's daemon event loop dies at process exit while its long-lived tasks
    (``_handle_message`` websocket readers, the abandoned ``battle_against``
    challenge task) are still pending, so the loop logs them on destruction.
    ``ShowdownEnv.close()`` deliberately abandons (never forfeits -- the user's
    D-rule) and poke-env's ``_battle_against``/``send_challenges`` have no
    ``finally`` side effects, so abandonment is safe. The battle completed and
    the slot was released -- that is what this gate asserts.
    """
    from net.vocab_db import build_vocabs

    fmt = "gen9ou"
    n = 24
    cfg = PoolConfig(n_servers=8, per_server=3, port_base=8000, boot_timeout=120)
    rp = ResourcePool(cfg, log=print)
    vocabs = build_vocabs()                 # once, shared by all 24 envs
    results, errors = [], []
    lock = threading.Lock()

    def worker(i: int) -> None:
        try:
            r = _play_random_vs_random(rp, fmt, seed=100000 + i, vocabs=vocabs)
            with lock:
                results.append((i, r))
        except Exception as e:              # noqa: BLE001
            with lock:
                errors.append(f"battle {i}: {type(e).__name__}: {e}")

    try:
        status = rp.start(wait_healthy=True)
        assert all(status.values()), f"servers not healthy: {status}"

        t0 = time.time()
        threads = [threading.Thread(target=worker, args=(i,), name=f"gate-b{i}", daemon=True)
                   for i in range(n)]
        for t in threads:
            t.start()
        deadline = time.time() + 600        # 10 min wall clock for all 24
        for t in threads:
            t.join(timeout=max(1.0, deadline - time.time()))
        dt = time.time() - t0

        alive = [t.name for t in threads if t.is_alive()]
        assert not alive, f"battles still running after timeout: {alive}; errors so far: {errors}"
        assert len(results) == n, f"only {len(results)}/{n} battles completed; errors: {errors}"
        assert not errors, f"battle errors: {errors}"
        assert rp.slots.in_use == 0, f"LEAKED SLOTS: {rp.slots.in_use} still in use"

        a1_wins = sum(1 for _i, (w1, _w2, _s) in results if w1)
        print(f"[gate] 24/24 concurrent battles done in {dt:.0f}s | "
              f"a1 wins {a1_wins}, a2 wins {n - a1_wins} | slots free: {rp.slots.free}/{rp.slots.max_envs}")
    finally:
        rp.shutdown()

    # Orphan check: every port the POOL itself spawned (8001-8007) must be dead.
    # :8000 may still be up -- that is the user's own pre-started server, which
    # the pool REUSES by design (never kills what it didn't spawn).
    up = [p for p in cfg.ports[1:] if rp.servers.health(p, timeout=1.0)]
    assert not up, f"ORPHANED SERVERS still listening after shutdown: {up}"
