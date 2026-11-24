"""Teardown of ``net.env`` — stopping the per-env asyncio loop (the poke-env leak).

poke-env's ``PsEnv`` spawns its event loop on an ANONYMOUS daemon thread
(``Thread(target=self._loop.run_forever, daemon=True)``) and never reaps it
(it only reaps the shared ``POKE_LOOP`` at atexit), so without ``_stop_env_loop``
every battle leaks one live thread + one live loop for the life of the process.

All fast + offline: real asyncio loop on a daemon thread (the exact shape
poke-env spawns), fake long-lived task, no fork / no network / no GPU.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Tuple

from net.env import _stop_env_loop


def _spawn_loop() -> Tuple[asyncio.AbstractEventLoop, threading.Thread]:
    """The exact construction poke-env's PsEnv uses (env.py:265-266)."""
    loop = asyncio.new_event_loop()
    t = threading.Thread(target=loop.run_forever, daemon=True)
    t.start()
    return loop, t


async def _forever() -> None:
    await asyncio.sleep(3600)


def test_stops_loop_cancels_tasks_and_reaps_thread():
    loop, t = _spawn_loop()
    time.sleep(0.1)  # let run_forever start
    fut = asyncio.run_coroutine_threadsafe(_forever(), loop)
    time.sleep(0.1)  # let the task start
    assert not fut.done()

    _stop_env_loop(loop)

    # the previously-leaked thread must now EXIT (run_forever returns after stop)
    t.join(timeout=5)
    assert not t.is_alive(), "env loop thread still alive after close()"
    assert not loop.is_running()
    # the pending task was cancelled (not just orphaned)
    assert fut.done() and fut.cancelled()


def test_thread_count_comes_back_down():
    loop, t = _spawn_loop()
    time.sleep(0.1)
    n_before = threading.active_count()
    _stop_env_loop(loop)
    deadline = time.time() + 5
    while time.time() < deadline and threading.active_count() >= n_before:
        time.sleep(0.05)
    assert threading.active_count() < n_before, "leaked thread not reaped"


def test_default_executor_threads_are_reaped():
    """CPython names the per-loop default-executor workers 'asyncio_N'.
    Stopping the loop does NOT shut that executor down and its workers block
    on the work queue forever — so close() must reap them explicitly, or
    every battle leaks one more."""
    loop, t = _spawn_loop()
    time.sleep(0.1)

    def _blocking():  # runs in the loop's DEFAULT executor
        time.sleep(0.05)
        return "ok"

    async def _run_blocking():
        return await loop.run_in_executor(None, _blocking)

    fut = asyncio.run_coroutine_threadsafe(_run_blocking(), loop)
    assert fut.result(timeout=5) == "ok"
    time.sleep(0.1)

    exec_threads = [th for th in threading.enumerate()
                    if th.name.startswith("asyncio_")]
    assert exec_threads, "precondition: executor thread must exist"

    _stop_env_loop(loop)
    t.join(timeout=5)

    deadline = time.time() + 5
    while time.time() < deadline and any(th.is_alive() for th in exec_threads):
        time.sleep(0.05)
    assert not any(th.is_alive() for th in exec_threads), \
        "default-executor thread leaked"


def test_none_and_closed_loop_are_safe_noops():
    _stop_env_loop(None)  # must not raise

    loop, t = _spawn_loop()
    time.sleep(0.1)
    loop.call_soon_threadsafe(loop.stop)
    t.join(timeout=5)
    loop.close()
    _stop_env_loop(loop)  # already closed: must not raise


def test_stubborn_task_is_bounded_by_timeout():
    """A task that swallows cancellation must not hang teardown forever —
    the timeout caps the wait, the loop is still stopped, the thread exits."""
    loop, t = _spawn_loop()
    time.sleep(0.1)

    async def _stubborn() -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await asyncio.sleep(3600)  # ignores cancellation
            raise

    asyncio.run_coroutine_threadsafe(_stubborn(), loop)
    time.sleep(0.1)

    t0 = time.time()
    _stop_env_loop(loop, timeout=1.0)
    elapsed = time.time() - t0
    assert elapsed < 3.0, f"timeout not honored: {elapsed:.1f}s"
    t.join(timeout=5)
    assert not t.is_alive()


def test_close_calls_stop_env_loop_on_real_env_shape():
    """Integration shape: a ShowdownEnv-like object (agents with ps_client=None
    + a live _loop) goes through ShowdownEnv.close() without raising, and its
    loop thread is reaped. We drive the real close() on a minimal stand-in so
    the ordering (clients first, loop last) stays pinned."""
    import types

    from net.env import ShowdownEnv

    loop, t = _spawn_loop()
    time.sleep(0.1)
    fut = asyncio.run_coroutine_threadsafe(_forever(), loop)
    time.sleep(0.1)

    env = object.__new__(ShowdownEnv)  # skip __init__ (no server, no agents)
    env._loop = loop
    env._challenge_task = None
    env.battle1 = None
    env.battle2 = None
    agent = types.SimpleNamespace(ps_client=None, battle=None,
                                  order_queue=types.SimpleNamespace(empty=lambda: True),
                                  battle_queue=types.SimpleNamespace(empty=lambda: True))
    env.agent1 = agent
    env.agent2 = agent

    env.close()

    t.join(timeout=5)
    assert not t.is_alive(), "close() must reap the env loop thread"
    assert fut.done() and fut.cancelled()
