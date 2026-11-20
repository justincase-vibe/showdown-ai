"""Self-play run engine — FAST tests (no node servers, no net, no GPU).

The pool and the battle driver are INJECTED fakes (semaphore slots + a fake
driver), so the engine's real logic is exercised hermetically: scheduling,
stop semantics, error handling, capture, runstate files. The real
pool+driver path is covered by ``test_selfplay_live.py``.

Run:  python -m pytest control/tests/test_selfplay.py -q
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from control.selfplay import SelfPlayConfig, SelfPlayRun
from replays import ReplayLibrary


# ── fakes ───────────────────────────────────────────────────────────────────
class FakePool:
    """The pool surface SelfPlayRun uses: start/shutdown + battle_slot()."""

    def __init__(self, slots: int = 3, healthy: bool = True):
        self._sem = threading.BoundedSemaphore(slots)
        self.healthy = healthy
        self.started = False
        self.shutdowns = 0

    def start(self, wait_healthy: bool = True):
        self.started = True
        return {8000: self.healthy}

    def shutdown(self):
        self.shutdowns += 1

    @contextmanager
    def battle_slot(self):
        self._sem.acquire()
        try:
            yield "server-cfg"
        finally:
            self._sem.release()


class FakeBattle:
    """The capture surface make_record uses (duck-typed poke-env Battle)."""

    def __init__(self, tag: str = "gen9ou-fake",
                 log: str = "|turn|1\n|move|p1a|tackle\n|win|p1a",
                 raise_on_build: bool = False):
        self.battle_tag = tag
        self.rating = 100.0
        self.player_role = "p1"
        self.player_username = "alice"
        self.opponent_username = "bob"
        self.won = True
        self._log = log
        self._raise = raise_on_build

    def _build_replay_log(self):
        if self._raise:
            raise RuntimeError("replay build exploded")
        return self._log


def make_driver(run_holder: dict, fail: bool = False,
                capture_raise: bool = False, stop_after: int = 0):
    """A fake battle driver with the REAL signature. ``run_holder`` gets the
    SelfPlayRun after construction so tests can trigger a stop from inside a
    battle (exactly how a GUI stop lands mid-run). The counter is LOCKED —
    an unsynchronized ``dict`` ``+=`` across worker threads can SKIP values
    (LOAD/ADD/STORE is not atomic), which would make ``n == X`` conditions
    flaky (caught: the stop-flag test's unguarded counter skipped 2)."""
    calls = {"n": 0, "lk": threading.Lock()}

    def _bump() -> int:
        with calls["lk"]:
            calls["n"] += 1
            return calls["n"]

    def driver(server_cfg, fmt, seed, vocabs, max_steps, on_finish):
        n = _bump()
        assert server_cfg == "server-cfg"
        if fail:
            raise RuntimeError("battle exploded")
        if on_finish is not None:
            on_finish(FakeBattle(tag=f"gen9ou-{seed}", raise_on_build=capture_raise))
        if stop_after and n >= stop_after:
            run_holder["run"].request_stop()
        return True, False, 3

    driver.calls = calls
    return driver


def _cfg(tmp_path, **kw) -> SelfPlayConfig:
    base = dict(save_replays=True, library_root=str(tmp_path / "lib"),
                n_servers=1, per_server=3, n_workers=3)
    base.update(kw)
    return SelfPlayConfig(**base)


# ── the happy path ──────────────────────────────────────────────────────────
def test_run_completes_and_saves_all(tmp_path):
    holder: dict = {}
    run = SelfPlayRun(_cfg(tmp_path, n_battles=6),
                      battle_driver=make_driver(holder), pool=FakePool(slots=3))
    holder["run"] = run
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["state"] == "completed"
    assert s["done"] == 6 and s["saved"] == 6
    assert s["a1_wins"] == 6 and s["errors"] == 0 and s["capture_errors"] == 0
    lib = ReplayLibrary(str(tmp_path / "lib"))
    assert lib.count() == 6
    assert len({r["id"] for r in lib.fetch_last(10)}) == 6      # no dupes
    assert FakePool is type(run.pool) and run.pool.started and run.pool.shutdowns == 1


def test_save_disabled_writes_nothing(tmp_path):
    holder: dict = {}
    run = SelfPlayRun(_cfg(tmp_path, n_battles=3, save_replays=False),
                      battle_driver=make_driver(holder), pool=FakePool())
    holder["run"] = run
    run.start()
    s = run.wait()
    assert s["saved"] == 0 and s["done"] == 3
    assert not (tmp_path / "lib").exists()


def test_config_roundtrip_and_worker_cap():
    cfg = SelfPlayConfig.from_dict({"n_battles": 4, "n_servers": 2, "per_server": 3,
                                    "n_workers": 99, "bogus_key": "ignored"})
    assert cfg.n_battles == 4 and not hasattr(cfg, "bogus_key")
    assert cfg.total_slots() == 6
    assert cfg.worker_count() == 6            # capped at the pool's slots


# ── stop semantics (D80: between battles, never mid-battle) ─────────────────
def test_graceful_stop_between_battles(tmp_path):
    holder: dict = {}
    run = SelfPlayRun(_cfg(tmp_path, n_battles=20),
                      battle_driver=make_driver(holder, stop_after=4),
                      pool=FakePool(slots=3))
    holder["run"] = run
    run.start()
    s = run.wait()
    assert s["state"] == "stopped"
    # the stop fires DURING battle #4 (cooperative, between-battles): that one
    # finishes, and battles already claimed by the other 2 workers may finish
    # too — but NOTHING claimed after the stop may start.
    assert 4 <= s["done"] <= 6, s
    assert all(r["error"] is None for r in run.results())   # no damaged battles


def test_stop_flag_file_is_respected(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    lk = threading.Lock()
    calls = {"n": 0}

    def driver(server_cfg, fmt, seed, vocabs, max_steps, on_finish):
        with lk:  # locked: an unsynchronized += across threads can skip values
            calls["n"] += 1
            n = calls["n"]
        if on_finish is not None:
            on_finish(FakeBattle(tag=f"gen9ou-{seed}"))
        if n == 2:
            (run_dir / "stop.flag").write_text("now", encoding="utf-8")
        return True, False, 3

    run = SelfPlayRun(_cfg(tmp_path, n_battles=20), run_dir=run_dir,
                      battle_driver=driver, pool=FakePool(slots=3))
    run.start()
    s = run.wait()
    assert s["state"] == "stopped"
    assert 2 <= s["done"] <= 4                # in-flight battles finished first
    st = __import__("json").loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert st["state"] == "stopped" and st["stop_requested"] is True


def test_request_stop_writes_flag(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run = SelfPlayRun(_cfg(tmp_path), run_dir=run_dir,
                      battle_driver=make_driver({}), pool=FakePool())
    run.request_stop()
    assert (run_dir / "stop.flag").exists()


# ── error handling (a broken pool must fail LOUD, not burn the run) ─────────
def test_consecutive_errors_fail_run(tmp_path):
    holder: dict = {}
    run = SelfPlayRun(_cfg(tmp_path, n_battles=50, max_consecutive_errors=3),
                      battle_driver=make_driver(holder, fail=True), pool=FakePool(slots=3))
    holder["run"] = run
    run.start()
    s = run.wait()
    assert s["state"] == "failed"
    assert s["done"] == 3 and s["errors"] == 3            # stopped at the limit
    assert run.pool.shutdowns == 0                        # caller's job (finally block)


def test_capture_error_preserves_battle_result(tmp_path):
    holder: dict = {}
    run = SelfPlayRun(_cfg(tmp_path, n_battles=2),
                      battle_driver=make_driver(holder, capture_raise=True),
                      pool=FakePool())
    holder["run"] = run
    run.start()
    s = run.wait()
    assert s["state"] == "completed"                 # the BATTLE still counts
    assert s["done"] == 2 and s["a1_wins"] == 2
    assert s["capture_errors"] == 2 and s["saved"] == 0
    assert "replay build exploded" in run.results()[0]["capture_error"]


def test_colliding_battle_tags_both_saved(tmp_path):
    # D95.1 end-to-end: two battles with the SAME server tag (a real fork
    # behavior across pool servers) must BOTH land in the library
    holder: dict = {}
    calls = {"n": 0, "lk": threading.Lock()}

    def driver(server_cfg, fmt, seed, vocabs, max_steps, on_finish):
        with calls["lk"]:
            calls["n"] += 1
            n = calls["n"]
        if on_finish is not None:
            on_finish(FakeBattle(
                tag="battle-gen9ou-7",  # identical tag, DIFFERENT battle
                log=f"|turn|1\n|move|p1a|move{n}\n|win|p1a"))
        return True, False, 3

    run = SelfPlayRun(_cfg(tmp_path, n_battles=2), battle_driver=driver, pool=FakePool())
    holder["run"] = run
    run.start()
    s = run.wait()
    assert s["state"] == "completed" and s["saved"] == 2, s
    lib = ReplayLibrary(str(tmp_path / "lib"))
    assert lib.count() == 2
    assert len({r["log"] for r in lib.fetch_last(2)}) == 2


def test_pool_start_failure_is_a_clean_fail(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    holder: dict = {}
    run = SelfPlayRun(_cfg(tmp_path), run_dir=run_dir,
                      battle_driver=make_driver(holder), pool=FakePool(healthy=False))
    holder["run"] = run
    try:
        run.start()
        raise AssertionError("expected pool-unhealthy to raise")
    except RuntimeError:
        pass
    assert run.state == "failed"
    st = __import__("json").loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert st["state"] == "failed" and "pool not healthy" in st["error"]


def test_double_start_rejected(tmp_path):
    holder: dict = {}
    run = SelfPlayRun(_cfg(tmp_path, n_battles=2),
                      battle_driver=make_driver(holder, stop_after=1), pool=FakePool())
    holder["run"] = run
    run.start()
    run.wait()
    try:
        run.start()
        raise AssertionError("expected double start to raise")
    except RuntimeError:
        pass


# ── D100: the per-battle driver factory (the league's extension point) ──────
def test_driver_factory_used_per_battle(tmp_path):
    """battle idx i must be played by the factory's driver for i — every idx
    exactly once (the league fights different checkpoints per pairing)."""
    holder: dict = {}
    seen = {"idxs": [], "lk": threading.Lock()}

    def factory(idx: int):
        base = make_driver(holder)  # a real-signature driver per battle

        def driver(server_cfg, fmt, seed, vocabs, max_steps, on_finish):
            with seen["lk"]:
                seen["idxs"].append(idx)
            return base(server_cfg, fmt, seed, vocabs, max_steps, on_finish)
        return driver

    run = SelfPlayRun(_cfg(tmp_path, n_battles=6, save_replays=False),
                      battle_driver_factory=factory, pool=FakePool(slots=3))
    holder["run"] = run
    run.start()
    s = run.wait()
    run.shutdown()
    assert s["state"] == "completed" and s["done"] == 6
    assert sorted(seen["idxs"]) == [0, 1, 2, 3, 4, 5]   # each battle, exactly once


# ── A6: wait(timeout) is a TOTAL budget, not per-thread ──────────────────────
def test_wait_finite_timeout_is_total_not_per_thread(tmp_path):
    """A6 — a FINITE ``wait(timeout)`` bounds the WHOLE wait, not each thread.
    With 2 workers both stuck, a 0.5s budget must NOT stretch to ~1.0s
    (2 x 0.5) the way the old per-thread ``t.join(timeout)`` did."""
    release = threading.Event()

    def stuck_driver(server_cfg, fmt, seed, vocabs, max_steps, on_finish):
        release.wait(timeout=3)  # block well past the 0.5s test budget
        return True, False, 3

    holder: dict = {}
    # exactly 2 worker threads, both grabbing a (stuck) battle
    run = SelfPlayRun(_cfg(tmp_path, n_battles=2, per_server=2, n_workers=2,
                           save_replays=False),
                      battle_driver=stuck_driver, pool=FakePool(slots=2))
    holder["run"] = run
    run.start()
    t0 = time.monotonic()
    s = run.wait(timeout=0.5)
    elapsed = time.monotonic() - t0
    release.set()          # let the daemon threads finish (cleanup)
    run.shutdown()
    assert s["state"] == "failed"          # timed out: workers still alive
    assert elapsed < 0.85, f"wait() took {elapsed:.2f}s — looks per-thread, not total"


def test_wait_none_still_blocks_until_done(tmp_path):
    """A6 guard — the production path (``timeout=None``) is UNCHANGED: it blocks
    until every battle completes and reports ``completed``."""
    holder: dict = {}
    run = SelfPlayRun(_cfg(tmp_path, n_battles=4, save_replays=False),
                      battle_driver=make_driver(holder), pool=FakePool(slots=2))
    holder["run"] = run
    run.start()
    s = run.wait()   # None timeout -> blocks until all 4 done
    run.shutdown()
    assert s["state"] == "completed" and s["done"] == 4
