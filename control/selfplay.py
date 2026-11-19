"""The self-play RUN ENGINE (Mission Control feature 1's "adjustable games"
half + feature 2's save side + the feature 5/6 building blocks).

What it does — ONE run:
  * starts a resource pool (``pool.ResourcePool``: n_servers x per_server
    slots, the RAM ceiling enforced by construction);
  * plays ``n_battles`` random-vs-random battles across WORKER THREADS
    (thread-per-battle — the verified concurrency model, D93.2), each battle
    reserving a pool slot itself (back-pressure when the pool is full);
  * captures EVERY battle into the ``replays.ReplayLibrary`` (BC-fixture-
    shaped, byte-compatible — the Phase 2 core);
  * speaks the ``gui.runstate`` on-disk protocol (status / heartbeat /
    stop.flag) so the GUI — web or native — drives it identically;
  * is STOP-safe (D80 discipline: the stop is checked BETWEEN battles, a
    running battle always finishes) and fails LOUD on a broken pool
    (``max_consecutive_errors`` — a dead server must not burn n_battles
    worth of errors).

Deliberate scope (held, per the user):
  * **league mechanics** (multi-bot / ELO / pairing) — NOT here; the driver
    hook (``battle_driver``) is where a league driver slots in later;
  * **PPO training** — this engine COLLECTS (battles + replay capture); the
    agent/training wiring is the next step and stays out until it's designed.

No GPU is touched: battles are CPU inference at most (random vs random is
pure legality-path exercise), and the module imports nothing heavy at import
time (``net`` is pulled lazily inside ``vocabs``).
"""
from __future__ import annotations

import dataclasses
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pool import PoolConfig, ResourcePool
from pool.battle import play_battle_on
from replays import ReplayLibrary, make_record

#: the battle driver's signature:
#:   (server_cfg, fmt, seed, vocabs, max_steps, on_finish) -> (a1_won, a2_won, n_steps)
BattleDriver = Callable[..., Any]


def _port_of(server_cfg: Any) -> Optional[str]:
    """Extract the server port from a ServerConfiguration (None if unknown —
    e.g. a fake in tests; the library's content-dedup still protects then)."""
    url = str(getattr(server_cfg, "websocket_url", server_cfg) or "")
    m = re.search(r":(\d+)/", url)
    return m.group(1) if m else None


@dataclass
class SelfPlayConfig:
    """All knobs a self-play run needs (GUI-serializable)."""

    # ── the run ────────────────────────────────────────────────────────
    n_battles: int = 8                  # "adjustable games" (feature 1)
    battle_format: str = "gen9ou"       # "adjustable formats" (one per run for now)
    seed: int = 20261119
    max_steps: int = 200                # per-battle step cap (the driver's)
    # ── the pool (the RAM ceiling is enforced by the pool, not here) ───
    n_servers: int = 2
    per_server: int = 2
    port_base: int = 8000
    boot_timeout: float = 90.0
    n_workers: Optional[int] = None     # default = total slots (full saturation)
    # ── replay capture (feature 2 save side) ───────────────────────────
    save_replays: bool = True
    library_root: Optional[str] = None  # default = ReplayLibrary.DEFAULT_ROOT
    # ── robustness ─────────────────────────────────────────────────────
    max_consecutive_errors: int = 3     # a broken pool fails the run LOUD

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SelfPlayConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})

    def total_slots(self) -> int:
        return max(1, int(self.n_servers)) * max(1, int(self.per_server))

    def worker_count(self) -> int:
        return min(self.total_slots(), int(self.n_workers) if self.n_workers
                   else self.total_slots())


class SelfPlayRun:
    """One self-play run: pool + N worker threads + replay capture + runstate.

    Usage (library):
        run = SelfPlayRun(cfg, run_dir=Path("runs/..."))
        run.start()
        run.wait()                      # blocks until all battles are done
        run.summary()                   # small dict (for the GUI)

    Usage (subprocess): ``python -m control.selfplay_main --config ... --run-dir ...``
    """

    def __init__(self, cfg: SelfPlayConfig,
                 run_dir: Optional[Any] = None,
                 battle_driver: Optional[BattleDriver] = None,
                 library: Optional[ReplayLibrary] = None,
                 pool: Optional[Any] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir is not None else None
        # injectable for hermetic fast tests (a fake driver never touches net,
        # a fake pool never spawns node processes)
        self._driver = battle_driver or play_battle_on
        self._need_vocabs = battle_driver is None
        # capture target: an injected library wins; otherwise BUILD one from
        # the config (save_replays=True) — the hook must never be a silent
        # no-op when the user asked for replay capture (caught by test).
        if library is not None:
            self._library = library if cfg.save_replays else None
        else:
            self._library = ReplayLibrary(cfg.library_root) if cfg.save_replays else None
        self.log = log or (lambda m: None)

        self.pool = pool if pool is not None else ResourcePool(
            PoolConfig(n_servers=cfg.n_servers,
                       per_server=cfg.per_server,
                       port_base=cfg.port_base,
                       boot_timeout=cfg.boot_timeout),
            log=self.log)
        self._vocabs: Any = None
        self._lock = threading.Lock()
        self._next_battle = 0
        self._results: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._stop_reason: Optional[str] = None      # "user" | "error"
        self._threads: List[threading.Thread] = []
        self._consec_errors = 0
        self._vocabs_lock = threading.Lock()
        self.state = "created"          # created -> running -> completed|stopped|failed

    # ── lifecycle ──────────────────────────────────────────────────────────
    def start(self) -> None:
        if self.state != "created":
            raise RuntimeError(f"SelfPlayRun.start: already {self.state!r}")
        self._status(state="running", started=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        try:
            status = self.pool.start(wait_healthy=True)
        except Exception as exc:  # noqa: BLE001 — a start failure must be a clean FAIL
            self.state = "failed"
            self._status(state="failed", error=f"pool start: {type(exc).__name__}: {exc}")
            raise
        if not all(status.values()):
            self.state = "failed"
            self._status(state="failed", error=f"pool not healthy: {status}")
            raise RuntimeError(f"pool not healthy: {status}")
        self.state = "running"
        self._status(state="running", pool=status)
        n_workers = self.cfg.worker_count()
        self.log(f"[run] {self.cfg.n_battles} battles x {n_workers} workers "
                 f"on {self.cfg.n_servers} server(s) (per_server={self.cfg.per_server}) "
                 f"fmt={self.cfg.battle_format}")
        for i in range(n_workers):
            t = threading.Thread(target=self._worker, name=f"selfplay-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)

    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Block until every worker exits (or ``timeout`` s) and finalize the
        run state. Returns the summary."""
        for t in self._threads:
            t.join(timeout=timeout)
        alive = [t for t in self._threads if t.is_alive()]
        user_stop = (self._stop_event.is_set() and self._stop_reason == "user"
                     or (self.run_dir is not None and self._flag_stop_requested()
                         and self._stop_reason is None))
        if alive:
            self._finalize("failed", error=f"timeout with {len(alive)} worker(s) still alive")
        elif self._stop_reason == "error" or self._consec_errors >= self.cfg.max_consecutive_errors:
            # an ERROR-triggered stop is a FAILURE, not a user stop (checked
            # first — request_stop(reason="error") also sets the stop event)
            self._finalize("failed",
                           error=f"{self._consec_errors} consecutive battle errors (pool broken?)")
        elif user_stop:
            self._finalize("stopped")
        else:
            self._finalize("completed")
        return self.summary()

    def request_stop(self, reason: str = "user") -> None:
        """Graceful stop: checked BETWEEN battles (D80) — running battles
        finish. ``reason="error"`` marks an error-triggered stop (a broken
        pool), which must finalize as FAILED, not 'stopped'."""
        with self._lock:
            if self._stop_reason is None:
                self._stop_reason = reason
        self._stop_event.set()
        if self.run_dir is not None:
            try:
                (self.run_dir / "stop.flag").write_text(
                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), encoding="utf-8")
            except OSError:
                pass

    def shutdown(self) -> None:
        """Tear the pool down (idempotent). Call after ``wait()`` (or on error).
        Only kills servers THIS run spawned — the pool reuses (never kills)
        pre-existing ones, e.g. the user's :8000 play server."""
        try:
            self.pool.shutdown()
        except Exception as e:  # noqa: BLE001 — teardown must never raise into the caller
            self.log(f"[run] WARNING: pool shutdown error: {e}")

    # ── workers ────────────────────────────────────────────────────────────
    def _worker(self) -> None:
        while True:
            if self._stop_event.is_set():
                return
            if self.run_dir is not None and self._flag_stop_requested():
                self._stop_event.set()
                return
            with self._lock:
                if self._next_battle >= self.cfg.n_battles:
                    return
                idx = self._next_battle
                self._next_battle += 1
            self._play_one(idx)

    def _play_one(self, idx: int) -> None:
        rec_id: Optional[str] = None
        saved = False
        cap_err = ""
        battle_error: Optional[str] = None
        a1_won = a2_won = None
        steps = 0

        def on_finish(battle: Any) -> None:
            # the capture hook (defensive: a capture bug must NOT kill the
            # battle result — record the error on the result instead)
            nonlocal rec_id, saved, cap_err
            if self._library is None:
                return
            try:
                rec = make_record(battle, self.cfg.battle_format)
                # The fork's battle tag is a PER-SERVER counter (two pool
                # servers can issue the same tag — D95.1), so namespace the
                # record id with the server port: honest + unique.
                port = _port_of(server_cfg)
                if port:
                    rec["id"] = f"{port}-{rec['id']}"
                out = self._library.save(rec)
                rec_id = rec["id"]
                saved = not out["deduped"]
            except Exception as e:  # noqa: BLE001
                cap_err = f"{type(e).__name__}: {e}"

        try:
            with self.pool.battle_slot() as server_cfg:
                a1_won, a2_won, steps = self._driver(
                    server_cfg, fmt=self.cfg.battle_format,
                    seed=self.cfg.seed + idx, vocabs=self._driver_vocabs(),
                    max_steps=self.cfg.max_steps, on_finish=on_finish)
            with self._lock:
                self._consec_errors = 0
        except Exception as e:  # noqa: BLE001 — a battle failure must not kill the run
            battle_error = f"{type(e).__name__}: {e}"
            with self._lock:
                self._consec_errors += 1
                consec = self._consec_errors
            self.log(f"[run] battle {idx} FAILED: {type(e).__name__}: {e}"
                     f"  (consecutive={consec}/{self.cfg.max_consecutive_errors})")
            if consec >= self.cfg.max_consecutive_errors:
                self.request_stop(reason="error")

        result = {"battle": idx, "a1_won": bool(a1_won), "a2_won": bool(a2_won),
                  "steps": int(steps), "record_id": rec_id, "saved": saved,
                  "capture_error": cap_err, "error": battle_error}
        with self._lock:
            self._results.append(result)
            done = len(self._results)
        self._heartbeat(battle=idx, note="done" if result["error"] is None else "error")
        self._status(battles_done=done)
        self.log(f"[run] battle {idx}: a1={result['a1_won']} a2={result['a2_won']} "
                 f"steps={steps} saved={saved}"
                 + (f" CAPTURE-ERR={cap_err!r}" if cap_err else "")
                 + (f" ERR={battle_error!r}" if battle_error else ""))

    # ── introspection ──────────────────────────────────────────────────────
    def results(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._results)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            rs = list(self._results)
        n = len(rs)
        a1 = sum(1 for r in rs if r["a1_won"] and not r["a2_won"])
        a2 = sum(1 for r in rs if r["a2_won"] and not r["a1_won"])
        both = sum(1 for r in rs if r["a1_won"] and r["a2_won"])
        neither = sum(1 for r in rs if not r["a1_won"] and not r["a2_won"])
        return {
            "state": self.state,
            "planned": self.cfg.n_battles,
            "done": n,
            "a1_wins": a1, "a2_wins": a2,
            "both_flagged": both,          # a decided battle must flag exactly one side
            "neither_flagged": neither,    # undecided (max_steps hit) / failed
            "saved": sum(1 for r in rs if r["saved"]),
            "capture_errors": sum(1 for r in rs if r["capture_error"]),
            "errors": sum(1 for r in rs if r["error"]),
            "avg_steps": round(sum(r["steps"] for r in rs) / n, 1) if n else 0.0,
        }

    # ── runstate protocol (gui.runstate-compatible files) ──────────────────
    def _status(self, **fields: Any) -> None:
        if self.run_dir is None:
            return
        try:
            from gui import runstate
            runstate.write_status(self.run_dir, **fields)
        except Exception as e:  # noqa: BLE001 — status writes must never kill the run
            self.log(f"[run] WARNING: status write failed: {e}")

    def _heartbeat(self, battle: int, note: str = "") -> None:
        if self.run_dir is None:
            return
        try:
            from gui import runstate
            runstate.write_heartbeat(self.run_dir, round_i=0, battle=battle, note=note)
        except Exception:  # noqa: BLE001
            pass

    def _flag_stop_requested(self) -> bool:
        return (self.run_dir / "stop.flag").exists()

    def _finalize(self, state: str, error: Optional[str] = None) -> None:
        self.state = state
        fields: Dict[str, Any] = {"state": state}
        if error:
            fields["error"] = error
        fields["stop_requested"] = bool(self._stop_event.is_set()
                                       or (self.run_dir is not None and self._flag_stop_requested()))
        self._status(**fields)
        self.log(f"[run] FINAL: {state} — {self.summary()}")

    # ── vocabs (lazy: only the REAL driver needs them) ────────────────────
    def _driver_vocabs(self) -> Any:
        """Build-once vocabs for the real driver (double-checked lock: N
        worker threads can hit this concurrently on the first battle)."""
        if not self._need_vocabs:
            return None
        if self._vocabs is None:
            with self._vocabs_lock:
                if self._vocabs is None:
                    from net.vocab_db import build_vocabs
                    self._vocabs = build_vocabs()
        return self._vocabs
