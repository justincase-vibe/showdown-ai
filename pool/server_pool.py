"""Manage the Showdown server PROCESSES (spawn / health / restart / stop).

Long-term design choices:
  * Config-driven (ports, node exe, fork path, skip-build) -- nothing hardcoded.
  * **Health = HTTP 200 on ``/showdown/info``** -- the exact endpoint the browser
    client and poke-env both use, so "healthy" means "actually serving battles".
  * **Bounded auto-restart** (a background thread) so a crash can't fork-bomb the
    machine, and a dead server is surfaced to the log.
  * **Tree-kill** on shutdown (``taskkill /F /T`` on Windows) -- node spawns
    workers, so a plain kill would leak RAM.
  * **Blocking I/O lives here (threads / setup calls), never in the async loop.**
    The GUI will call the blocking parts via ``asyncio.to_thread``.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
import urllib.request
from typing import Dict, List, Optional

from .config import PoolConfig

DETACHED_PROCESS = 0x00000200  # Windows: don't tie the server to a console lifetime


def port_serving(port: int, host: str = "localhost", timeout: float = 2.0) -> bool:
    """True if a Showdown instance answers on ``host:port``.

    THE single health-check implementation (used by :meth:`ServerPool.health`
    and by the ``strict_ports`` preflight): HTTP 200 on ``/showdown/info`` --
    the exact endpoint the browser client and poke-env both use.
    """
    url = f"http://{host}:{port}/showdown/info"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class ServerPool:
    def __init__(self, cfg: PoolConfig, log=None):
        self.cfg = cfg
        self._log = log or (lambda msg: None)
        self._procs: Dict[int, "subprocess.Popen"] = {}
        self._restarts: Dict[int, int] = {p: 0 for p in cfg.ports}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._monitor: Optional[threading.Thread] = None

    # --- command / spawn -----------------------------------------------------
    def _cmd(self) -> List[str]:
        cmd = [self.cfg.node_exe, "pokemon-showdown"]
        if self.cfg.skip_build:
            cmd.append("--skip-build")
        return cmd

    def _spawn(self, port: int) -> "subprocess.Popen":
        env = os.environ.copy()
        env["SHOWDOWN_PORT"] = str(port)
        self._log(f"[pool] starting server on :{port} (fork={self.cfg.fork_path})")
        # Server stdout/stderr go to a per-server log file (scratch/ is untracked):
        # when a server crashes, the monitor only sees "process died" -- the log
        # tells us why. Also keeps a GUI console clean.
        log_dir = self.cfg.fork_path.parent / "scratch" / "pool"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = open(log_dir / f"server_{port}.log", "ab")
        kwargs: Dict = dict(
            cwd=str(self.cfg.fork_path), env=env,
            stdin=subprocess.DEVNULL, stdout=log_file, stderr=log_file,
        )
        if os.name == "nt":
            # Detach from the console so closing a window never kills the servers;
            # we still keep the Popen handle and tree-kill them on shutdown.
            kwargs["creationflags"] = DETACHED_PROCESS
        return subprocess.Popen(self._cmd(), **kwargs)

    def _ensure(self, port: int) -> None:
        with self._lock:
            existing = self._procs.get(port)
        if existing is not None and existing.poll() is None:
            return                      # we already have a live server on this port
        if self.health(port):
            self._log(f"[pool] server already up on :{port} -- reusing it")
            return                      # e.g. a pre-started server; don't double-spawn
        p = self._spawn(port)
        with self._lock:
            self._procs[port] = p

    def start_all(self, wait_healthy: bool = True) -> Dict[int, bool]:
        if self.cfg.strict_ports:
            taken = [p for p in self.cfg.ports if port_serving(p, self.cfg.host)]
            if taken:
                raise SystemExit(
                    f"[pool] ABORT (strict_ports): port(s) {taken} already serve "
                    f"another Showdown instance. A different run (or the GUI) "
                    f"owns them and this pool would silently attach to foreign "
                    f"servers instead of its own.\n"
                    f"  find the owner : netstat -ano | findstr :{taken[0]}\n"
                    f"  stop it        : taskkill /F /PID <pid>\n"
                    f"  or give this run its own range (port_base / --port-base)")
        for port in self.cfg.ports:
            try:
                self._ensure(port)
            except Exception as e:      # never let one bad server block the others
                self._log(f"[pool] FAILED to start server on :{port}: {e}")
        if wait_healthy:
            return self.wait_healthy()
        return self.health_all()

    def wait_healthy(self, timeout: Optional[float] = None) -> Dict[int, bool]:
        timeout = self.cfg.boot_timeout if timeout is None else timeout
        deadline = time.time() + timeout
        status: Dict[int, bool] = {}
        while True:
            status = self.health_all()
            if all(status.values()):
                break
            if time.time() >= deadline:
                break
            time.sleep(0.5)
        self._log(f"[pool] server health: {status}")
        return status

    # --- health --------------------------------------------------------------
    def health(self, port: int, timeout: float = 2.0) -> bool:
        return port_serving(port, self.cfg.host, timeout)

    def health_all(self) -> Dict[int, bool]:
        return {p: self.health(p) for p in self.cfg.ports}

    # --- monitor (bounded auto-restart) -------------------------------------
    def start_monitor(self, interval: float = 3.0) -> None:
        if not self.cfg.auto_restart:
            return
        if self._monitor is not None:
            return
        self._stop.clear()

        def run() -> None:
            while not self._stop.is_set():
                for port in self.cfg.ports:
                    with self._lock:
                        proc = self._procs.get(port)
                    down = proc is None or proc.poll() is not None
                    if down and not self.health(port):
                        self._revive(port)
                self._stop.wait(interval)

        self._monitor = threading.Thread(target=run, name="pool-monitor", daemon=True)
        self._monitor.start()

    def _revive(self, port: int) -> None:
        with self._lock:
            if self._stop.is_set():
                return                     # shutdown in progress: never spawn a new process
            if self._restarts[port] >= self.cfg.max_restarts:
                self._log(f"[pool] server :{port} exceeded max_restarts={self.cfg.max_restarts}; giving up")
                return
            self._restarts[port] += 1
            attempt = self._restarts[port]
        self._log(f"[pool] reviving server :{port} (attempt {attempt})")
        try:
            p = self._spawn(port)
            with self._lock:
                self._procs[port] = p
        except Exception as e:
            self._log(f"[pool] revive FAILED for :{port}: {e}")

    # --- shutdown ------------------------------------------------------------
    def stop_all(self, timeout: float = 10.0) -> None:
        self._stop.set()                   # monitor will not revive anything past this point
        with self._lock:                   # snapshot + clear atomically: no revive can slip in
            procs = list(self._procs.items())
            self._procs.clear()
        for port, proc in procs:
            self._kill_tree(proc, timeout)
        self._log("[pool] all servers stopped")

    def _kill_tree(self, proc: "subprocess.Popen", timeout: float) -> None:
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                proc.kill()
        except Exception as e:
            self._log(f"[pool] force-kill failed for pid {proc.pid}: {e}")
        try:
            proc.wait(timeout=timeout)
        except Exception:
            pass
