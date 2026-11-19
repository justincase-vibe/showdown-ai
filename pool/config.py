"""Resource-pool configuration.

Long-term choice: **fully config-driven, nothing hardcoded.** The RAM ceiling the
user measured (8 Showdown servers, ~3 concurrent battles per server => 24 total)
is expressed as two independent knobs so it can be raised or lowered per machine
without touching code:

    per_server  -- max concurrent battles ONE server can host (the real RAM limit)
    n_servers   -- how many Showdown server processes to run

    max_envs = n_servers * per_server          # default 8 * 3 = 24
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

# pool/ lives at the project root, so the fork is always a sibling of it.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class PoolConfig:
    # --- the RAM ceiling (raise per_server if you have more RAM) -------------
    n_servers: int = 8
    per_server: int = 3

    # --- where the servers live ---------------------------------------------
    port_base: int = 8000          # servers bind 8000, 8001, ... (8000+n-1)
    host: str = "localhost"        # websocket host the clients connect to
    auth_url: str = "https://play.pokemonshowdown.com/action.php?"
    node_exe: str = "node"
    fork_cwd: Optional[str] = None  # None => <project_root>/pokemon-showdown
    skip_build: bool = True        # dist/ is already built; boot fast

    # --- resilience ----------------------------------------------------------
    auto_restart: bool = True      # revive a crashed server (bounded)
    max_restarts: int = 3          # per server before giving up
    boot_timeout: float = 60.0     # seconds to wait for a server to become healthy

    def __post_init__(self) -> None:
        if self.n_servers < 1:
            raise ValueError("n_servers must be >= 1")
        if self.per_server < 1:
            raise ValueError("per_server must be >= 1")
        if self.port_base < 1 or self.port_base > 65535:
            raise ValueError("port_base must be a valid port")
        if self.n_servers - 1 > (65535 - self.port_base):
            raise ValueError("port range exceeds the maximum port number")

    # --- derived -------------------------------------------------------------
    @property
    def ports(self) -> List[int]:
        return [self.port_base + i for i in range(self.n_servers)]

    @property
    def max_envs(self) -> int:
        return self.n_servers * self.per_server

    @property
    def fork_path(self) -> Path:
        return Path(self.fork_cwd) if self.fork_cwd else PROJECT_ROOT / "pokemon-showdown"

    def to_dict(self) -> dict:
        return {
            "n_servers": self.n_servers,
            "per_server": self.per_server,
            "port_base": self.port_base,
            "host": self.host,
            "skip_build": self.skip_build,
            "auto_restart": self.auto_restart,
            "max_restarts": self.max_restarts,
            "boot_timeout": self.boot_timeout,
            "ports": self.ports,
            "max_envs": self.max_envs,
            "fork_path": str(self.fork_path),
        }
