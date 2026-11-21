"""Bot registry — track bots (checkpoint, name, ELO, W/L) in a JSON file.

The registry is the league's "roster": which bots exist, where their
checkpoints are, and how well they've been doing. It's a simple JSON-backed
store (no database) that the league engine reads/writes.

Design:
  * One JSON file per league (default: ``league/bots.json``)
  * Each bot: ``{name, ckpt_path, elo, wins, losses, draws, last_fought}``
  * ELO: standard K=32, initial 1500 (the Smogon / chess convention)
  * Thread-safe: the league runs CONCURRENT battles (worker threads), so the
    mutating methods serialize under an internal reentrant lock
  * The registry is framework-agnostic (no torch import)

Usage:
    reg = BotRegistry()
    reg.add("my_bot", "scratch/50b_bc_a3f2.pt")
    reg.add("opp_bot", "scratch/2000b_bc_c21d.pt")
    reg.update_result("my_bot", "opp_bot", winner="my_bot")
    standings = reg.standings()
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


#: default registry location (relative to the project root)
DEFAULT_REGISTRY = Path("league") / "bots.json"

#: ELO constants (K=32, initial 1500 — standard)
ELO_K = 32
ELO_INITIAL = 1500


@dataclass
class Bot:
    """One entry in the league roster."""
    name: str
    ckpt_path: str
    elo: int = ELO_INITIAL
    wins: int = 0
    losses: int = 0
    draws: int = 0
    last_fought: Optional[str] = None  # ISO timestamp

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Bot":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class BotRegistry:
    """A JSON-backed roster of league bots with ELO tracking.

    The registry is the single source of truth for "who's in the league
    and how well are they doing." The league engine reads the roster,
    pairs bots, runs battles, and writes results back.

    Thread-safety: the league runs CONCURRENT battles (SelfPlayRun worker
    threads), so the mutating methods (``update_result`` / ``add`` / ``remove``
    / ``save``) serialize under an internal reentrant lock — a concurrent
    battle's read-mutate-persist can no longer interleave with another's.
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else DEFAULT_REGISTRY
        self._bots: Dict[str, Bot] = {}
        # The league runs CONCURRENT battles (SelfPlayRun worker threads all
        # call update_result -> save), so read-mutate-persist must be atomic.
        # RLock (reentrant): a guarded method may call save(), which also takes
        # the lock, without deadlocking.
        self._lock = threading.RLock()
        self._load()

    # ── persistence ──────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load the roster from disk (empty if the file doesn't exist)."""
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for b in data.get("bots", []):
                bot = Bot.from_dict(b)
                self._bots[bot.name] = bot
        except Exception:
            self._bots = {}

    def save(self) -> None:
        """Write the roster to disk (serialized — concurrent battles persist)."""
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {"bots": [b.to_dict() for b in self._bots.values()]}
            self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    # ── roster management ────────────────────────────────────────────────

    def add(self, name: str, ckpt_path: str, elo: int = ELO_INITIAL) -> Bot:
        """Add a bot to the roster (or update if it already exists)."""
        with self._lock:
            bot = Bot(name=name, ckpt_path=ckpt_path, elo=elo)
            self._bots[name] = bot
            self.save()
        return bot

    def remove(self, name: str) -> None:
        """Remove a bot from the roster."""
        with self._lock:
            if name not in self._bots:
                raise KeyError(f"bot {name!r} not in roster")
            del self._bots[name]
            self.save()

    def get(self, name: str) -> Bot:
        """Look up a bot by name."""
        if name not in self._bots:
            raise KeyError(f"bot {name!r} not in roster")
        return self._bots[name]

    def all(self) -> List[Bot]:
        """All bots in the roster."""
        return list(self._bots.values())

    def names(self) -> List[str]:
        """All bot names."""
        return list(self._bots.keys())

    # ── results ──────────────────────────────────────────────────────────

    def update_result(self, a: str, b: str, winner: Optional[str]) -> None:
        """Record a battle result and update ELO (thread-safe).

        ``winner`` is the bot name (``a`` or ``b``), or ``None`` for a draw.
        The league runs concurrent battles, so the read-mutate-persist is
        serialized under the registry lock (RLock — ``save`` re-acquires it).
        """
        with self._lock:
            ba = self.get(a)
            bb = self.get(b)
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # ``_elo_delta(rating_a, rating_b, score)`` = K*(score - E_a) with
            # E_a = a's expected score. So the delta is ALWAYS a's gain, computed
            # with a's ACTUAL score (1.0 win / 0.0 loss / 0.5 draw) — the standard
            # ELO formula. (The old code used ``-K*(1-E_a)`` for a loss, which
            # under-rewarded upsets: an underdog win gave +K*E_a's complement
            # instead of +K*(1-E_a). Magnitudes are pinned by
            # tests/test_registry.py::test_elo_magnitudes_standard.)
            if winner == a:
                ba.wins += 1
                bb.losses += 1
                delta = self._elo_delta(ba.elo, bb.elo, 1.0)
            elif winner == b:
                bb.wins += 1
                ba.losses += 1
                delta = self._elo_delta(ba.elo, bb.elo, 0.0)
            else:  # draw
                ba.draws += 1
                bb.draws += 1
                delta = self._elo_delta(ba.elo, bb.elo, 0.5)

            ba.elo += delta
            bb.elo -= delta
            ba.last_fought = now
            bb.last_fought = now
            self.save()

    @staticmethod
    def _elo_delta(rating_a: int, rating_b: int, score: float) -> int:
        """Standard ELO update. ``score`` = 1.0 (win), 0.5 (draw), 0.0 (loss)."""
        import math
        expected = 1.0 / (1.0 + math.pow(10, (rating_b - rating_a) / 400.0))
        return round(ELO_K * (score - expected))

    # ── standings ────────────────────────────────────────────────────────

    def standings(self) -> List[Dict[str, Any]]:
        """The roster sorted by ELO (descending), then by name."""
        bots = sorted(self._bots.values(), key=lambda b: (-b.elo, b.name))
        return [
            {
                "name": b.name,
                "ckpt": b.ckpt_path,
                "elo": b.elo,
                "wins": b.wins,
                "losses": b.losses,
                "draws": b.draws,
                "win_rate": round(b.wins / max(1, b.wins + b.losses), 3),
                "last_fought": b.last_fought,
            }
            for b in bots
        ]

    def __len__(self) -> int:
        return len(self._bots)

    def __contains__(self, name: str) -> bool:
        return name in self._bots
