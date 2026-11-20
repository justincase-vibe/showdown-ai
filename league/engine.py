"""League RUN ENGINE — pair the roster, battle the pairs, update ELO.

What it reuses (NO code written twice, D100):
  * ``control.selfplay.SelfPlayRun`` — pool lifecycle, worker threads,
    stop-safety, runstate protocol, fail-loud on a broken pool, replay
    capture — via its ``battle_driver_factory`` extension point;
  * ``league.registry.BotRegistry`` — the roster + ELO bookkeeping;
  * ``league.pairing`` — the pairing strategies (pure functions);
  * ``league.driver.play_agent_battle_on`` — one checkpoint-vs-checkpoint
    battle (the live-play decision path, ``net.agent.PokeAgent``);
  * ``replays.ReplayLibrary`` — every league battle is captured too (the
    league's own battles are legitimate BC fixtures for the loser's style).

What it adds (the league half):
  * roster management (``add_bot``), pairing planning (``plan``),
    checkpoint loading (once, keyed by path — two bots may share one file),
    per-battle agent construction (fresh RNG per battle — thread-safe),
    ELO bookkeeping per pairing (``_record``), and a league ``summary()``
    (standings + pairings played/errored).

Semantics (pinned by tests):
  * one battle per pairing; a pairing whose battle ERRORED (pool/infra) gets
    NO ELO update — it is counted in the summary, not silently a draw;
  * ``a1_won XOR a2_won`` decides the winner; both/neither = DRAW
    (max_steps hit or an undecided end state) — standard ELO score 0.5;
  * default ``greedy=True`` (argmax) so ELO reflects the checkpoints' actual
    strength, not sampling noise.

CPU-first: no torch at import time (``net`` is pulled lazily, same discipline
as ``pool`` / ``control.selfplay``).
"""
from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import pairing
from .registry import BotRegistry, ELO_INITIAL, Bot

__all__ = ["LeagueConfig", "LeagueRun"]


@dataclass
class LeagueConfig:
    """All knobs a league run needs (GUI-serializable).

    Defaults mirror the capture runs (2 servers x 2, port_base 49960) so a
    league run fits the same RAM ceiling by construction."""

    # ── roster + pairing ───────────────────────────────────────────────
    registry_path: Optional[str] = None   # default: league/bots.json (CWD-relative)
    pairing: str = "round_robin"          # "round_robin" | "elo_weighted"
    best_of: int = 1                      # repeat each pairing n times
    # ── the battle ─────────────────────────────────────────────────────
    battle_format: str = "gen9ou"
    seed: int = 20261119
    max_steps: int = 200
    greedy: bool = True                   # argmax policy (deterministic ELO)
    device: str = "cpu"                   # agent inference device (cpu|cuda)
    # ── the pool (the RAM ceiling is enforced by the pool, not here) ───
    n_servers: int = 2
    per_server: int = 2
    port_base: int = 49960
    boot_timeout: float = 90.0
    n_workers: Optional[int] = None       # default = total slots (full saturation)
    # ── capture + robustness ───────────────────────────────────────────
    save_replays: bool = True
    library_root: Optional[str] = None
    max_consecutive_errors: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LeagueConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


class LeagueRun:
    """One league round over a roster.

    Usage (library / GUI):
        run = LeagueRun(cfg, run_dir=Path("runs/..."))
        run.add_bot("alpha", "scratch/50b_bc_a3f2.pt")
        run.add_bot("beta",  "scratch/2000b_bc_c21d.pt")
        run.start()
        summary = run.wait()
        run.shutdown()

    Usage (subprocess): ``python -m league.league_main --config ... --run-dir ...``
    """

    def __init__(self, cfg: LeagueConfig,
                 run_dir: Optional[Any] = None,
                 registry: Optional[BotRegistry] = None,
                 pool: Optional[Any] = None,
                 library: Optional[Any] = None,
                 driver_factory: Optional[Callable[[int], Any]] = None,
                 log: Optional[Callable[[str], None]] = None):
        self.cfg = cfg
        self.run_dir = Path(run_dir) if run_dir is not None else None
        if registry is not None:
            self.registry = registry
        else:
            self.registry = (BotRegistry(Path(cfg.registry_path))
                             if cfg.registry_path else BotRegistry())
        self.log = log or (lambda m: None)
        # injectables (hermetic fast tests: a fake pool, a fake factory —
        # exactly how control.tests.test_selfplay drives SelfPlayRun)
        self._pool = pool
        self._library = library
        self._driver_factory_fn = driver_factory or self._driver_factory
        # state
        self._nets: Dict[str, Any] = {}
        self._sp: Optional[Any] = None    # the SelfPlayRun (the run mechanics)
        self._pairings: List[Tuple[str, str]] = []
        self._results: List[Dict[str, Any]] = []
        self._lk = threading.Lock()
        self.state = "created"

    # ── roster ──────────────────────────────────────────────────────────
    def add_bot(self, name: str, ckpt_path: str,
                elo: int = ELO_INITIAL) -> Bot:
        """Add (or update) a bot on the roster. ``ckpt_path`` must be a
        loadable PokeNet checkpoint (BC or self-play format — both handled)."""
        return self.registry.add(name, ckpt_path, elo)

    def remove_bot(self, name: str) -> None:
        self.registry.remove(name)

    # ── planning ────────────────────────────────────────────────────────
    def plan(self) -> List[Tuple[str, str]]:
        """Resolve the roster into the ordered pairing list (stored on the
        run; battle idx i is played by pairing i)."""
        bots = self.registry.all()
        if len(bots) < 2:
            raise ValueError(
                f"league needs >= 2 bots in the roster (have {len(bots)}: {self.registry.names()})")
        strat = self.cfg.pairing
        n = max(1, int(self.cfg.best_of))
        if strat == "round_robin":
            pairs = pairing.best_of_n(bots, n)
        elif strat == "elo_weighted":
            base = pairing.elo_weighted(bots)
            pairs = [p for _ in range(n) for p in base]
        else:
            raise ValueError(f"unknown pairing strategy {strat!r} "
                             "(want 'round_robin' or 'elo_weighted')")
        self._pairings = list(pairs)
        self.log(f"[league] {len(bots)} bots -> {len(pairs)} pairing(s) "
                 f"(strategy={strat}, best_of={n})")
        return list(pairs)

    # ── checkpoint loading (once, before workers start) ─────────────────
    def _load_nets(self) -> None:
        """Load every distinct checkpoint ONCE (keyed by path — two roster
        bots may share one file; the net is then shared read-only across
        battles, which is safe: eval mode + no_grad forward, no mutation)."""
        if self._nets:
            return
        import torch
        from net.model import PokeNet
        from net.vocab_db import build_vocabs

        vocabs = build_vocabs()
        by_path: Dict[str, Any] = {}
        for name in self.registry.names():
            ckpt = self.registry.get(name).ckpt_path
            if ckpt not in by_path:
                self.log(f"[league] loading checkpoint {name!r} <- {ckpt} "
                         f"(device={self.cfg.device})")
                payload = torch.load(ckpt, map_location=self.cfg.device,
                                     weights_only=True)
                # BOTH checkpoint formats (mirrors net.human_play._load_checkpoint):
                # BC: {state_dict, ...} | self-play: {"net": {state_dict, ...}, ...}
                if "net" in payload and isinstance(payload["net"], dict) \
                        and "state_dict" in payload["net"]:
                    net = PokeNet.load(payload["net"], vocabs,
                                       device=self.cfg.device)
                else:
                    net = PokeNet.load(payload, vocabs,
                                       device=self.cfg.device)
                net.eval()
                by_path[ckpt] = net
            self._nets[name] = by_path[ckpt]
        self.log(f"[league] checkpoints ready: {len(by_path)} file(s), "
                 f"{len(self._nets)} bot(s)")

    # ── the SelfPlayRun extension points (D100) ──────────────────────────
    def _driver_factory(self):
        """``idx -> battle_driver``: the pairing at idx decides whose
        checkpoints fight. The driver ONLY plays — bookkeeping lives in
        :meth:`_on_battle` (the engine-level hook), so no driver can bypass
        the registry."""
        self._load_nets()
        from net.agent import PokeAgent
        from .driver import play_agent_battle_on

        def factory(idx: int):
            a_name, b_name = self._pairings[idx]
            net_a, net_b = self._nets[a_name], self._nets[b_name]
            # a FRESH PokeAgent per battle: a fresh torch.Generator each time,
            # so stochastic play never shares RNG state across worker threads
            # (the shared net itself is read-only — safe to share).
            agent_a = PokeAgent(net_a, seed=self.cfg.seed + 2 * idx,
                                device=self.cfg.device)
            agent_b = PokeAgent(net_b, seed=self.cfg.seed + 2 * idx + 1,
                                device=self.cfg.device)

            def driver(server_cfg, fmt="gen9ou", seed=0, vocabs=None,
                       max_steps=200, on_finish=None):
                return play_agent_battle_on(
                    server_cfg, agent_a=agent_a, agent_b=agent_b,
                    fmt=fmt, seed=seed, vocabs=vocabs, max_steps=max_steps,
                    greedy=self.cfg.greedy, log=self.log, on_finish=on_finish,
                    # D102 guard: N back-to-back decision errors -> the battle
                    # FAILS LOUD (trips the run's max_consecutive_errors) instead
                    # of silently pass-festing all n_battles (the device bug).
                    max_decision_errors=self.cfg.max_consecutive_errors)
            return driver
        return factory

    def _on_battle(self, idx: int, result: Dict[str, Any]) -> None:
        """The engine-level result hook (SelfPlayRun ``on_battle``).

        A pairing whose battle ERRORED (infra/pool) gets NO ELO update — it
        is already counted in the run summary; silently booking it as a draw
        would corrupt the standings."""
        if result.get("error") is not None:
            self.log(f"[league] battle {idx}: ERRORED - no ELO update "
                     f"({result['error']})")
            return
        a_name, b_name = self._pairings[idx]
        self._record(idx, a_name, b_name, result["a1_won"], result["a2_won"],
                     result["steps"])

    # ── ELO bookkeeping ─────────────────────────────────────────────────
    def _record(self, idx: int, a_name: str, b_name: str,
                a1_won: bool, a2_won: bool, steps: int) -> None:
        """Pairing result -> registry. Winner mapping is positional:
        a1 = pairing[0] (``a_name``), a2 = pairing[1] (``b_name``).
        Both/neither = draw (standard ELO score 0.5)."""
        if a1_won and not a2_won:
            winner = a_name
        elif a2_won and not a1_won:
            winner = b_name
        else:
            winner = None  # both flagged (weird) OR undecided (max_steps) -> draw
        row = {"battle": idx, "a": a_name, "b": b_name, "winner": winner,
               "a1_won": bool(a1_won), "a2_won": bool(a2_won), "steps": int(steps)}
        with self._lk:
            self._results.append(row)
        self.registry.update_result(a_name, b_name, winner)
        self.log(f"[league] battle {idx}: {a_name} vs {b_name} -> "
                 f"{'WIN ' + winner if winner else 'DRAW'} (steps={steps})")

    # ── lifecycle (delegated to SelfPlayRun — one run engine) ───────────
    def start(self) -> None:
        if self._sp is not None:
            raise RuntimeError("LeagueRun.start: already started")
        self.plan()
        from control.selfplay import SelfPlayConfig, SelfPlayRun
        sp_cfg = SelfPlayConfig(
            n_battles=len(self._pairings),
            battle_format=self.cfg.battle_format,
            seed=self.cfg.seed,
            max_steps=self.cfg.max_steps,
            n_servers=self.cfg.n_servers,
            per_server=self.cfg.per_server,
            port_base=self.cfg.port_base,
            boot_timeout=self.cfg.boot_timeout,
            n_workers=self.cfg.n_workers,
            save_replays=self.cfg.save_replays,
            library_root=self.cfg.library_root,
            max_consecutive_errors=self.cfg.max_consecutive_errors)
        self._sp = SelfPlayRun(sp_cfg, run_dir=self.run_dir,
                               battle_driver_factory=self._driver_factory_fn(),
                               on_battle=self._on_battle,
                               pool=self._pool, library=self._library, log=self.log)
        self._sp.start()
        self.state = "running"

    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        if self._sp is None:
            raise RuntimeError("LeagueRun.wait: not started")
        s = self._sp.wait(timeout)
        self.state = s.get("state", self.state)
        s["pairings_planned"] = len(self._pairings)
        s["pairings_played"] = len(self._results)
        s["pairings_errored"] = int(s.get("errors", 0))
        s["standings"] = self.registry.standings()
        return s

    def request_stop(self, reason: str = "user") -> None:
        if self._sp is None:
            raise RuntimeError("LeagueRun.request_stop: not started")
        self._sp.request_stop(reason)

    def shutdown(self) -> None:
        if self._sp is not None:
            self._sp.shutdown()

    # ── introspection ───────────────────────────────────────────────────
    def results(self) -> List[Dict[str, Any]]:
        with self._lk:
            return list(self._results)

    def standings(self) -> List[Dict[str, Any]]:
        return self.registry.standings()

    def summary(self) -> Dict[str, Any]:
        """A small dict for the GUI (safe to call before ``wait()``)."""
        with self._lk:
            rs = list(self._results)
        return {
            "state": self.state,
            "pairings_planned": len(self._pairings),
            "pairings_played": len(rs),
            "wins_by_bot": {b: sum(1 for r in rs if r["winner"] == b)
                            for b in self.registry.names()},
            "standings": self.registry.standings(),
        }
