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
    # ── the learning loop (D103) ───────────────────────────────────────
    learning: bool = True                 # True: bots buffer experience (the league
                                         # IS the self-improving loop); False: the
                                         # legacy eval-only run (rank checkpoints,
                                         # no capture) — one engine, a mode flag
    # ── the PPO update (D103 step 3; defaults mirror net.trainer.TrainConfig) ──
    n_rounds: int = 1                     # play+update cycles per GUI launch (1 = one)
    epochs: int = 2                       # 1–2 on-policy PPO passes per round-end update
    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_norm: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
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
        self._bots: Dict[str, Any] = {}  # learning mode: name -> LearningBot
        self._collectors: Dict[int, Tuple[Any, Any]] = {}  # pairing idx -> (coll_a, coll_b)
        self._sp: Optional[Any] = None    # the CURRENT round's SelfPlayRun
        self._pairings: List[Tuple[str, str]] = []
        self._results: List[Dict[str, Any]] = []
        self._lk = threading.Lock()
        self.state = "created"
        # multi-round (D103 step 3): the shared pool is reused across rounds
        # (servers boot once, not per round); bookkeeping below survives the
        # per-round buffer drain so the summary reports CUMULATIVE learning.
        self._started = False
        self._n_rounds = 0
        self._round_idx = 0
        self._stop_requested = False
        self._total_steps: Dict[str, int] = {}        # cumulative decisions collected
        self._total_updates: Dict[str, int] = {}      # cumulative PPO updates per bot
        self._total_errors = 0
        self._last_update_metrics: Dict[str, Dict[str, float]] = {}

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
        from net.model import PokeNet
        from net.vocab_db import build_vocabs

        vocabs = build_vocabs()
        by_path: Dict[str, Any] = {}
        for name in self.registry.names():
            ckpt = self.registry.get(name).ckpt_path
            if ckpt not in by_path:
                self.log(f"[league] loading checkpoint {name!r} <- {ckpt} "
                         f"(device={self.cfg.device})")
                # THE ONE load path (D103 step 1): both checkpoint formats are
                # handled inside PokeNet.load_file (no format sniffing here).
                net = PokeNet.load_file(ckpt, vocabs, device=self.cfg.device)
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
        the registry.

        Learning mode (D103 step 2): each side is a LIVE ``LearningBot``'s
        per-battle agent, and every real decision is captured into that bot's
        per-battle :class:`TrajectoryCollector` (finalized in ``_on_battle``).
        Eval mode: the legacy checkpoint-cache path, unchanged."""
        if self.cfg.learning:
            return self._learning_driver_factory()
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

    # ── learning mode (D103 step 2) ─────────────────────────────────────
    def _load_bots(self) -> None:
        """Materialize the live bots ONCE (``materialize_roster``: the ONE
        load path + the ONE-FILE-PER-BOT enforcement + loud errors before
        any battle)."""
        if self._bots:
            return
        from net.vocab_db import build_vocabs
        from .learner import materialize_roster

        vocabs = build_vocabs()
        self._bots = materialize_roster(self.registry, vocabs,
                                        device=self.cfg.device)
        self.log(f"[league] learning bots ready: {len(self._bots)} "
                 f"({', '.join(sorted(self._bots))})")

    def _learning_driver_factory(self):
        """The learning-mode factory: LIVE bots, fresh per-battle agents,
        per-battle trajectory collectors (one per side)."""
        self._load_bots()
        from .collect import TrajectoryCollector
        from .driver import play_agent_battle_on

        def factory(idx: int):
            a_name, b_name = self._pairings[idx]
            bot_a, bot_b = self._bots[a_name], self._bots[b_name]
            # ONE collector per (battle, side): owned by this battle's worker
            # thread until _on_battle finalizes/discards it.
            coll_a, coll_b = TrajectoryCollector(), TrajectoryCollector()
            with self._lk:
                self._collectors[idx] = (coll_a, coll_b)
            # a FRESH PokeAgent per battle (bot.agent = the same contract as
            # the eval path: fresh torch.Generator, shared net read-only).
            agent_a = bot_a.agent(seed=self.cfg.seed + 2 * idx)
            agent_b = bot_b.agent(seed=self.cfg.seed + 2 * idx + 1)

            def driver(server_cfg, fmt="gen9ou", seed=0, vocabs=None,
                       max_steps=200, on_finish=None):
                return play_agent_battle_on(
                    server_cfg, agent_a=agent_a, agent_b=agent_b,
                    fmt=fmt, seed=seed, vocabs=vocabs, max_steps=max_steps,
                    greedy=self.cfg.greedy, log=self.log, on_finish=on_finish,
                    max_decision_errors=self.cfg.max_consecutive_errors,
                    # the collection seam: each side's real decisions land in
                    # its bot's collector (finalize + buffer in _on_battle).
                    capture_a=coll_a.record, capture_b=coll_b.record)
            return driver
        return factory

    def _finalize_experience(self, idx: int, result: Dict[str, Any]) -> None:
        """Attach the decided outcome to the battle's trajectories and extend
        the bots' buffers (the bot's OWN POV: a1 = pairing[0]).

        Reward shape = ``gae()``'s sparse-terminal model: winner +1, loser
        −1, draw 0 — on the LAST decision of the battle (``collect.finalize``).
        A side that never acted contributes nothing (empty collector)."""
        coll_a, coll_b = self._collectors.get(idx, (None, None))
        if coll_a is None and coll_b is None:
            return
        a_name, b_name = self._pairings[idx]
        if result["a1_won"] and not result["a2_won"]:
            out_a, out_b = +1.0, -1.0
        elif result["a2_won"] and not result["a1_won"]:
            out_a, out_b = -1.0, +1.0
        else:
            out_a, out_b = 0.0, 0.0
        n_a = len(coll_a)
        # GAE-at-finalize (D103 step 3): the battle boundary is known HERE, so
        # fill_gae runs per battle (never across the whole buffer at update
        # time). The config's gamma/lam drive it.
        coll_a = coll_a.finalize(out_a, gamma=self.cfg.gamma, lam=self.cfg.lam)
        self._bots[a_name].buffer.extend(coll_a.steps)
        n_b = len(coll_b)
        coll_b = coll_b.finalize(out_b, gamma=self.cfg.gamma, lam=self.cfg.lam)
        self._bots[b_name].buffer.extend(coll_b.steps)
        with self._lk:
            # per-battle collection count (accurate even if the <2 buffer is
            # retained across rounds, since this is what THIS battle captured)
            self._total_steps[a_name] = self._total_steps.get(a_name, 0) + n_a
            self._total_steps[b_name] = self._total_steps.get(b_name, 0) + n_b
            self._collectors.pop(idx, None)
        self.log(f"[league] battle {idx}: buffered {n_a} step(s) for "
                 f"{a_name} ({out_a:+.0f}) + {n_b} step(s) for {b_name} "
                 f"({out_b:+.0f})")

    def _on_battle(self, idx: int, result: Dict[str, Any]) -> None:
        """The engine-level result hook (SelfPlayRun ``on_battle``).

        A pairing whose battle ERRORED (infra/pool) gets NO ELO update — it
        is already counted in the run summary; silently booking it as a draw
        would corrupt the standings. In learning mode an errored battle's
        PARTIAL trajectories are DISCARDED too — no decided outcome means
        no terminal reward, and a half-trajectory is poison for PPO.
        """
        if result.get("error") is not None:
            self.log(f"[league] battle {idx}: ERRORED - no ELO update "
                     f"({result['error']})")
            with self._lk:
                coll_a, coll_b = self._collectors.pop(idx, (None, None))
            if coll_a is not None or coll_b is not None:
                self.log(f"[league] battle {idx}: DISCARDED partial "
                         f"trajectories ({len(coll_a)}/{len(coll_b)} decisions) "
                         f"- an errored battle has no decided outcome")
            return
        a_name, b_name = self._pairings[idx]
        self._record(idx, a_name, b_name, result["a1_won"], result["a2_won"],
                     result["steps"])
        self._finalize_experience(idx, result)

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

    # ── lifecycle (multi-round over ONE shared pool; one run engine) ─────
    def start(self) -> None:
        if self._started:
            raise RuntimeError("LeagueRun.start: already started")
        self._started = True
        self.plan()
        self._n_rounds = max(1, int(self.cfg.n_rounds))
        self._round_idx = 0
        self._ensure_pool()
        self._begin_round()
        self.state = "running"

    def _ensure_pool(self) -> None:
        """ONE shared pool for the WHOLE run — every round reuses it, so the
        servers boot ONCE (not once per round). An injected pool (tests) is used
        as-is; otherwise a real ``ResourcePool`` is created here and torn down
        exactly once in :meth:`shutdown`."""
        if self._pool is not None:
            return
        from pool import PoolConfig, ResourcePool
        self._pool = ResourcePool(PoolConfig(
            n_servers=self.cfg.n_servers, per_server=self.cfg.per_server,
            port_base=self.cfg.port_base, boot_timeout=self.cfg.boot_timeout),
            log=self.log)

    def _begin_round(self) -> None:
        """Start the NEXT round's SelfPlayRun over the shared pool (a fresh
        driver factory + fresh per-battle agents/collectors each round)."""
        from control.selfplay import SelfPlayConfig, SelfPlayRun
        sp_cfg = SelfPlayConfig(
            n_battles=len(self._pairings),
            battle_format=self.cfg.battle_format,
            seed=self.cfg.seed + 100000 * self._round_idx,   # distinct per round
            max_steps=self.cfg.max_steps,
            n_servers=self.cfg.n_servers, per_server=self.cfg.per_server,
            port_base=self.cfg.port_base, boot_timeout=self.cfg.boot_timeout,
            n_workers=self.cfg.n_workers, save_replays=self.cfg.save_replays,
            library_root=self.cfg.library_root,
            max_consecutive_errors=self.cfg.max_consecutive_errors)
        self._round_idx += 1
        self._sp = SelfPlayRun(sp_cfg, run_dir=self.run_dir,
                               battle_driver_factory=self._driver_factory_fn(),
                               on_battle=self._on_battle,
                               pool=self._pool, library=self._library, log=self.log)
        self._sp.start()
        self.log(f"[league] round {self._round_idx}/{self._n_rounds} started "
                 f"({len(self._pairings)} pairing(s))")

    def wait(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Block until EVERY round has played + updated (or a stop/failed).

        One round = ``SelfPlayRun.wait`` (its battles) + :meth:`_after_round`
        (the step-3 PPO update). The loop advances rounds until ``n_rounds``
        is reached, a stop is requested, or a round fails (a broken pool must
        not burn the remaining rounds)."""
        if self._sp is None:
            raise RuntimeError("LeagueRun.wait: not started")
        while True:
            s = self._sp.wait(timeout)
            self._total_errors += int(s.get("errors", 0))
            self._after_round(s)
            done = (self._stop_requested
                    or s.get("state") in ("failed", "stopped")
                    or self._round_idx >= self._n_rounds)
            if done:
                break
            self._begin_round()
        self.state = self._sp.state
        return self._final_summary()

    def _after_round(self, s: Dict[str, Any]) -> None:
        """End of a round: the D103 step-3 PPO update over each bot's rollout."""
        if self.cfg.learning and self._bots:
            self._update_round()

    def request_stop(self, reason: str = "user") -> None:
        if self._sp is None:
            raise RuntimeError("LeagueRun.request_stop: not started")
        self._stop_requested = True
        self._sp.request_stop(reason)

    def shutdown(self) -> None:
        """Tear down the shared pool + the Node calc workers ONCE (idempotent).

        Called once at the very end (``league_main``'s ``finally``) — NOT per
        round: the pool is shared across rounds and must stay up until the last
        round finishes (``SelfPlayRun.shutdown`` is never called per round for
        exactly this reason)."""
        # best-effort: stop an in-flight round (if any) before killing servers
        if self._sp is not None and getattr(self._sp, "state", None) == "running":
            try:
                self._sp.request_stop(reason="shutdown")
            except Exception as e:  # noqa: BLE001 — teardown must never raise
                self.log(f"[league] WARNING: stop in-flight round failed: {e}")
        if self._pool is not None:
            try:
                self._pool.shutdown()
            except Exception as e:  # noqa: BLE001
                self.log(f"[league] pool shutdown error: {e}")
        # reap the long-lived Node calc workers (idempotent; atexit is the
        # backstop). Covers a mid-run exception since this runs in the caller's
        # finally (mirrors SelfPlayRun.shutdown's worker teardown).
        try:
            from damage import worker as _worker
            _worker.shutdown_pool()
        except Exception as e:  # noqa: BLE001
            self.log(f"[league] calc worker pool shutdown error: {e}")

    # ── the D103 step-3 update (per bot, after each round) ─────────────
    def _update_round(self) -> None:
        """Run ONE on-policy PPO update per learning bot over its GAE'd rollout,
        save the updated weights to the bot's OWN path, and drain the buffer
        (fresh rollout each round — the on-policy discipline).

        Reuses ``net.ppo`` EXACTLY as ``SelfPlayTrainer.update`` does (no PPO
        math re-derived): ``steps_to_batch`` -> on-policy ``old_logprob``
        refresh -> ``PPOTrainer.update`` (1–2 epochs) -> ``bot.save`` -> clear.

        A bot with <2 buffered steps is SKIPPED (not enough for a meaningful
        PPO pass) and keeps its buffer to accumulate more next round — the same
        ``<2 -> skip`` contract as ``SelfPlayTrainer.train_round`` (which
        raises; the league must NOT raise on a bot's thin round)."""
        import numpy as np
        import torch
        from net.ppo import PPOTrainer, steps_to_batch, chosen_logprobs

        for name, bot in self._bots.items():
            steps = bot.buffer.steps
            n = len(steps)
            if n == 0:
                continue
            if n < 2:
                self.log(f"[league] {name}: {n} step(s) — too few for a PPO "
                         f"pass (skipped update; buffer kept for next round)")
                continue
            batch = steps_to_batch(steps, bot.device)
            # on-policy baseline refresh (mirror SelfPlayTrainer.update): the
            # ratio baseline is the CURRENT net's logprobs, recomputed right
            # before the passes (the stored old_logprob was from collection).
            with torch.no_grad():
                batch["old_logprob"] = chosen_logprobs(bot.net, batch).clone()
            trainer = PPOTrainer(bot.net, lr=self.cfg.lr, clip_eps=self.cfg.clip_eps,
                                 vf_coef=self.cfg.vf_coef, ent_coef=self.cfg.ent_coef,
                                 max_norm=self.cfg.max_norm, epochs=self.cfg.epochs,
                                 device=bot.device)
            adv = np.array([st.advantage for st in steps], dtype=np.float32)
            ret = np.array([st.ret for st in steps], dtype=np.float32)
            metrics = trainer.update(batch, advantages=adv, returns=ret)
            bot.n_updates += 1
            bot.save()                     # its OWN path (pristine-snapshot rule)
            bot.clear_buffer()
            self._total_updates[name] = self._total_updates.get(name, 0) + 1
            self._last_update_metrics[name] = metrics
            self.log(f"[league] {name}: PPO update #{bot.n_updates} over {n} steps "
                     f"(loss={metrics.get('loss', float('nan')):.4f}) -> saved "
                     f"{Path(bot.ckpt_path).name}")

    # ── the final summary (cumulative across rounds) ────────────────────
    def _final_summary(self) -> Dict[str, Any]:
        s: Dict[str, Any] = {
            "state": self.state,
            "rounds": self._round_idx,
            "rounds_planned": self._n_rounds,
            "pairings_planned": len(self._pairings),
            "pairings_played": len(self._results),
            "pairings_errored": self._total_errors,
            "errors": self._total_errors,
            "standings": self.registry.standings(),
        }
        if self.cfg.learning:
            s["steps_collected_by_bot"] = dict(self._total_steps)
            s["updates_by_bot"] = {n: self._total_updates.get(n, 0)
                                   for n in self._bots}
            s["n_saved_by_bot"] = {n: self._bots[n].n_saved for n in self._bots}
            s["last_update_metrics_by_bot"] = dict(self._last_update_metrics)
        return s

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
        s = {
            "state": self.state,
            "rounds": self._round_idx,
            "rounds_planned": self._n_rounds,
            "pairings_planned": len(self._pairings),
            "pairings_played": len(rs),
            "wins_by_bot": {b: sum(1 for r in rs if r["winner"] == b)
                            for b in self.registry.names()},
            "standings": self.registry.standings(),
        }
        if self._bots:
            s["steps_collected_by_bot"] = {
                name: len(b.buffer) for name, b in sorted(self._bots.items())}
        return s
