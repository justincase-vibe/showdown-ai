"""P4.1 — the self-play training CORE (the thing the P4 GUI drives).

This is the reusable, config-driven trainer that composes the pieces P1–P3
already built and verified — it adds **no model logic and no new PPO math**:

    P1 env (``net.env.ShowdownEnv``)  ->  drives live self-play battles
    P2 agent (``net.agent.PokeAgent``) ->  the policy that collects decisions
    P3 PPO core (``net.ppo``)           ->  GAE + rollout buffer + the update

``SelfPlayTrainer`` is **device-agnostic** (CPU here, GPU when the user runs it
via the P4 GUI) and **on-policy** by construction: each round collects FRESH
rollouts, GAEs them, and does a small number (1–2) of PPO passes over that fresh
batch — the P4 discipline D76 called for (the P3 smoke's 4-epochs-on-one-stale
batch drove ``frac_clipped≈0.6``, which is wrong for real training).

What this adds over the P3.4 smoke gate:
  * a **``TrainConfig``** dataclass (all PPO hyperparameters + rollout params +
    device + checkpoint dir) with ``to_dict``/``from_dict`` so the GUI can
    serialize/edit it;
  * **on-policy** rounds (fresh rollouts + 1–2 epochs per update);
  * **draw-tolerant** collection (a draw is reward −1, 2026-11-23 ruling,
    still buffered — the P3.4 gate's ``|reward|=1`` terminal assumption was
    a smoke simplification);
  * **checkpoint save/resume** (``state_for_save`` + config + round) so a long
    GPU run is safe and resumable;
  * **GPU-ready sampling** (the P2 agent's sampling generator now lives on the
    net's device — see ``net.agent`` — so the same code path works on CUDA).

CPU-first: this module is verified on CPU (the GPU is the LLM's). It performs NO
training by itself — the unit tests and the live gate prove the loop is sound
(collect -> GAE -> update -> checkpoint -> resume), which is exactly the "build
stable before scaling" bar.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

import numpy as np

from .agent import PokeAgent
from .ppo import (PPOTrainer, RolloutBuffer, RolloutStep, fill_gae,
                 steps_to_batch, chosen_logprobs)
from .env import ShowdownEnv, StrongTeamBuilder, build_action_list, action_to_pe, random_action


# ═══════════════════════════════════════════════════════════════════════════════
#  training configuration  (GUI-serializable)
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class TrainConfig:
    """All knobs a training run needs. Defaults = the D76 PPO defaults + the
    D76 P4 on-policy discipline (``epochs=2`` on fresh rollouts)."""

    # ── rollout ───────────────────────────────────────────────────────
    battle_format: str = "gen9ou"
    battles_per_round: int = 50      # the user: "we start with 50 battles and slowly scale up"
    max_steps: int = 500
    opponent: str = "random"         # "random" for now; P5 league pool replaces this
    seed: int = 20261118
    # ── device (CPU here; the user runs "cuda" via the GUI) ───────────
    device: str = "cpu"
    # ── PPO (exposed to + tuned in the P4 GUI; D76 reversible defaults) ─
    lr: float = 3e-4
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    max_norm: float = 0.5
    gamma: float = 0.99
    lam: float = 0.95
    epochs: int = 2                  # D76: 1–2 epochs on FRESH rollouts (on-policy)
    # ── checkpointing ─────────────────────────────────────────────────
    checkpoint_dir: Optional[str] = None
    save_every: int = 1              # save a checkpoint every N rounds
    # ── P4.3 (GUI) ────────────────────────────────────────────────────
    n_rounds: int = 10               # how many rounds a GUI-launched run executes
    resume_from: Optional[str] = None  # checkpoint path to resume from (PokeNet.load)
    battle_timeout: int = 240          # per-battle watchdog (s); 0 = disabled. A battle
                                       # stuck inside env.step()/reset() (a network
                                       # wait that never resolves) is killed after this
                                       # and SKIPPED, so one hung battle can never stall
                                       # the whole run (the 2026-11-22 hang, a transient
                                       # server contention, is the motivating case).

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainConfig":
        """Build from a dict (GUI JSON); unknown keys are ignored so an old
        serialized config can't break a newer dataclass and vice-versa."""
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


# ═══════════════════════════════════════════════════════════════════════════════
#  battle collection  (the P3.4 gate loop, promoted to a reusable function)
# ═══════════════════════════════════════════════════════════════════════════════
def play_and_collect(agent: PokeAgent, fmt: str, seed: int,
                     max_steps: int = 500,
                     server_configuration: Optional[Any] = None,
                     log_level: Optional[int] = None) -> Dict[str, Any]:
    """Play ONE live battle (``agent`` vs random) and return the agent's
    trajectory + battle stats.

    Draw-tolerant: a decided win is reward +1, a loss −1, a **draw −1**
    (2026-11-23 ruling — draw is punished like a loss to discourage stalling).
    All three are valid terminal outcomes.
    Every agent decision is legality-checked at collection time (the P2
    guarantee, re-checked live). The returned dict carries ``traj`` (a1's
    :class:`RolloutStep`s), ``n_illegal``, ``a1_won``/``a2_won``, ``crashed``,
    ``error``, and ``steps``.
    """
    import logging
    from poke_env.ps_client import LocalhostServerConfiguration
    if server_configuration is None:
        server_configuration = LocalhostServerConfiguration
    if log_level is None:
        log_level = logging.WARNING

    team = StrongTeamBuilder(fmt, seed=seed)
    env = ShowdownEnv(battle_format=fmt, team=team,
                      server_configuration=server_configuration, log_level=log_level)
    a1, a2 = env.possible_agents
    out: Dict[str, Any] = {"traj": [], "crashed": False, "error": "", "n_illegal": 0,
                           "a1_won": None, "a2_won": None, "steps": 0, "draw": False}
    pending = None
    try:
        obs, _info = env.reset()
        for _ in range(max_steps):
            if env.battle1 is not None and env.battle1.finished:
                break
            actions = {a1: np.int64(-2), a2: np.int64(-2)}
            if env.agent1_to_move:
                # defensive: if a decision is still pending (shouldn't happen — a
                # step always resolves before the agent is asked again), flush it
                # as reward 0 rather than silently dropping it.
                if pending is not None:
                    out["traj"].append(RolloutStep(obs=pending[0], action_index=pending[1],
                                                    old_logprob=pending[2], old_value=pending[3],
                                                    reward=0.0, done=False))
                    pending = None
                obs_dict = obs[a1]["observation"]
                acts = build_action_list(env.battle1)
                d = agent.act(obs_dict, acts)
                if d.action is None:
                    actions[a1] = np.int64(-2)  # pass/wait turn
                else:
                    if not (0 <= d.action_index < len(acts)) or d.action is not acts[d.action_index]:
                        out["n_illegal"] += 1
                    pe = action_to_pe(d.action, env.battle1)
                    if pe < 0:
                        out["n_illegal"] += 1
                    actions[a1] = np.int64(pe)
                    pending = (obs_dict, d.action_index, float(d.logprob), float(d.value))
            if env.agent2_to_move:
                act2 = random_action(env.battle2)
                actions[a2] = np.int64(action_to_pe(act2, env.battle2) if act2 is not None else -2)

            obs, rew, term, trunc, _info = env.step(actions)
            out["steps"] += 1
            if pending is not None:
                r = float(rew[a1])
                done = bool(term[a1] or trunc[a1])
                out["traj"].append(RolloutStep(obs=pending[0], action_index=pending[1],
                                                old_logprob=pending[2], old_value=pending[3],
                                                reward=r, done=done))
                pending = None
            if term[a1] or term[a2] or trunc[a1] or trunc[a2]:
                break
        out["a1_won"] = bool(env.battle1.won)
        out["a2_won"] = bool(env.battle2.won)
        out["draw"] = (not out["a1_won"]) and (not out["a2_won"])
    except Exception as exc:  # noqa: BLE001
        out["crashed"] = True
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  the trainer
# ═══════════════════════════════════════════════════════════════════════════════
class SelfPlayTrainer:
    """Config-driven self-play PPO trainer (P4.1 core; the P4 GUI drives this).

    One ``train_round()`` = collect ``cfg.battles_per_round`` FRESH battles,
    GAE each, do ``cfg.epochs`` (1–2, on-policy) PPO passes over the fresh batch,
    and checkpoint. ``train(n_rounds)`` loops it. All CPU-verifiable; GPU only
    changes ``cfg.device``.
    """

    def __init__(self, net: Any, cfg: Optional[TrainConfig] = None,
                 agent: Optional[PokeAgent] = None) -> None:
        self.cfg = cfg or TrainConfig()
        self.net = net
        self.agent = agent or PokeAgent(net, seed=self.cfg.seed, device=self.cfg.device)
        self.ppo = PPOTrainer(net, lr=self.cfg.lr, clip_eps=self.cfg.clip_eps,
                              vf_coef=self.cfg.vf_coef, ent_coef=self.cfg.ent_coef,
                              max_norm=self.cfg.max_norm, epochs=self.cfg.epochs,
                              device=self.cfg.device)
        self.buffer = RolloutBuffer()
        self.round = 0
        self._last: Dict[str, float] = {}

    # ── GAE (the 2V-1 value mapping lives here — see net/ppo.py module doc) ─
    def apply_gae(self, traj: List[RolloutStep]) -> None:
        """Fill ``advantage``/``ret`` on each step from the sparse terminal reward.

        Delegates to the shared ``net.ppo.fill_gae`` (D103 step 3 — ONE GAE
        implementation for the solo trainer AND the league). The value head is
        a Sigmoid = P(win) in (0,1); the 2V-1 reward-space map lives inside
        ``fill_gae`` (see net/ppo.py)."""
        fill_gae(traj, gamma=self.cfg.gamma, lam=self.cfg.lam)

    # ── rollout ─────────────────────────────────────────────────────────
    def rollout(self, n_battles: Optional[int] = None,
                battle_hook: Optional[Any] = None,
                stop_check: Optional[Any] = None) -> Dict[str, Any]:
        """Collect ``n_battles`` (default ``cfg.battles_per_round``) FRESH battles,
        GAE each, and extend the buffer. Returns a rollout summary (incl. any
        per-battle crashes/draws).

        ``battle_hook`` (P4.3): optional ``Callable[[int], None]`` called with
        the 0-based battle index BEFORE each battle — the GUI's per-battle
        heartbeat hook (liveness for long rounds).

        ``stop_check`` (P4.3, D80): optional ``Callable[[], bool]`` checked
        BEFORE every battle — the GUI's graceful-stop signal. The in-flight
        battle ALWAYS finishes (the check is between battles, never inside
        one); when it fires, collection stops early and
        ``summary["stopped_early"]`` is True. ``train_round`` still completes
        the round atomically (update + checkpoint on what was collected), so
        a stop never abandons a half-finished battle's state mid-update."""
        n = self.cfg.battles_per_round if n_battles is None else int(n_battles)
        summary: Dict[str, Any] = {"battles": 0, "transitions": 0, "crashed": 0,
                                  "draws": 0, "a1_wins": 0, "a1_losses": 0,
                                  "n_illegal": 0, "errors": [], "n_timeouts": 0,
                                  "stopped_early": False}
        for i in range(n):
            if stop_check is not None and stop_check():
                summary["stopped_early"] = True
                break
            if battle_hook is not None:
                battle_hook(i)
            seed = self.cfg.seed + self.round * 1000 + i
            r = self._play_with_watchdog(seed)
            if r.get("timed_out"):
                summary["crashed"] += 1
                summary["n_timeouts"] += 1
                summary["errors"].append(r["error"])
                continue
            if r["crashed"]:
                summary["crashed"] += 1
                summary["errors"].append(r["error"])
                continue
            self.apply_gae(r["traj"])
            if r["traj"]:
                self.buffer.extend(r["traj"])
            summary["battles"] += 1
            summary["transitions"] += len(r["traj"])
            summary["draws"] += 1 if r.get("draw") else 0
            summary["a1_wins"] += 1 if r["a1_won"] else 0
            summary["a1_losses"] += 1 if (r["a2_won"] and not r["a1_won"]) else 0
            summary["n_illegal"] += r["n_illegal"]
        return summary

    # ── per-battle watchdog (robustness: one hung battle must never stall a run) ──
    def _play_with_watchdog(self, seed: int) -> Dict[str, Any]:
        """Run ONE battle under a watchdog (``cfg.battle_timeout`` seconds).

        ``play_and_collect`` is a synchronous block that can HANG inside
        ``env.step()``/``reset()`` (a network wait that never resolves — the
        ``max_steps`` cap can't help because the stall is INSIDE a single step
        call). The 2026-11-22 hang (a transient server contention) stalled a
        160-battle run for 14+ min until it was killed by hand. This watchdog
        runs the battle on a daemon thread and, if it exceeds the timeout,
        reports it as a crashed (SKIPPED) battle so the round still completes.

        ``battle_timeout=0`` disables the watchdog (direct call, the old path).
        Trade-off (documented): a timed-out battle leaks its daemon thread (a
        stuck thread can't be killed in-process); that thread's server
        connection is orphaned and the server times it out. For a bounded run
        that's a few leaked threads at most — acceptable, and far better than a
        run that hangs forever. (A fully clean kill would need per-battle
        process isolation, which is a larger change.)
        """
        timeout_s = int(getattr(self.cfg, "battle_timeout", 0) or 0)
        if timeout_s <= 0:
            return play_and_collect(self.agent, self.cfg.battle_format, seed,
                                    max_steps=self.cfg.max_steps)
        import threading
        result: Dict[str, Any] = {}

        def run() -> None:
            try:
                result.update(play_and_collect(self.agent, self.cfg.battle_format,
                                               seed, max_steps=self.cfg.max_steps))
            except Exception as exc:  # noqa: BLE001
                result.update({"crashed": True,
                               "error": f"{type(exc).__name__}: {exc}"})

        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout_s)
        if t.is_alive():
            return {"crashed": True, "timed_out": True, "traj": [], "steps": 0,
                    "error": (f"battle TIMEOUT after {timeout_s}s (killed) — "
                             "a transient server/network stall")}
        return result

    # ── update ──────────────────────────────────────────────────────────
    def update(self) -> Dict[str, float]:
        """One on-policy PPO update over the buffer (FRESH rollouts; the
        on-policy baseline is refreshed to the current net right before the
        passes). Clears the buffer afterwards."""
        if len(self.buffer) < 2:
            raise RuntimeError("SelfPlayTrainer.update: need >=2 transitions to update "
                               "(collect more battles per round)")
        batch = steps_to_batch(self.buffer.steps, self.cfg.device)
        import torch
        with torch.no_grad():
            batch["old_logprob"] = chosen_logprobs(self.net, batch).clone()
        advantages = np.array([s.advantage for s in self.buffer.steps], dtype=np.float32)
        returns = np.array([s.ret for s in self.buffer.steps], dtype=np.float32)
        metrics = self.ppo.update(batch, advantages=advantages, returns=returns)
        self.buffer.clear()
        return metrics

    # ── a full round ────────────────────────────────────────────────────
    def train_round(self, battle_hook: Optional[Any] = None,
                    stop_check: Optional[Any] = None) -> Dict[str, Any]:
        """One round: FRESH rollout -> GAE -> on-policy PPO update -> checkpoint.
        Returns ``{**rollout_summary, **update_metrics, "round", "transitions"}``
        (``rollout_summary`` includes ``stopped_early``).

        ``battle_hook`` is forwarded to :meth:`rollout` (per-battle liveness);
        ``stop_check`` is the graceful-stop signal (D80) — see :meth:`rollout`."""
        roll = self.rollout(battle_hook=battle_hook, stop_check=stop_check)
        m = self.update() if len(self.buffer) >= 2 else {"skipped": True, "reason": "too_few_transitions"}
        self.round += 1
        out: Dict[str, Any] = {"round": self.round, **roll, **m}
        if (self.cfg.checkpoint_dir is not None
                and self.cfg.save_every >= 1
                and self.round % self.cfg.save_every == 0):
            out["checkpoint"] = self.save_checkpoint()
        self._last = out
        return out

    def train(self, n_rounds: int) -> List[Dict[str, Any]]:
        """Run ``n_rounds`` of :meth:`train_round`; return each round's metrics."""
        return [self.train_round() for _ in range(int(n_rounds))]

    # ── checkpoint save / resume ────────────────────────────────────────
    def checkpoint_path(self) -> str:
        import os
        base = self.cfg.checkpoint_dir or "checkpoints"
        return os.path.join(base, f"selfplay_round{self.round}.pt")

    def save_checkpoint(self) -> str:
        import os
        import torch
        os.makedirs(self.cfg.checkpoint_dir or "checkpoints", exist_ok=True)
        path = self.checkpoint_path()
        payload = {"net": self.net.state_for_save(),
                   "config": self.cfg.to_dict(),
                   "round": self.round,
                   "last": self._last}
        torch.save(payload, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Resume into THIS net object (same process). In-place ``load_state_dict``
        is shape-safe because the net was built with the same vocabs; the
        vocab/layout verification lives in ``PokeNet.load`` (fresh-process resume,
        e.g. the GUI relaunching a run)."""
        import torch
        payload = torch.load(path, map_location="cpu")
        self.net.load_state_dict(payload["net"]["state_dict"])
        self.net.eval()
        if "config" in payload:
            self.cfg = TrainConfig.from_dict(payload["config"])
        self.round = int(payload.get("round", 0))
        self._last = payload.get("last", {})
        self.buffer.clear()
