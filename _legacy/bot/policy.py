"""Compact, correct masked PPO for a discrete action space with per-step action masks.

Why custom (instead of Stable-Baselines3 as-is): SB3 has no first-class action masking,
and our single shared policy spans gens 6-9 in a unified 963-action *move-identity* space
(move ids + 6 switches + 3 gimmick transforms). Masking must be applied both at *rollout*
(sampling) and at *training* (recomputing log-probs from stored masks). The legal set per
situation is small (~4 moves + a few switches + <=1 transform) though the space is 963.
This ~150-line PPO does exactly that and is device-agnostic (CPU/CUDA).

Reward is the paper's scheme: +1 win / -1 loss, delivered as the terminal transition of
an episode. GAE is computed per completed episode (terminal bootstrap = 0).
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def _mlp(in_dim: int, out_dim: int, hidden: Tuple[int, ...], act: type = nn.ReLU) -> nn.Sequential:
    layers: List[nn.Module] = []
    d = in_dim
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def _blank_calib() -> Dict[str, float]:
    """Zero-initialized critic-calibration accumulators (see ``_calib_stats``)."""
    return {
        "n": 0, "s_err": 0.0, "s_abs": 0.0, "s_sq": 0.0,
        "s_v0": 0.0, "s_R": 0.0, "s_v0R": 0.0, "s_v0sq": 0.0, "s_Rsq": 0.0,
        "s_abs_v0": 0.0, "sign_hit": 0, "sign_n": 0,
    }


@dataclass
class _Episode:
    obs: List[np.ndarray] = field(default_factory=list)
    act: List[int] = field(default_factory=list)
    lp: List[float] = field(default_factory=list)
    rew: List[float] = field(default_factory=list)
    mask: List[np.ndarray] = field(default_factory=list)
    val: List[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.act)


class MaskedPPO:
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        hidden: Tuple[int, ...] = (128, 128),
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        epochs: int = 4,
        minibatch: int = 64,
        max_grad_norm: float = 0.5,
        device: Optional[str] = None,
        seed: int = 0,
    ):
        torch.manual_seed(seed)
        np.random.seed(seed)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.lam = lam
        self.clip = clip
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.epochs = epochs
        self.minibatch = minibatch
        self.max_grad_norm = max_grad_norm

        self.actor = _mlp(obs_dim, n_actions, hidden).to(self.device)
        self.critic = _mlp(obs_dim, 1, hidden).to(self.device)
        params = list(self.actor.parameters()) + list(self.critic.parameters())
        self.opt = torch.optim.Adam(params, lr=lr)

        self._episodes: Dict[int, _Episode] = {}
        self._buffer: Optional[dict] = None
        self.n_updates = 0
        # Critic-calibration accumulators: populated at episode flush, reported +
        # reset on each update(). "val0" = critic's opening read of the game, "R" =
        # the realized terminal reward (+1/-1). This is the *built-in* progress
        # signal (the 50/50 win-rate is useless by design in both-sides self-play).
        self._calib = _blank_calib()

    # ------------------------------------------------------------------ act
    @torch.no_grad()
    def act(self, obs: np.ndarray, mask: np.ndarray) -> Tuple[int, float, float]:
        self.actor.eval()
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(o)[0]
        m = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        if not bool(m.any()):
            raise ValueError(
                "policy.act() called with an empty action mask: this is a Showdown "
                "'wait' request (nothing to decide this turn). The loop must pass "
                "(TraineeEnv.step sends the default -2 order) and record the step "
                "with store_pass() instead of consulting the policy."
            )
        logits = logits.masked_fill(~m, float("-inf"))
        dist = Categorical(logits=logits)
        a = int(dist.sample().item())
        lp = float(dist.log_prob(torch.tensor(a, device=self.device)).item())
        v = float(self.critic(o).item())
        return a, lp, v

    @torch.no_grad()
    def value(self, obs: np.ndarray) -> float:
        """Critic estimate for ``obs`` (used to record no-decision 'wait' steps)."""
        self.critic.eval()
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return float(self.critic(o).item())

    def store_pass(
        self,
        env_id: int,
        obs: np.ndarray,
        reward: float,
        value: float,
        done: bool,
    ) -> None:
        """Record a no-decision step (Showdown 'wait' request).

        The policy made no decision here, so a placeholder action/logprob keeps the
        parallel episode lists aligned; the all-zero mask makes the row (nearly)
        gradient-inert at update time (``update()`` expands empty masks to all-legal,
        so the stored placeholder's log-prob stays finite). If ``done``, the episode
        is flushed so a terminal reward (+1/-1) landing on a wait step is still
        attributed and no episode leaks.
        """
        ep = self._episodes.get(env_id)
        if ep is None:
            return  # wait state before any decision: nothing in flight for this env
        ep.obs.append(obs)
        ep.act.append(0)
        ep.lp.append(0.0)
        ep.rew.append(reward)
        ep.mask.append(np.zeros(self.n_actions, dtype=np.int8))
        ep.val.append(value)
        if done:
            self._flush_episode(env_id)

    # ------------------------------------------------------------- collect
    def store(
        self,
        env_id: int,
        obs: np.ndarray,
        action: int,
        logprob: float,
        reward: float,
        done: bool,
        mask: np.ndarray,
        value: float,
    ) -> None:
        ep = self._episodes.setdefault(env_id, _Episode())
        ep.obs.append(obs)
        ep.act.append(action)
        ep.lp.append(logprob)
        ep.rew.append(reward)
        ep.mask.append(mask)
        ep.val.append(value)
        if done:
            self._flush_episode(env_id)

    def _flush_episode(self, env_id: int) -> None:
        ep = self._episodes.pop(env_id)
        T = len(ep)
        if T == 0:
            return
        # Per-episode GAE; last step is the terminal (done) step -> bootstrap 0.
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in range(T - 1, -1, -1):
            next_val = 0.0 if t == T - 1 else ep.val[t + 1]
            delta = ep.rew[t] + self.gamma * next_val - ep.val[t]
            gae = delta + self.gamma * self.lam * gae
            adv[t] = gae
        ret = (adv + np.asarray(ep.val, dtype=np.float32)).astype(np.float32)

        batch = {
            "obs": np.stack(ep.obs),
            "act": np.asarray(ep.act, dtype=np.int64),
            "lp": np.asarray(ep.lp, dtype=np.float32),
            "adv": adv,
            "ret": ret,
            "mask": np.stack(ep.mask),
            "val": np.asarray(ep.val, dtype=np.float32),
        }
        # Critic calibration: did the critic's opening read (val at the first
        # decision state) match the realized game outcome (terminal reward)?
        if len(ep.val) >= 1 and len(ep.rew) >= 1:
            self._record_calibration(float(ep.val[0]), float(ep.rew[-1]))
        self._buffer = self._merge(self._buffer, batch)

    @staticmethod
    def _merge(a: Optional[dict], b: dict) -> dict:
        if a is None:
            return b
        return {k: np.concatenate([a[k], b[k]]) for k in a}

    # ----------------------------------------------------- critic calibration
    def _record_calibration(self, val0: float, R: float) -> None:
        """Accumulate one episode's (opening value read vs realized terminal reward)."""
        c = self._calib
        err = val0 - R
        c["n"] += 1
        c["s_err"] += err
        c["s_abs"] += abs(err)
        c["s_sq"] += err * err
        c["s_v0"] += val0
        c["s_R"] += R
        c["s_v0R"] += val0 * R
        c["s_v0sq"] += val0 * val0
        c["s_Rsq"] += R * R
        c["s_abs_v0"] += abs(val0)
        if abs(val0) > 1e-9:  # "did the bot read the game right?" (sign agreement)
            c["sign_n"] += 1
            if (val0 > 0) == (R > 0):
                c["sign_hit"] += 1

    def _calib_stats(self) -> Dict[str, float]:
        """Report the calibration metrics for episodes flushed since the last
        update(), then reset the accumulators.

        * ``calib_rmse`` / ``calib_mae`` / ``calib_bias`` -- how far the critic's
          opening read is from the realized +1/-1 (want -> 0 / 0).
        * ``calib_corr`` -- Pearson corr(opening read, outcome) (want -> 1).
        * ``calib_sign_hit`` -- fraction of games whose *direction* the critic read
          correctly (want -> 1). The most interpretable single number.
        * ``calib_confidence`` -- mean |opening read| (want: rises toward 1 AND be
          right -- confident *and* correct beats timid *and* wrong).
        """
        c = self._calib
        out: Dict[str, float] = {}
        n = c["n"]
        if n > 0:
            mn_v = c["s_v0"] / n
            mn_R = c["s_R"] / n
            cov = c["s_v0R"] / n - mn_v * mn_R
            var_v = c["s_v0sq"] / n - mn_v * mn_v
            var_R = c["s_Rsq"] / n - mn_R * mn_R
            out["calib_rmse"] = math.sqrt(c["s_sq"] / n)
            out["calib_mae"] = c["s_abs"] / n
            out["calib_bias"] = c["s_err"] / n
            if var_v > 1e-12 and var_R > 1e-12:
                out["calib_corr"] = cov / math.sqrt(var_v * var_R)
            out["calib_sign_hit"] = (c["sign_hit"] / c["sign_n"]) if c["sign_n"] else float("nan")
            out["calib_confidence"] = c["s_abs_v0"] / n
            out["calib_n"] = float(n)
        self._calib = _blank_calib()  # reset for the next update
        return out

    # --------------------------------------------------------------- update
    def update(self) -> Optional[dict]:
        if self._buffer is None:
            return None
        buf = self._buffer
        self._buffer = None
        n = len(buf["act"])
        if n == 0:
            return None

        obs = torch.as_tensor(buf["obs"], device=self.device)
        act = torch.as_tensor(buf["act"], device=self.device)
        old_lp = torch.as_tensor(buf["lp"], device=self.device)
        adv = torch.as_tensor(buf["adv"], device=self.device)
        ret = torch.as_tensor(buf["ret"], device=self.device)
        masks = torch.as_tensor(buf["mask"], dtype=torch.bool, device=self.device)
        old_val = torch.as_tensor(buf["val"], device=self.device)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        self.actor.train()
        self.critic.train()
        idx_all = np.arange(n)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0}
        nmb = 0
        for _ in range(self.epochs):
            self.opt.zero_grad()
            np.random.shuffle(idx_all)
            for start in range(0, n, self.minibatch):
                idx = torch.as_tensor(
                    idx_all[start : start + self.minibatch], dtype=torch.long, device=self.device
                )
                o = obs[idx]
                m = masks[idx]
                logits = self.actor(o)
                m = torch.where(m.any(dim=1, keepdim=True), m, torch.ones_like(m))
                logits = logits.masked_fill(~m, float("-inf"))
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(act[idx])
                ratio = torch.exp(new_lp - old_lp[idx])
                a = adv[idx]
                surr = torch.min(ratio * a, torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * a)
                ent = dist.entropy()
                v = self.critic(o).squeeze(-1)
                vloss = nn.functional.mse_loss(v, ret[idx])
                loss = -surr.mean() + self.vf_coef * vloss - self.ent_coef * ent.mean()
                loss.backward()
                stats["policy_loss"] += (-surr.mean()).item()
                stats["value_loss"] += vloss.item()
                stats["entropy"] += ent.mean().item()
                stats["clip_frac"] += ((ratio - 1.0).abs() > self.clip).float().mean().item()
                nmb += 1
            nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                self.max_grad_norm,
            )
            self.opt.step()

        self.n_updates += 1
        out = {k: v / max(1, nmb) for k, v in stats.items()}
        out.update(self._calib_stats())  # per-episode calibration (NOT divided by nmb)
        return out

    # ------------------------------------------------------------ self-play
    def sync_opponent(self, opponent: "MaskedPPO") -> None:
        """Copy the current actor weights into ``opponent`` (the frozen self-play target)."""
        opponent.actor.load_state_dict(self.actor.state_dict())
        opponent.critic.load_state_dict(self.critic.state_dict())

    # ------------------------------------------------------------- io
    def save(self, path: str) -> None:
        """Write the checkpoint atomically: dump to a temp file, then rename over
        ``path``. If the process is killed mid-write (OOM, power loss, a crash) the
        *previous* checkpoint stays intact instead of leaving a truncated, unloadable
        file -- critical for unattended overnight runs that resume from ``path``."""
        tmp = path + ".tmp"
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "opt": self.opt.state_dict(),
                "n_updates": self.n_updates,
            },
            tmp,
        )
        os.replace(tmp, path)  # atomic on Windows / Linux / macOS

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        if "opt" in ck:
            self.opt.load_state_dict(ck["opt"])
        self.n_updates = ck.get("n_updates", 0)
