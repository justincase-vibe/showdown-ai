"""P3 — PPO (clipped surrogate) + GAE + a rollout buffer for self-play.

The P1 env (`net.env`) and P2 agent (`net.agent`) already produce, per decision:
an encoded observation (the registry-shaped `dict[str, np.ndarray]` including the
`action_*` channels), a sampled action index, its log-probability, and the value
head's estimate. This module turns those into a PPO training core:

  * **GAE** (`gae`) — the value-head credit assignment: given the per-step value
    estimates + the (sparse, terminal) rewards of one COMPLETED battle, produce
    the advantages and the return targets the value head regresses to.
  * **rollout buffer** (`RolloutStep` / `RolloutBuffer` / `steps_to_batch`) —
    stores transitions and re-batches them for the net (each transition is an
    independent `(B, T=1)` sample, the standard "flatten the trajectory" PPO).
  * **PPO update** (`PPOTrainer`) — the clipped-surrogate policy loss + value
    (MSE) loss + entropy bonus, one Adam step per epoch, gradient-clipped — the
    SAME optimizer conventions as `net.model.overfit_train`.

This is the TRAINING CORE (P3). It is deliberately CPU-first and batch-agnostic:
the P4 GUI runs the same `PPOTrainer` on a GPU with real rollout lengths. Nothing
here does real training by itself — the "disaster guard" unit test proves the
loop is structurally sound (finite loss, gradients flow, the policy moves in the
right direction), which is exactly the "never learns" check the user is built on.

Reward model (D73 lock): **sparse terminal ±1** (win +1 / loss -1 / 0 otherwise),
zero-sum. So `rewards[t] == 0` for every step except the battle's last, where it
is ±1. `gae` assumes the LAST step of a buffer entry is terminal (a self-play
battle always ends decided — see the P3.4 loop, which only buffers COMPLETED
battles).

Value-head contract (P3 bug-hunt fix): the net's value head is a **Sigmoid**
(see `net/model.py`), so it emits a **win-probability in (0, 1)** (P(win) — the
pass-turn default `value=0.5` is the 50/50 state). The reward space is (−1, +1),
so a value used by GAE / the value loss must be mapped to reward space as
**2·V − 1**. (Feeding the raw (0,1) sigmoid directly — the pre-fix behaviour —
could not represent negative values, so a "likely-loss" state and an "even"
state both collapsed to V≈0, destroying the critic and distorting advantages.)
The mapping is applied at the two consumption points (`PPOTrainer._loss` and
the GAE call sites); `RolloutStep.old_value` stays the raw (0,1) sigmoid the
agent reports, and the generic `gae()` takes reward-space values.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .model import PokeNet, _masked_softmax
from .registry import LAYOUT


# ═══════════════════════════════════════════════════════════════════════════════
#  P3.1 — GAE-λ  (value-head credit assignment)
# ═══════════════════════════════════════════════════════════════════════════════
def gae(rewards: Sequence[float], values: Sequence[float],
        gamma: float = 0.99, lam: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation over one COMPLETED battle.

    Parameters
    ----------
    rewards : (T,) per-step rewards. Sparse terminal model: 0 for t < T-1, ±1 at
        t = T-1 (the battle's decided outcome).
    values  : (T,) the value head's estimate V(s_t) at each decision (the state
        BEFORE action t) — i.e. the ``value`` field of each stored decision.
    gamma   : discount.
    lam     : GAE lambda (1.0 -> plain Monte-Carlo returns).

    Returns ``(advantages (T,), returns (T,))`` (float32) where
    ``returns[t] = advantages[t] + values[t]`` (the value-head regression target).

    The last step is treated as TERMINAL (no bootstrap beyond it) — correct for a
    self-play battle that always ends decided. If the last step were a truncation
    (not a real terminal), bootstrap ``values[-1]`` instead (flagged for P4).
    """
    r = np.asarray(rewards, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if r.shape != v.shape or r.size == 0:
        raise ValueError(f"gae: rewards/values must be same non-empty length "
                         f"(got {r.shape} vs {v.shape})")
    T = v.size
    adv = np.zeros(T, dtype=np.float64)
    g = 0.0
    for t in range(T - 1, -1, -1):
        nonterm = 0.0 if t == T - 1 else 1.0          # no bootstrap past the terminal
        next_v = 0.0 if t == T - 1 else v[t + 1]
        delta = r[t] + gamma * next_v * nonterm - v[t]
        g = delta + gamma * lam * nonterm * g
        adv[t] = g
    returns = adv + v
    return adv.astype(np.float32), returns.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
#  P3.2 — rollout buffer
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class RolloutStep:
    """One stored decision (a PPO transition).

    ``obs`` is the FULL encoded observation (registry-shaped, including the
    ``action_*`` channels built from the legal list at collection time) — storing
    it means the update re-batches EXACTLY what the agent saw (no re-encoding
    drift). ``action_index`` is the sampled position in that legal list;
    ``old_logprob`` / ``old_value`` are the collection-time policy log-prob and
    value estimate (the PPO ratio baseline). ``reward`` / ``done`` drive GAE.
    ``advantage`` / ``ret`` are the per-battle GAE outputs, filled in by the
    collector right after the battle ends (defaults 0.0 — a step not yet GAE'd).
    """
    obs: Dict[str, np.ndarray]
    action_index: int
    old_logprob: float
    old_value: float
    reward: float
    done: bool
    advantage: float = 0.0
    ret: float = 0.0


class RolloutBuffer:
    """A flat store of :class:`RolloutStep`s (one entry per decision).

    P3 keeps it in RAM (a CPU smoke is small). For P4 scale-up this becomes a
    disk-backed / prioritized store — the ``steps``/``slice``/``clear`` contract
    stays the same, so the trainer is unaffected.
    """

    def __init__(self, capacity: Optional[int] = None) -> None:
        self.capacity = capacity
        self._steps: List[RolloutStep] = []

    def append(self, step: RolloutStep) -> None:
        self._steps.append(step)
        if self.capacity is not None and len(self._steps) > self.capacity:
            self._steps = self._steps[-self.capacity:]

    def extend(self, steps: Sequence[RolloutStep]) -> None:
        for s in steps:
            self.append(s)

    @property
    def steps(self) -> List[RolloutStep]:
        return self._steps

    def __len__(self) -> int:
        return len(self._steps)

    def slice(self, start: int, end: int) -> List[RolloutStep]:
        return self._steps[start:end]

    def clear(self) -> None:
        self._steps.clear()


def steps_to_batch(steps: Sequence[RolloutStep],
                   device: str = "cpu") -> Dict[str, object]:
    """Stack a list of transitions into one net batch (each a T=1 sample).

    Returns a dict with the registry fields as ``(B, 1, *field.shape)`` tensors +
    ``battle_mask (B, 1)``, plus the per-row ``action_index (B,)``,
    ``old_logprob (B,)``, ``old_value (B,)`` (float32) and ``action_mask``
    already in the batch. Field order follows the registry (deterministic).
    """
    if not steps:
        raise ValueError("steps_to_batch: empty step list")
    dev = torch.device(device)
    names = [f.name for f in LAYOUT]
    batch: Dict[str, torch.Tensor] = {}
    for name in names:
        stacked = np.stack([np.ascontiguousarray(s.obs[name]) for s in steps])
        batch[name] = torch.from_numpy(stacked).to(dev).unsqueeze(1)  # (B, 1, *shape)
    batch["battle_mask"] = torch.ones(len(steps), 1, dtype=torch.float32, device=dev)
    batch["action_index"] = torch.tensor([s.action_index for s in steps],
                                         dtype=torch.int64, device=dev)
    batch["old_logprob"] = torch.tensor([s.old_logprob for s in steps],
                                        dtype=torch.float32, device=dev)
    batch["old_value"] = torch.tensor([s.old_value for s in steps],
                                      dtype=torch.float32, device=dev)
    return batch


def chosen_logprobs(net: PokeNet, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The CURRENT net's log-probability of each stored action index, ``(B,)``.

    This is what PPO's ratio is measured against at each update (recomputed
    forward), and what the "disaster guard" test tracks to confirm the policy
    actually moves.
    """
    with torch.no_grad():
        out = net(batch)
    logits = out["logits"][:, 0, :]                 # (B, A)
    mask = batch["action_mask"][:, 0, :]            # (B, A)
    prob = _masked_softmax(logits, mask)
    logp = prob.clamp_min(1e-12).log()
    return logp.gather(-1, batch["action_index"][:, None]).squeeze(-1)  # (B,)


# ═══════════════════════════════════════════════════════════════════════════════
#  P3.3 — PPO update
# ═══════════════════════════════════════════════════════════════════════════════
class PPOTrainer:
    """Clipped-surrogate PPO over the :class:`PokeNet` (D73: PPO vs self-target).

    ``update(batch, advantages, returns, epochs)`` runs ``epochs`` full-batch
    Adam steps on  ``-E[min(ratio*A, clip(ratio)*A)] + vf_coef*MSE(V, R)
    - ent_coef*H(pi)`` and returns mean metrics. Full-batch (not minibatched) for
    P3 simplicity — P4 scales to minibatches with the same loss.
    """

    def __init__(self, net: PokeNet, *, lr: float = 3e-4, clip_eps: float = 0.2,
                 vf_coef: float = 0.5, ent_coef: float = 0.01, max_norm: float = 0.5,
                 epochs: int = 4, device: str = "cpu") -> None:
        self.net = net
        self.device = torch.device(device)
        self.net.to(self.device)
        self.lr = lr
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_norm = max_norm
        self.epochs = epochs          # default passes per update (overridable in update())
        self.opt = torch.optim.Adam(net.parameters(), lr=lr)

    # ── one forward loss (full batch) ─────────────────────────────────
    def _loss(self, batch: Dict[str, torch.Tensor], advantages: torch.Tensor,
              returns: torch.Tensor):
        out = self.net(batch)
        logits = out["logits"][:, 0, :]              # (B, A)
        mask = batch["action_mask"][:, 0, :]         # (B, A)
        action_index = batch["action_index"]         # (B,)
        old_logprob = batch["old_logprob"]           # (B,)
        prob = _masked_softmax(logits, mask)
        logp = prob.clamp_min(1e-12).log()
        new_logprob = logp.gather(-1, action_index[:, None]).squeeze(-1)  # (B,)

        ratio = torch.exp(new_logprob - old_logprob)
        s1 = ratio * advantages
        s2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(s1, s2).mean()

        # value head = Sigmoid = P(win) in (0,1); map to reward space (2V-1) so
        # the critic can represent negative returns (a loss) — see module doc.
        value_pred = 2.0 * out["value"][:, 0, 0] - 1.0   # (B,) reward space
        value_loss = F.mse_loss(value_pred, returns)

        entropy = -(prob * logp).sum(-1).mean()
        frac_clipped = (ratio < 1.0 - self.clip_eps).float().mean() \
            + (ratio > 1.0 + self.clip_eps).float().mean()

        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
        return loss, policy_loss, value_loss, entropy, frac_clipped

    def update(self, batch: Dict[str, torch.Tensor], advantages: np.ndarray,
               returns: np.ndarray, epochs: Optional[int] = None) -> Dict[str, float]:
        """Run ``epochs`` full-batch PPO Adam steps (default ``self.epochs``).
        Returns mean metrics."""
        n_epochs = max(1, epochs if epochs is not None else self.epochs)
        dev = self.device
        adv = torch.as_tensor(advantages, dtype=torch.float32, device=dev)
        ret = torch.as_tensor(returns, dtype=torch.float32, device=dev)
        self.net.train()
        tot = {"loss": 0.0, "policy": 0.0, "value": 0.0, "entropy": 0.0,
               "frac_clipped": 0.0, "kl_approx": 0.0}
        for _ in range(n_epochs):
            self.opt.zero_grad()
            loss, pol, val, ent, frac = self._loss(batch, adv, ret)
            loss.backward()
            if self.max_norm:
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_norm)
            self.opt.step()
            with torch.no_grad():
                # cheap KL(old||new) proxy for the metrics log (standard PPO
                # estimator E_old[ratio * log(ratio)] = E[ratio*(logp_new-logp_old)],
                # always >= 0). The pre-fix "exp(d)-1" form was ≈ -KL for small
                # drift and then max(0,.)-clamped to ~0, i.e. a misleading ~0.
                new_lp = chosen_logprobs(self.net, batch)
                ratio_m = torch.exp(new_lp - batch["old_logprob"])
                kl = float((ratio_m * (new_lp - batch["old_logprob"])).mean().item())
            tot["loss"] += float(loss.item()); tot["policy"] += float(pol.item())
            tot["value"] += float(val.item()); tot["entropy"] += float(ent.item())
            tot["frac_clipped"] += float(frac.item()); tot["kl_approx"] += max(0.0, kl)
        self.net.eval()
        return {k: v / n_epochs for k, v in tot.items()}
