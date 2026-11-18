"""P2 — the self-play agent: a thin sampling wrapper over ``net.model.PokeNet``.

This turns the (random-init, D73-locked) ``PokeNet`` into a policy the self-play
loop and the P2 gate can drive: ``(obs_dict, action_list) -> Decision``.

It deliberately adds **NO model logic**. The masked softmax (illegal = exactly
0), the identity (pointer) policy head, and the value head all live in
``net/model.py``. This module only:

  * reshapes a single pre-encoded observation — the ``net.encoder.encode_context``
    output, i.e. a registry-shaped ``dict[str, np.ndarray]`` of 2-D fields — into
    a ``(B=1, T=1)`` batch + ``battle_mask`` and moves it to the net's device;
  * runs one forward pass under ``eval()``/``no_grad()``;
  * reads the masked ``prob`` and either **samples** a legal action index
    (stochastic, for P3/exploration) or **argmax**es it (deterministic, for
    eval/greedy play);
  * reports the index, the chosen action dict, its log-probability, the value
    head's estimate, and the entropy — everything P3 (PPO/GAE) will need.

Index/action consistency by construction: the pointer head scores actions by
IDENTITY (D70, position-independent), and the ``prob`` we sample from is the
masked softmax over the SAME ``action_list`` the encoder wrote into the
``action_*`` channels. So a sampled index maps straight onto ``action_list`` at
that position — the same list, the same order, no lookup table to keep in sync.

CPU-first (P2 is CPU-only, the GPU is the LLM's). The device is taken from the
net itself, so a GPU net (P4) just works once the net is on the GPU — the only
GPU note is the sampling RNG (see :meth:`PokeAgent.act`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .model import PokeNet
from .registry import MAX_ACTIONS


@dataclass
class Decision:
    """One policy decision (a single action + the numbers P3 needs).

    ``action`` is ``None`` (and ``action_index`` is ``-1``) for a **pass/wait**
    turn — no legal action this moment (e.g. poke-env's ``_wait`` / a no-op
    turn). ``logprob`` is the log-probability of the sampled index under the
    masked distribution (finite, <= 0); ``value`` is the value head's estimate
    of the terminal outcome in (0, 1) (GAE-ready); ``entropy`` is the policy
    entropy over the legal set (illegal positions contribute 0); ``prob`` is the
    full masked distribution over the padded ``MAX_ACTIONS`` axis (illegal = 0).
    """
    action: Optional[Dict]
    action_index: int
    logprob: float
    value: float
    entropy: float
    prob: np.ndarray


class PokeAgent:
    """A thin, seedable policy wrapper over :class:`net.model.PokeNet`.

    The agent owns no weights — it just reads ``net.parameters()`` for the device
    and samples/argmaxes the net's masked ``prob``. ``net`` is left in ``eval()``
    (no dropout/batchnorm state mutation during inference).
    """

    def __init__(self, net: PokeNet, seed: Optional[int] = None,
                 device: Optional[str] = None) -> None:
        self.net = net
        # default to the net's own device (a GPU net in P4 "just works").
        self.device = torch.device(device) if device else next(net.parameters()).device
        self.net.eval()
        # A per-agent CPU generator keeps stochastic sampling reproducible under
        # a seed (the P2 "determinism under a seed" gate) without touching the
        # global RNG. NOTE (P4/GPU): ``Categorical.sample`` requires the generator
        # to live on the tensor's device — on a GPU net, either drop the generator
        # (global RNG) or move this to a CUDA generator. P2 is CPU, so CPU is fine.
        self._gen = torch.Generator(device="cpu")
        if seed is None:
            seed = (torch.initial_seed() >> 0) & 0xFFFFFFFF
        self._gen.manual_seed(seed)

    # ── batch building ─────────────────────────────────────────────────
    def _to_batch(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Registry-shaped 2-D arrays -> ``(B=1, T=1, *field.shape)`` + battle_mask.

        Accepts numpy arrays (the ``encode_context`` contract) or tensors already.
        """
        b: Dict[str, torch.Tensor] = {}
        for name, v in obs.items():
            if isinstance(v, torch.Tensor):
                t = v.to(self.device)
            else:
                t = torch.from_numpy(np.ascontiguousarray(v)).to(self.device)
            b[name] = t.unsqueeze(0).unsqueeze(0)  # (1, 1, *field.shape)
        b["battle_mask"] = torch.ones((1, 1), dtype=torch.float32, device=self.device)
        return b

    # ── raw forward (exposed for the legality/no-NaN unit tests) ──────
    def forward_obs(self, obs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """One ``(B=1, T=1)`` forward pass -> ``{logits, value, prob}`` (masked)."""
        b = self._to_batch(obs)
        with torch.no_grad():
            return self.net(b)

    # ── the policy ─────────────────────────────────────────────────────
    def act(self, obs: Dict[str, Any], action_list: List[Dict],
            deterministic: bool = False) -> Decision:
        """Choose one legal action from ``action_list`` given the observation.

        * ``deterministic=True``  -> argmax over the masked ``prob`` (eval/greedy).
        * ``deterministic=False`` -> sample from the masked ``prob`` (stochastic).

        An empty ``action_list`` is a **pass/wait** turn: returns a pass Decision
        (``action=None``, ``action_index=-1``) and never touches the net.
        """
        acts: List[Dict] = list(action_list or [])
        if not acts:
            return Decision(action=None, action_index=-1, logprob=0.0, value=0.5,
                            entropy=0.0, prob=np.zeros(MAX_ACTIONS, dtype=np.float32))

        out = self.forward_obs(obs)
        prob = torch.clamp(out["prob"][0, 0], min=0.0)          # (A,) illegal = 0
        value = out["value"][0, 0, 0]                            # scalar in (0, 1)

        if deterministic:
            idx = int(torch.argmax(prob).item())
        else:
            # ``torch.multinomial`` (unlike ``Categorical.sample``) accepts a
            # per-agent ``generator`` on the same device, which is what makes the
            # "determinism under a seed" gate hold without touching the global RNG.
            idx = int(torch.multinomial(prob, 1, replacement=False,
                                        generator=self._gen).item())

        idx = min(idx, len(acts) - 1)  # belt-and-suspenders: illegal are prob 0
        p_sel = float(prob[idx].item())
        logprob = float(np.log(p_sel + 1e-12))
        entropy = float(-(prob * torch.log(prob + 1e-12)).sum().item())
        action = acts[idx]
        return Decision(action=action, action_index=idx, logprob=logprob,
                        value=float(value.item()), entropy=entropy,
                        prob=prob.detach().cpu().numpy())

    # ── deterministic alias (eval / greedy play) ───────────────────────
    def act_greedy(self, obs: Dict[str, Any], action_list: List[Dict]) -> Decision:
        """The argmax (eval) decision — the gate/eval path the P2 spec asks for."""
        return self.act(obs, action_list, deterministic=True)


def make_random_net(vocabs, device: str = "cpu") -> PokeNet:
    """A fresh random-init :class:`PokeNet` (the D73-locked default start) on
    ``device``, in eval mode. DRY helper shared by the unit tests and the P2 gate
    so "random-init net" is one code path, not two."""
    net = PokeNet(vocabs)
    net.to(device)
    net.eval()
    return net
