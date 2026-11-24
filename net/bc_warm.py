"""(c) BC warm start — the TRAINER (Arm B, 2026-11-24 DECISION).

Trains a fresh :class:`net.model.PokeNet` to imitate the strategy
teacher's live decisions (the ``tools.bc_collect`` dataset: npz of
registry-LAYOUT fields + action_index + meta.json), and produces the
Arm B start point. The Arm B RUN is then: this BC init -> PPO with an
annealed KL pull to the BC net (:class:`net.ppo.PPOTrainer`
``ref_net``/``ref_kl_beta`` — the caller anneals with
``set_ref_kl_beta``).

Reuses (no code written twice):
  * the loss: :func:`net.model.bc_loss` (masked cross-entropy — the SAME
    math the S4.2 overfit guard and the PPO policy head use)
  * the batch shape: T=1 rows with the exact :func:`net.ppo.steps_to_batch`
    contract (LAYOUT fields (B,1,*shape) + battle_mask (B,1) +
    action_index (B,)) — minus old_logprob/old_value, which the BC loss
    does not need
  * the checkpoint format: ``PokeNet.state_for_save`` (the "BC format" —
    :meth:`PokeNet.load_file` loads it with vocab-size +
    layout-checksum verification, so a mismatch is loud)

The teacher's action is LEGAL BY CONSTRUCTION (it came from
``build_action_list`` in a live battle), which is exactly the premise
``bc_loss`` documents. Rows with ``action_index < 0`` (a teacher
anomaly — the collection never re-maps) are rejected LOUDLY at load,
never silently dropped.

CPU by default (DECISION: the agent is the LLM on the GPU). Nothing here
starts servers — collection is ``tools.bc_collect`` (only when servers
are free), this module trains from the saved npz. Arm B itself runs only
AFTER the (a) learnability gate passes (DECISION order).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from net.model import PokeNet, bc_loss
from net.registry import LAYOUT
from tools.stdio_guard import safe_stdio  # dependency-free (see tools/__init__.py)


# ── dataset I/O ───────────────────────────────────────────────────────────────
def load_bc_dataset(path_dir: str) -> Dict[str, Any]:
    """Load a ``tools.bc_collect`` dataset directory (data.npz + meta.json).

    Returns ``{"fields": {name: (N, *shape) np.ndarray},
    "action_index": (N,) int64, "battle_id": (N,) int32, "meta": dict}``.

    Raises LOUDLY on missing files, a missing LAYOUT field, an empty
    dataset, or any row with ``action_index < 0`` (a teacher anomaly —
    the collection records them but never re-maps; a dataset containing
    them must be investigated, not silently trained on).
    """
    npz_path = os.path.join(path_dir, "data.npz")
    meta_path = os.path.join(path_dir, "meta.json")
    if not (os.path.exists(npz_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(
            f"load_bc_dataset: missing {npz_path} and/or {meta_path}")
    data = np.load(npz_path)
    missing = [f.name for f in LAYOUT if f.name not in data.files]
    if missing:
        raise ValueError(f"load_bc_dataset: npz is missing LAYOUT fields: "
                         f"{missing}")
    fields = {f.name: np.ascontiguousarray(data[f.name]) for f in LAYOUT}
    n = fields[LAYOUT[0].name].shape[0]
    if n == 0:
        raise ValueError("load_bc_dataset: empty dataset (0 decisions) — "
                         "check the collection meta.json")
    action_index = np.asarray(data["action_index"], dtype=np.int64)
    bad = int((action_index < 0).sum())
    if bad:
        raise ValueError(f"load_bc_dataset: {bad}/{n} rows have "
                         "action_index < 0 (teacher anomaly) — investigate "
                         "the collection before training")
    with open(meta_path, encoding="utf8") as fh:
        meta = json.load(fh)
    return {"fields": fields, "action_index": action_index,
            "battle_id": np.asarray(data["battle_id"], dtype=np.int32),
            "meta": meta}


def bc_batch(fields: Dict[str, np.ndarray], action_index: np.ndarray,
             start: int, length: int, device: str = "cpu"
             ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    """One T=1 training batch from dataset rows ``[start, start+length)``.

    The SAME shape contract as :func:`net.ppo.steps_to_batch` (LAYOUT
    fields as (B,1,*shape) + battle_mask (B,1)) plus the (B,) target
    action-index tensor. Returns ``(batch, target)``.
    """
    dev = torch.device(device)
    batch: Dict[str, torch.Tensor] = {}
    for fld in LAYOUT:
        arr = np.ascontiguousarray(fields[fld.name][start:start + length])
        batch[fld.name] = torch.from_numpy(arr).to(dev).unsqueeze(1)
    batch["battle_mask"] = torch.ones(length, 1, dtype=torch.float32,
                                      device=dev)
    target = torch.as_tensor(action_index[start:start + length],
                             dtype=torch.int64, device=dev)
    return batch, target


# ── training ──────────────────────────────────────────────────────────────────
def _validate(net: PokeNet, fields: Dict[str, np.ndarray],
              action_index: np.ndarray, start: int, length: int,
              device: str) -> Tuple[float, float]:
    """NLL + argmax-accuracy over a deterministic slice (the val split)."""
    batch, target = bc_batch(fields, action_index, start, length, device)
    net.eval()
    with torch.no_grad():
        out = net(batch)
        nll = float(bc_loss(out["logits"], batch["action_mask"],
                            target, 0).mean().item())
        pred = out["prob"][:, 0, :].argmax(-1)
        acc = float((pred == target).float().mean().item())
    return nll, acc


def train_bc(net: PokeNet, dataset: Dict[str, Any], *,
             epochs: int = 10, batch_size: int = 256, lr: float = 1e-3,
             max_norm: float = 0.5, device: str = "cpu",
             seed: Optional[int] = None, log=None) -> Dict[str, Any]:
    """Adam over shuffled dataset batches; loss = masked cross-entropy
    (``net.model.bc_loss``) on the teacher's action.

    A deterministic VAL split (the last ``min(256, n//10)`` rows, or all
    rows when n < 10) is measured every epoch — train/val NLL + accuracy
    in the returned history, so overfitting vs underfitting is visible
    from the metrics (the "is it imitating?" check).

    Returns ``{"n", "val_size", "history": {"train_nll": [...],
    "val_nll": [...], "val_accuracy": [...]}, "final_val_nll",
    "final_val_accuracy"}``. Deterministic for a given ``seed``.
    """
    lg = log or (lambda m: print(m, flush=True))
    dev = torch.device(device)
    net.to(dev)
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed if seed is not None else 0))
    n = len(dataset["action_index"])
    n_val = min(256, max(1, n // 10)) if n >= 10 else n
    val_start = n - n_val
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hist: Dict[str, List[float]] = {"train_nll": [], "val_nll": [],
                                    "val_accuracy": []}
    for epoch in range(1, epochs + 1):
        perm = torch.randperm(n, generator=g).tolist()
        net.train()
        tot, nb = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            sub_fields = {k: np.take(v, idx, axis=0)
                          for k, v in dataset["fields"].items()}
            sub_ai = dataset["action_index"][idx]
            batch, target = bc_batch(sub_fields, sub_ai, 0, len(idx), device)
            opt.zero_grad()
            out = net(batch)
            loss = bc_loss(out["logits"], batch["action_mask"],
                           target, 0).mean()
            loss.backward()
            if max_norm:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            opt.step()
            tot += float(loss.item())
            nb += 1
        v_nll, v_acc = _validate(net, dataset["fields"],
                                 dataset["action_index"], val_start, n_val,
                                 device)
        t_nll = tot / max(1, nb)
        hist["train_nll"].append(t_nll)
        hist["val_nll"].append(v_nll)
        hist["val_accuracy"].append(v_acc)
        lg(f"[bc] epoch {epoch}/{epochs}: train NLL={t_nll:.4f} "
           f"val NLL={v_nll:.4f} val acc={v_acc:.3f}")
    net.eval()
    return {"n": n, "val_size": n_val, "history": hist,
            "final_val_nll": hist["val_nll"][-1],
            "final_val_accuracy": hist["val_accuracy"][-1]}


def save_bc_checkpoint(net: PokeNet, path: str) -> None:
    """The "BC format" (``PokeNet.state_for_save`` payload) —
    :meth:`PokeNet.load_file` loads it with vocab-size +
    layout-checksum verification. THE Arm B start point."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    torch.save(net.state_for_save(), path)


# ── CLI (CPU) ─────────────────────────────────────────────────────────────────
def main() -> None:
    safe_stdio()  # Windows cp1252 redirect guard: a Unicode print must never kill the run
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU by default (DECISION)
    p = argparse.ArgumentParser(
        description="Train the BC warm-start net (Arm B init) from a "
                    "tools.bc_collect dataset.")
    p.add_argument("--data", required=True,
                   help="dataset dir (data.npz + meta.json)")
    p.add_argument("--out", default=None,
                   help="checkpoint path (default: scratch/bc_init_<ts>.pt)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from net.vocab_db import build_vocabs
    vocabs = build_vocabs()
    net = PokeNet(vocabs)
    dataset = load_bc_dataset(args.data)
    print(f"[bc] dataset: {dataset['n'] if 'n' in dataset else len(dataset['action_index'])} "
          f"decisions, {len(set(dataset['battle_id'].tolist()))} battles "
          f"(meta: {dataset['meta'].get('n_kept')} kept / "
          f"{dataset['meta'].get('n_dropped')} dropped)")
    result = train_bc(net, dataset, epochs=args.epochs,
                      batch_size=args.batch_size, lr=args.lr, seed=args.seed)
    print(f"[bc] final: val NLL={result['final_val_nll']:.4f} "
          f"val acc={result['final_val_accuracy']:.3f}")
    out = args.out or os.path.join(
        "scratch", f"bc_init_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.pt")
    save_bc_checkpoint(net, out)
    print(f"[bc] checkpoint -> {out}")


if __name__ == "__main__":
    main()
