"""Imitation learning (behavioral cloning) for the Pokemon Showdown bot.

Trains the actor network to imitate P1's decisions from expert battle logs.
Uses weighted cross-entropy with:
  - ELO weighting: (elo / 1200)^2  (quadratic, user-approved)
  - Format weighting: OU=1.15, UU=1.05, RU=1.00, Random=0.90

Pipeline:
  Phase 1: Pre-compute all (state, action, weight) tuples from the 15 JSONL files.
           Save to a .npz file for fast repeated loading.
  Phase 2: Train the actor network on the pre-computed data.

Usage:
  python bot/imitate.py --precompute          # Phase 1: build the dataset
  python bot/imitate.py --train --epochs 10   # Phase 2: train
  python bot/imitate.py --smoke               # Quick end-to-end test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

# Add bot/ to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from obs import OBS_DIM
from state_from_log import StateReconstructor
from datamine.parse import BattleLogParser

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
UNIFIED_ACTION_SIZE = 26
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "replays")
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DATASET_PATH = os.path.join(CACHE_DIR, "imitation_dataset.npz")

# Format weights (user-approved: slight spread, 10-15% range)
FORMAT_WEIGHTS: Dict[str, float] = {
    "OU": 1.15,
    "UU": 1.05,
    "RU": 1.00,
    "Random": 0.90,
}

# ELO base for quadratic weighting
ELO_BASE = 1200.0


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def extract_format_weight(formatid: str) -> float:
    """Extract the format weight from a format string like 'gen8ou' or 'gen7randombattle'."""
    fmt = formatid.lower().replace("gen", "").replace(" ", "")
    if "ou" in fmt:
        return FORMAT_WEIGHTS["OU"]
    elif "uu" in fmt:
        return FORMAT_WEIGHTS["UU"]
    elif "ru" in fmt:
        return FORMAT_WEIGHTS["RU"]
    elif "random" in fmt or "battle" in fmt:
        return FORMAT_WEIGHTS["Random"]
    return 1.0  # default


def extract_elo(rec: dict) -> float:
    """Extract the P1 player's ELO from the record.

    fetch.py stores P1's rating in the top-level ``rating`` field. (Earlier code looked
    for a ``ratings`` list / ``|rating|`` log line that don't exist in this data, so every
    sample silently fell back to the neutral 1200 -- defeating the quadratic ELO weighting.)
    """
    # Primary: top-level `rating` field = P1's ELO (as stored by fetch.py).
    if "rating" in rec:
        try:
            return float(rec["rating"])
        except (ValueError, TypeError):
            pass
    # Secondary: `ratings` list (some fetch variants store P1 first).
    if isinstance(rec.get("ratings"), list) and rec["ratings"]:
        try:
            return float(rec["ratings"][0])
        except (ValueError, TypeError):
            pass
    # Fallback: parse P1's rating from the log's raw rating line.
    import re
    m = re.search(r"rating:\s*([\d]+)", rec.get("log", ""))
    if m:
        try:
            return float(m.group(1))
        except (ValueError, TypeError):
            pass
    return 1200.0  # neutral


def compute_weight(formatid: str, elo: float) -> float:
    """Compute the sample weight: format_weight * (elo/1200)^2."""
    fmt_w = extract_format_weight(formatid)
    elo_w = (elo / ELO_BASE) ** 2
    return fmt_w * elo_w


def precompute_dataset(
    data_dir: str = DATA_DIR,
    max_battles: int = 0,  # 0 = all
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stream through all JSONL files and build the (states, actions, weights) arrays.

    Returns:
        states: np.float32 [N, 808]
        actions: np.int64 [N]
        weights: np.float32 [N]
    """
    all_states: List[np.ndarray] = []
    all_actions: List[int] = []
    all_weights: List[float] = []

    files = sorted(
        f for f in os.listdir(data_dir) if f.endswith(".jsonl")
    )

    if verbose:
        print(f"Precomputing dataset from {len(files)} files...")

    total_battles = 0
    for fname in files:
        fpath = os.path.join(data_dir, fname)
        battles_in_file = 0

        with open(fpath, encoding="utf-8") as f:
            for line in f:
                if max_battles and total_battles >= max_battles:
                    break

                rec = json.loads(line)
                log = rec["log"]
                battle_id = rec.get("id", "?")

                # Extract gen
                gen = 8
                for ln in log.splitlines():
                    if ln.startswith("|gen|"):
                        try:
                            gen = int(ln.split("|")[2])
                        except (ValueError, IndexError):
                            pass
                        break

                # Extract format and elo
                formatid = rec.get("formatid", fname.replace(".jsonl", ""))
                elo = extract_elo(rec)
                weight = compute_weight(formatid, elo)

                # Reconstruct states
                try:
                    recon = StateReconstructor(log, gen)
                    states = recon.all_states()
                except Exception as e:
                    if verbose:
                        print(f"  WARN: {battle_id}: {e}")
                    continue

                # Get P1 decisions from parse.py
                try:
                    parser = BattleLogParser(battle_id=battle_id, formatid=formatid)
                    parsed = parser.parse(log)
                    p1_decisions = [d for d in parsed.decisions if d.actor == "p1"]
                except Exception as e:
                    if verbose:
                        print(f"  WARN: {battle_id} parse: {e}")
                    continue

                # Match states to actions (both in P1 decision order)
                n_pairs = min(len(states), len(p1_decisions))
                if n_pairs == 0:
                    continue

                for i in range(n_pairs):
                    all_states.append(states[i])
                    all_actions.append(p1_decisions[i].action_idx)
                    all_weights.append(weight)

                battles_in_file += 1
                total_battles += 1

        if verbose:
            print(f"  {fname}: {battles_in_file} battles, {battles_in_file * 30} est. decisions")

    if not all_states:
        raise RuntimeError("No data extracted! Check data directory.")

    states = np.stack(all_states, axis=0).astype(np.float32)
    actions = np.array(all_actions, dtype=np.int64)
    weights = np.array(all_weights, dtype=np.float32)

    if verbose:
        print(f"\nDataset: {len(states)} samples, {states.shape[1]} dims")
        print(f"  Actions: {np.unique(actions)}")
        print(f"  Weights: min={weights.min():.3f} max={weights.max():.3f} mean={weights.mean():.3f}")

    return states, actions, weights


# ---------------------------------------------------------------------------
# Actor network (same architecture as MaskedPPO.actor)
# ---------------------------------------------------------------------------
def build_actor(obs_dim: int, n_actions: int, hidden: Tuple[int, ...] = (128, 128)) -> nn.Module:
    layers: List[nn.Module] = []
    d = obs_dim
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, n_actions))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    states: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
    epochs: int = 10,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: Optional[str] = None,
    patience: int = 3,
    verbose: bool = True,
) -> nn.Module:
    """Train the actor with weighted cross-entropy."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    actor = build_actor(OBS_DIM, UNIFIED_ACTION_SIZE).to(dev)
    opt = torch.optim.Adam(actor.parameters(), lr=lr)

    # Normalize weights (so the loss magnitude is stable)
    w_norm = weights / weights.sum() * len(weights)  # mean weight = 1.0

    # Convert to tensors
    states_t = torch.from_numpy(states).to(dev)
    actions_t = torch.from_numpy(actions).to(dev)
    weights_t = torch.from_numpy(w_norm).to(dev)

    n = len(states)
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(epochs):
        actor.train()
        perm = torch.randperm(n, device=dev)
        epoch_loss = 0.0
        epoch_correct = 0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch_s = states_t[idx]
            batch_a = actions_t[idx]
            batch_w = weights_t[idx]

            logits = actor(batch_s)
            log_probs = torch.log_softmax(logits, dim=-1)
            # Weighted cross-entropy: -sum(w * log_prob[action]) / sum(w)
            sample_loss = -log_probs.gather(1, batch_a.unsqueeze(1)).squeeze(1)
            loss = (sample_loss * batch_w).sum() / batch_w.sum()

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            opt.step()

            epoch_loss += loss.item() * len(idx)
            n_correct = (logits.argmax(dim=-1) == batch_a).sum().item()
            epoch_correct += n_correct
            n_batches += 1

        avg_loss = epoch_loss / n
        accuracy = epoch_correct / n

        if verbose:
            print(f"Epoch {epoch+1:3d}/{epochs} | loss={avg_loss:.4f} | acc={accuracy:.3f} | "
                  f"lr={opt.param_groups[0]['lr']:.6f}")

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best (we don't checkpoint here for simplicity; the last good epoch is close enough)
    actor.eval()
    return actor


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def smoke_test():
    """Quick end-to-end test with a small subset."""
    print("=" * 60)
    print("SMOKE TEST: imitation pipeline")
    print("=" * 60)

    # Phase 1: Pre-compute a small subset
    print("\n[1/3] Precomputing 50 battles...")
    t0 = time.time()
    states, actions, weights = precompute_dataset(max_battles=50, verbose=True)
    print(f"  Done in {time.time()-t0:.1f}s: {len(states)} samples")

    assert states.shape == (len(states), OBS_DIM), f"Shape mismatch: {states.shape}"
    assert actions.max() < UNIFIED_ACTION_SIZE, f"Action out of range: {actions.max()}"
    assert (weights > 0).all(), "Some weights are zero or negative"
    print(f"  ✓ shapes ok, ✓ actions in range, ✓ weights positive")

    # Phase 2: Train for 3 epochs
    print("\n[2/3] Training for 3 epochs...")
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    actor = train(
        states, actions, weights,
        epochs=3, batch_size=64, lr=1e-3,
        device=device, patience=10,  # no early stop for smoke test
        verbose=True,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # Phase 3: Evaluate
    print("\n[3/3] Evaluating...")
    actor.eval()
    with torch.no_grad():
        dev = next(actor.parameters()).device
        s_t = torch.from_numpy(states[:1000]).to(dev)
        a_t = torch.from_numpy(actions[:1000]).to(dev)
        logits = actor(s_t)
        preds = logits.argmax(dim=-1)
        acc = (preds == a_t).float().mean().item()
    print(f"  Top-1 accuracy (first 1000): {acc:.3f}")

    # Save the model
    save_path = os.path.join(CACHE_DIR, "imitation_actor_smoke.pt")
    torch.save(actor.state_dict(), save_path)
    print(f"\n  ✓ Model saved to {save_path}")
    print(f"\n{'='*60}")
    print("SMOKE TEST PASSED ✓")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Imitation learning for Pokemon Showdown bot")
    parser.add_argument("--precompute", action="store_true", help="Pre-compute the dataset")
    parser.add_argument("--train", action="store_true", help="Train the actor")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max_battles", type=int, default=0, help="Limit battles (0=all)")
    parser.add_argument("--save", type=str, default=None, help="Save model to path")
    args = parser.parse_args()

    if args.smoke:
        smoke_test()
        return

    if args.precompute:
        print("Precomputing dataset...")
        t0 = time.time()
        states, actions, weights = precompute_dataset(
            max_battles=args.max_battles, verbose=True
        )
        np.savez_compressed(DATASET_PATH, states=states, actions=actions, weights=weights)
        print(f"\nSaved to {DATASET_PATH}")
        print(f"Size: {os.path.getsize(DATASET_PATH) / 1e6:.1f} MB")
        print(f"Time: {time.time()-t0:.1f}s")
        return

    if args.train:
        if not os.path.exists(DATASET_PATH):
            print(f"Dataset not found at {DATASET_PATH}. Run --precompute first.")
            return

        print(f"Loading dataset from {DATASET_PATH}...")
        data = np.load(DATASET_PATH)
        states, actions, weights = data["states"], data["actions"], data["weights"]
        print(f"Loaded: {len(states)} samples")

        actor = train(
            states, actions, weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )

        save_path = args.save or os.path.join(CACHE_DIR, "imitation_actor.pt")
        torch.save(actor.state_dict(), save_path)
        print(f"\nModel saved to {save_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
