"""S4.3 — behavior-clone warm-start (roadmap step 6), increment 1: the BC
trainer + the held-out "always-switch" verdict (PLAN.md §3c S4.3).

This is a THIN trainer on top of the existing S4.2 pieces — it reuses, it does
not rewrite, ``build_samples`` / ``samples_to_batch`` / ``bc_loss`` / ``PokeNet``::

    build_vocabs -> load_local(N) -> replay_many -> build_samples (per battle)
        -> split BATTLES into train / held-out (a battle's decisions never
           straddle the split, so there is no train/eval leakage)
        -> train PokeNet (Adam, grad clip, early stopping on held-out top-1)
        -> evaluate held-out  -> the "always-switch" verdict (with numbers)
        -> save the best checkpoint (best held-out top-1)
        -> report

The whole point of increment 1 is to answer, WITH NUMBERS, the user's core
concern — hit in the old system: *the net memorized "default = switch" instead
of state -> action*. So the verdict is built to FAIL a net that only gets the
switches right:

  * **move-accuracy**   = correct move-decisions / move-decisions — a HARD
    gate: must beat chance (1 / |legal moves|). A collapse-to-switch net scores
    ~0 here, far below chance.
  * **switch-accuracy** = correct switch-decisions / switch-decisions — must
    beat chance (1 / |legal switches|).
  * **predicted switch-rate** = frac(argmax is a switch) vs the **human**
    switch-rate (frac of decisions that ARE switches). A net that predicts
    switch on ~100% while humans do ~38% is following the action prior, not the
    state — this is the collapse signature. The same check catches the mirror
    collapse ("always use a move, never switch") via the switch-accuracy gate.
  * **held-out top-1**  = overall accuracy vs chance (1 / |legal|).

POV: p1-POV samples only (the S4.2 batch is p1-POV; p2-POV is a corpus
presentation flip and is deferred — "let's get a working system first").
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from corpus.pull import load_local
from corpus.trajectories import replay_many
from net import (
    PokeNet,
    bc_loss,
    batch_to_torch,
    build_samples,
    build_vocabs,
    samples_to_batch,
)
from net.batch import Sample

#: the "always-switch" verdict margins (recorded design choices, S4.3):
#:  * a predicted switch-rate within this margin of the human rate is "near";
#:    a net that deviates by more is following the action prior (collapse).
SWITCH_RATE_MARGIN = 0.15
#:  * early-stop after this many epochs without a held-out top-1 gain.
DEFAULT_PATIENCE = 4


@dataclass
class Split:
    """A seeded train / held-out split of the corpus, BY BATTLE."""
    n_loaded: int
    n_replayed: int
    n_skipped: int
    skipped_reasons: Dict[str, int]
    train_samples: List[Sample]
    heldout_samples: List[Sample]
    n_train_battles: int
    n_heldout_battles: int

    @property
    def human_switch_rate(self) -> float:
        """The human switch-rate over the TRAINING decisions (p1-POV, legal):
        the target the net's predicted switch-rate is compared against."""
        n = len(self.train_samples)
        if not n:
            return 0.0
        return sum(1 for s in self.train_samples if s.kind == "switch") / n


def make_split(battles: List[dict], vocabs, max_decisions: int = 40,
               heldout_frac: float = 0.2, seed: int = 0) -> Split:
    """Replay ``battles``, build p1-POV samples per battle, and split the
    REPLAYED battles into train / held-out. Every decision of a battle lands
    on exactly one side — no battle straddles the split (no leakage)."""
    traces = replay_many(battles)
    ok = [t for t in traces if t.skipped is None]
    per: List[Tuple[object, List[Sample]]] = []
    for t in ok:
        s = build_samples(t, vocabs, max_decisions=max_decisions)
        if s:
            per.append((t, s))

    idx = list(range(len(per)))
    random.Random(seed).shuffle(idx)
    n_held = max(1, int(round(len(per) * heldout_frac))) if len(per) >= 2 else 0
    held = set(idx[:n_held])

    train_samples: List[Sample] = []
    heldout_samples: List[Sample] = []
    n_tb = n_hb = 0
    for i, (t, s) in enumerate(per):
        if i in held:
            heldout_samples.extend(s)
            n_hb += 1
        else:
            train_samples.extend(s)
            n_tb += 1

    reasons: Dict[str, int] = {}
    for t in traces:
        if t.skipped is not None:
            reasons[t.skipped] = reasons.get(t.skipped, 0) + 1

    return Split(
        n_loaded=len(battles), n_replayed=len(ok), n_skipped=len(traces) - len(ok),
        skipped_reasons=reasons,
        train_samples=train_samples, heldout_samples=heldout_samples,
        n_train_battles=n_tb, n_heldout_battles=n_hb,
    )


# ── evaluation / the "always-switch" verdict ────────────────────────────
@torch.no_grad()
def evaluate(net: PokeNet, samples: List[Sample], device: str = "cpu",
             eval_batch: int = 64) -> Dict:
    """Run ``net`` over ``samples`` and compute the S4.3 verdict metrics.

    For each sample the net's prediction is ``argmax`` over its LEGAL actions
    at the sample's supervised timestep (the LAST real turn of its sequence).
    Correctness is ``argmax == target`` (the exact human action). Chance
    baselines come from the sample's own legal-action list (``s.actions``):

      * ``move_chance_legal``  = avg 1/|legal| over move-decisions — the HONEST
        random baseline (the net picks from the full legal list: moves +
        switches). This is what ``move_beats_chance`` gates on, so a fresh mon
        (|moves|=1) does not degenerate the gate to 1.0.
      * ``move_chance``        = avg 1/|moves| — a reported insight (random
        among the mon's moves only).
      * ``switch_chance``      = avg 1/|legal switches| (for a switch-decision
        |legal| IS the switch list).
      * ``overall_chance``     = avg 1/|legal| over all decisions.
    """
    if not samples:
        return {"n": 0}

    n_move = n_switch = 0
    move_correct = switch_correct = 0
    pred_switch = 0
    correct_total = 0
    chance_move = 0.0      # sum of 1/|moves| over move-decisions (insight: random
                          # among the mon's moves)
    chance_move_legal = 0.0  # sum of 1/|legal| over move-decisions — the HONEST
                          # random baseline: the net picks from the FULL legal list
                          # (moves + switches), so a random net hits the target with
                          # prob 1/|legal|. Robust to a fresh mon (|moves|=1, where
                          # 1/|moves|=1.0 would degenerate the gate).
    chance_switch = 0.0   # sum of 1/|legal switches| over switch-decisions
    chance_overall = 0.0  # sum of 1/|legal| over all decisions
    human_switch = 0

    for start in range(0, len(samples), eval_batch):
        chunk = samples[start:start + eval_batch]
        sb = samples_to_batch(chunk)
        b = batch_to_torch(sb["batch"], device)
        ts = torch.as_tensor(sb["ts"], dtype=torch.int64, device=device)
        out = net(b)
        logits = out["logits"]                 # (B, T, A)
        mask = b["action_mask"]                # (B, T, A)
        B, A = logits.shape[0], logits.shape[-1]
        idx = ts.view(B, 1, 1).expand(B, 1, A)
        row = torch.gather(logits, 1, idx).squeeze(1)     # (B, A)
        mrow = torch.gather(mask, 1, idx).squeeze(1)      # (B, A)
        neg = torch.finfo(row.dtype).min
        row_m = row.masked_fill(mrow == 0, neg)
        any_legal = (mrow == 1).any(dim=-1)               # (B,)
        # degenerate all-masked rows -> unmasked (never NaN, never all- -inf)
        row_m = torch.where(any_legal.unsqueeze(-1), row_m, row)
        pred = row_m.argmax(dim=-1).cpu().tolist()        # [B]

        for j, s in enumerate(chunk):
            acts = s.actions
            n_legal = len(acts)
            n_moves = sum(1 for a in acts if a.get("kind") == "move")
            n_sw = sum(1 for a in acts if a.get("kind") == "switch")
            is_correct = int(pred[j]) == int(s.target)
            is_pred_sw = acts[int(pred[j])].get("kind") == "switch"
            correct_total += is_correct
            pred_switch += is_pred_sw
            chance_overall += 1.0 / max(1, n_legal)
            if s.kind == "move":
                n_move += 1
                move_correct += is_correct
                chance_move += 1.0 / max(1, n_moves)
                chance_move_legal += 1.0 / max(1, n_legal)
            elif s.kind == "switch":
                n_switch += 1
                switch_correct += is_correct
                chance_switch += 1.0 / max(1, n_sw)
                human_switch += 1

    n = len(samples)
    move_acc = move_correct / n_move if n_move else 0.0
    switch_acc = switch_correct / n_switch if n_switch else 0.0
    move_chance = chance_move / n_move if n_move else 0.0
    move_chance_legal = chance_move_legal / n_move if n_move else 0.0
    switch_chance = chance_switch / n_switch if n_switch else 0.0
    top1 = correct_total / n
    overall_chance = chance_overall / n
    pred_switch_rate = pred_switch / n
    human_switch_rate = human_switch / n
    return {
        "n": n, "n_move": n_move, "n_switch": n_switch,
        "move_acc": move_acc, "move_chance": move_chance,
        "move_chance_legal": move_chance_legal,
        "move_beats_chance": bool(n_move) and move_acc > move_chance_legal,
        "switch_acc": switch_acc, "switch_chance": switch_chance,
        "switch_beats_chance": bool(n_switch) and switch_acc > switch_chance,
        "top1_acc": top1, "overall_chance": overall_chance,
        "top1_beats_chance": top1 > overall_chance,
        "pred_switch_rate": pred_switch_rate,
        "human_switch_rate": human_switch_rate,
        "switch_rate_gap": pred_switch_rate - human_switch_rate,
    }


def verdict(m: Dict) -> Tuple[bool, List[str]]:
    """The S4.3 "always-switch" gate. Returns ``(passed, failures)``.

    A net that only gets switches right FAILS (move-accuracy collapses to ~0,
    below chance). A net that only gets moves right also FAILS (switch-rate
    collapses to ~0, far from the human rate, and switch-accuracy ~0).
    """
    fails: List[str] = []
    if not m.get("n"):
        return False, ["no held-out samples to evaluate"]

    # HARD gate 1 — move-accuracy must beat chance (anti collapse-to-switch).
    if not m.get("n_move"):
        fails.append("no move-decisions in the held-out set")
    elif not m["move_beats_chance"]:
        fails.append(f"move-accuracy {m['move_acc']:.3f} <= chance "
                     f"{m['move_chance']:.3f}  (collapse to switches?)")

    # HARD gate 2 — switch-accuracy must beat chance.
    if not m.get("n_switch"):
        fails.append("no switch-decisions in the held-out set")
    elif not m["switch_beats_chance"]:
        fails.append(f"switch-accuracy {m['switch_acc']:.3f} <= chance "
                     f"{m['switch_chance']:.3f}")

    # HARD gate 3 — predicted switch-rate must sit near the human rate
    # (catches ANY "always one action type" collapse, both directions).
    if abs(m.get("switch_rate_gap", 0.0)) > SWITCH_RATE_MARGIN:
        fails.append(f"predicted switch-rate {m['pred_switch_rate']:.3f} far "
                     f"from human {m['human_switch_rate']:.3f} "
                     f"(gap {m['switch_rate_gap']:+.3f} > {SWITCH_RATE_MARGIN})")

    # overall top-1 must beat overall chance.
    if not m.get("top1_beats_chance"):
        fails.append(f"held-out top-1 {m['top1_acc']:.3f} <= chance "
                     f"{m['overall_chance']:.3f}")

    return (not fails), fails


# ── training loop (early stopping on held-out top-1) ────────────────────
def train(net: PokeNet, split: Split, device: str = "cpu",
          batch_size: int = 24, lr: float = 3e-3, epochs: int = 15,
          max_norm: float = 1.0, patience: int = DEFAULT_PATIENCE,
          seed: int = 0, log=print) -> Dict:
    """Train ``net`` on ``split.train_samples``, early-stopping on the
    held-out top-1. Returns the final held-out metrics + the best seen."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    N = len(split.train_samples)
    order = rng.permutation(N)

    best_top1 = -1.0
    best_metrics: Dict = {}
    best_state: Optional[Dict] = None
    bad = 0

    for epoch in range(1, epochs + 1):
        net.train()
        rng.shuffle(order)
        tot, nsteps = 0.0, 0
        for start in range(0, N, batch_size):
            idx = order[start:start + batch_size]
            batch = [split.train_samples[i] for i in idx]
            sb = samples_to_batch(batch)
            b = batch_to_torch(sb["batch"], device)
            targets = torch.as_tensor(sb["targets"], dtype=torch.int64, device=device)
            ts = torch.as_tensor(sb["ts"], dtype=torch.int64, device=device)
            opt.zero_grad()
            out = net(b)
            loss = bc_loss(out["logits"], b["action_mask"], targets, ts).mean()
            loss.backward()
            if max_norm:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            opt.step()
            tot += loss.item()
            nsteps += 1

        m = evaluate(net, split.heldout_samples, device=device)
        top1 = m["top1_acc"]
        log(f"epoch {epoch:2d}/{epochs}  train_loss {tot / max(1, nsteps):.4f}  "
            f"held top1 {top1:.3f} (chance {m['overall_chance']:.3f})  "
            f"move {m['move_acc']:.3f}/{m['move_chance']:.3f}  "
            f"sw {m['switch_acc']:.3f}/{m['switch_chance']:.3f}  "
            f"pred_sw {m['pred_switch_rate']:.3f} (human {m['human_switch_rate']:.3f})")
        if top1 > best_top1:
            best_top1 = top1
            best_metrics = m
            best_state = {k: v.detach().cpu().clone()
                          for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f"early stopping at epoch {epoch} "
                    f"(no held-out gain for {patience} epochs)")
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return {"best_top1": best_top1, "best_metrics": best_metrics,
            "final_metrics": evaluate(net, split.heldout_samples, device=device)}


def _save_checkpoint(net: PokeNet, path: str, extra: Optional[Dict] = None) -> None:
    """Save the net (weights + vocab sizes + layout checksum) as a torch
    payload, and the run metadata (metrics, verdict) as a JSON sidecar.

    ``state_for_save()`` returns TENSORS, so this must be ``torch.save`` — the
    old ``json.dump`` crashed with 'Object of type Tensor is not JSON
    serializable' (P1). The sidecar keeps the checkpoint pure (exactly what
    ``PokeNet.load`` reads) while the metrics stay human-inspectable."""
    import torch as _torch
    _torch.save(net.state_for_save(), path)
    if extra:
        with open(path + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(extra, fh, indent=2)


def load_checkpoint(path: str, vocabs) -> PokeNet:
    """Load a :func:`_save_checkpoint` torch payload back into a PokeNet.
    Verifies the vocab sizes + layout checksum match (loud error on drift)."""
    import torch as _torch
    payload = _torch.load(path, map_location="cpu", weights_only=True)
    return PokeNet.load(payload, vocabs)


# ── the increment-1 run ─────────────────────────────────────────────────
def run(n_battles: int = 50, fixture: str = "bot/data/replays/gen9ou.jsonl",
        max_decisions: int = 40, heldout_frac: float = 0.2,
        epochs: int = 15, batch_size: int = 24, lr: float = 3e-3,
        device: str = "cpu", seed: int = 0,
        out_dir: str = "scratch", ckpt_name: str = "s43_bc_best.pt",
        log=print) -> Dict:
    """Build the corpus split, train, evaluate, save the best checkpoint, and
    return the full result dict (metrics + the verdict)."""
    t0 = time.time()
    log(f"== S4.3 BC warm-start (increment 1) ==")
    log(f"N={n_battles} fixture={fixture} device={device} epochs={epochs} "
        f"batch={batch_size} lr={lr} seed={seed}")

    vocabs = build_vocabs()
    log(f"vocabs: {vocabs.sizes()}")

    log(f"[1/4] loading {n_battles} battles + replaying (cold node spawns, "
        f"~20-25 s/battle) ...")
    battles = load_local(fixture, n_battles)
    split = make_split(battles, vocabs, max_decisions=max_decisions,
                       heldout_frac=heldout_frac, seed=seed)
    log(f"      replayed {split.n_replayed}/{split.n_loaded} "
        f"(skipped {split.n_skipped}): {split.skipped_reasons}")
    log(f"      battles: {split.n_train_battles} train / "
        f"{split.n_heldout_battles} held-out")
    log(f"      samples: {len(split.train_samples)} train / "
        f"{len(split.heldout_samples)} held-out")
    log(f"      human switch-rate (train): {split.human_switch_rate:.3f}")
    if not split.train_samples or not split.heldout_samples:
        raise RuntimeError("empty train or held-out split — not enough replayed "
                           f"battles (replayed={split.n_replayed})")

    log(f"[2/4] building PokeNet ...")
    net = PokeNet(vocabs)
    n_params = sum(p.numel() for p in net.parameters())
    log(f"      {n_params/1e6:.1f}M parameters")

    log(f"[3/4] training (early stopping on held-out top-1) ...")
    res = train(net, split, device=device, batch_size=batch_size, lr=lr,
                epochs=epochs, seed=seed, log=log)

    log(f"[4/4] final held-out verdict ...")
    final = res["final_metrics"]
    ok, fails = verdict(final)
    log("\n" + "-" * 56)
    log(f"held-out n          : {final['n']}  (moves {final['n_move']}, "
        f"switches {final['n_switch']})")
    log(f"move-accuracy       : {final['move_acc']:.3f}   chance "
        f"{final['move_chance_legal']:.3f}  (1/|moves| {final['move_chance']:.3f})   "
        f"beats-chance {'YES' if final['move_beats_chance'] else 'no'}")
    log(f"switch-accuracy     : {final['switch_acc']:.3f}   chance "
        f"{final['switch_chance']:.3f}   beats-chance "
        f"{'YES' if final['switch_beats_chance'] else 'no'}")
    log(f"predicted switch-rate: {final['pred_switch_rate']:.3f}   human "
        f"{final['human_switch_rate']:.3f}   gap {final['switch_rate_gap']:+.3f}")
    log(f"held-out top-1      : {final['top1_acc']:.3f}   chance "
        f"{final['overall_chance']:.3f}   beats-chance "
        f"{'YES' if final['top1_beats_chance'] else 'no'}")
    log(f"VERDICT             : {'PASS — net is learning state->action' if ok else 'FAIL'}")
    for f in fails:
        log(f"    - {f}")
    log("-" * 56)

    import os
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, ckpt_name)
    _save_checkpoint(net, ckpt_path, extra={
        "step": "S4.3", "n_battles": n_battles, "seed": seed, "fixture": fixture,
        "human_switch_rate": split.human_switch_rate, "final": final,
        "verdict_pass": ok, "elapsed_s": round(time.time() - t0, 1),
    })
    log(f"best checkpoint -> {ckpt_path}  "
        f"(best held top-1 {res['best_top1']:.3f})")

    return {
        "vocabs_sizes": vocabs.sizes(),
        "split": {
            "n_loaded": split.n_loaded, "n_replayed": split.n_replayed,
            "n_skipped": split.n_skipped, "skipped_reasons": split.skipped_reasons,
            "n_train_battles": split.n_train_battles,
            "n_heldout_battles": split.n_heldout_battles,
            "n_train_samples": len(split.train_samples),
            "n_heldout_samples": len(split.heldout_samples),
            "human_switch_rate": split.human_switch_rate,
        },
        "n_params": n_params,
        "best_top1": res["best_top1"],
        "final_metrics": final,
        "verdict": {"pass": ok, "failures": fails},
        "checkpoint": ckpt_path,
        "elapsed_s": round(time.time() - t0, 1),
    }


def _cli(argv: Optional[List[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="S4.3 BC warm-start trainer + verdict")
    p.add_argument("--n", type=int, default=50, help="number of battles (corpus)")
    p.add_argument("--fixture", default="bot/data/replays/gen9ou.jsonl")
    p.add_argument("--max-decisions", type=int, default=40)
    p.add_argument("--heldout-frac", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="scratch")
    p.add_argument("--ckpt", default="s43_bc_best.pt")
    a = p.parse_args(argv)
    run(n_battles=a.n, fixture=a.fixture, max_decisions=a.max_decisions,
        heldout_frac=a.heldout_frac, epochs=a.epochs, batch_size=a.batch,
        lr=a.lr, device=a.device, seed=a.seed, out_dir=a.out, ckpt_name=a.ckpt)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
