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
import uuid
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

from corpus.pull import load_local
from corpus.trajectories import BattleTrace, replay_battle
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

#: D81 — the first-use/repeat leak-check (the friend's "is it learning or just
#: grabbing the new move?" signal). A net that keys on "is this a NEW move?" does
#: well on first-use decisions and poorly on repeats. The verdict FAILs that
#: shortcut even when overall move-accuracy still beats chance. D82 — the user
#: chose a MIX of a RELATIVE tell and an ABSOLUTE baseline (not a single
#: absolute gap), so a genuinely good net (both high, small gap) is not
#: false-flagged, and a weak net (both at chance) is not flagged either:
#:  * the gate only applies when BOTH splits have at least this many held-out
#:    move-decisions (below that the split is too small to be meaningful);
MIN_SPLIT_SAMPLES = 8
#:  * RELATIVE tell: FAIL only when first-use acc exceeds repeat acc by more
#:    than this (proportionally). A healthy state->action net does first-use and
#:    repeat comparably, so this stays small even when both are high (0.90/0.85
#:    passes); a "grab the new move?" net is far higher on first-use (0.95/0.40
#:    fails). Tunable (open question).
REL_SHORTCUT_MARGIN = 0.30
#:  * ABSOLUTE baseline: the relative tell only counts when first-use acc also
#:    clears this floor (well above a 4-move chance baseline ~0.25) — otherwise
#:    the "gap" is just both splits sitting at chance (a weak net, already caught
#:    by gates 1-3). Tunable (open question).
ABS_FIRST_USE_FLOOR = 0.35


class BCStopped(RuntimeError):
    """A graceful stop landed mid-run (the GUI's stop flag, D86).

    Raised by ``make_split`` when ``stop_check()`` is true between battle
    replays. ``run`` catches it: the run ends cleanly as 'stopped' — a stop
    during the REPLAY phase trains nothing (there is no corpus yet); a stop
    during the TRAIN phase keeps the work done so far (checkpoint saved,
    marked stopped)."""


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


def make_split(battles: Optional[List[dict]] = None, vocabs=None,
               max_decisions: int = 40, heldout_frac: float = 0.2, seed: int = 0,
               traces: Optional[List[BattleTrace]] = None,
               stop_check: Optional[Callable[[], bool]] = None,
               heartbeat: Optional[Callable[[Dict], None]] = None) -> Split:
    """Build p1-POV samples and split the REPLAYED battles into train / held-out.

    Every decision of a battle lands on exactly one side — no battle straddles
    the split (no leakage).

    Supply EITHER raw ``battles`` (replayed here, the slow path) OR pre-built
    ``traces`` (from ``corpus.cache.load_cache`` — the fast path, no replay).

    D86 hooks (both optional; ``None`` = the old synchronous behavior):
      * ``stop_check()`` — called BEFORE every battle replay; when it returns
        true, :class:`BCStopped` is raised AFTER the previous battle finished
        (a stop lands within one battle, never mid-battle).
      * ``heartbeat(info)`` — called after every battle replay with
        ``{"phase": "replay", "battle": i, "total": n}`` (progress for the
        GUI dashboard)."""
    if traces is None:
        if battles is None:
            raise ValueError("make_split needs battles= or traces=")
        # D86: the per-battle loop (``corpus.trajectories.replay_many`` was
        # exactly this loop; we inline it so the hooks can land between
        # battles — same replay_battle, same order, no second code path).
        traces: List[BattleTrace] = []
        n_b = len(battles)
        for i, b in enumerate(battles, 1):
            if stop_check is not None and stop_check():
                raise BCStopped(f"stop requested during replay "
                                f"(after {i - 1}/{n_b} battles)")
            traces.append(replay_battle(b, 30))
            if heartbeat is not None:
                heartbeat({"phase": "replay", "battle": i, "total": n_b})
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
        n_loaded=(len(battles) if battles is not None else len(traces)),
        n_replayed=len(ok), n_skipped=len(traces) - len(ok),
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
    # D81: first-use / repeat move-decision split (the leak-check signal).
    n_first = n_repeat = 0
    first_correct = repeat_correct = 0
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
                # D81: split the move-decision accuracy on first-use vs repeat.
                if getattr(s, "first_use", False):
                    n_first += 1
                    first_correct += is_correct
                else:
                    n_repeat += 1
                    repeat_correct += is_correct
            elif s.kind == "switch":
                n_switch += 1
                switch_correct += is_correct
                chance_switch += 1.0 / max(1, n_sw)
                human_switch += 1

    n = len(samples)

    # POSITION-BIAS baseline: the accuracy of always predicting ONE fixed
    # action index (the best one, per kind). A net that learned nothing can
    # still hit this if targets are concentrated on a few indices — which is
    # exactly what happens when action lists are short (a 1-2 revealed-move
    # list puts the human's move at index 0 most of the time). This is the
    # real "no learning" ceiling and MUST be in the verdict gate (the S4.3
    # N=50 "PASS" was 0.653 = exactly the always-index-0 baseline; the net
    # predicted index 0 on 219/219 held-out decisions and learned nothing).
    import collections as _c

    def _best_const(ss: List[Sample]) -> float:
        if not ss:
            return 0.0
        d = _c.Counter(s.target for s in ss)
        return max(d.values()) / len(ss)

    move_ss = [s for s in samples if s.kind == "move"]
    sw_ss = [s for s in samples if s.kind == "switch"]
    move_const = _best_const(move_ss)
    sw_const = _best_const(sw_ss)
    top1_const = _best_const(samples)

    move_acc = move_correct / n_move if n_move else 0.0
    switch_acc = switch_correct / n_switch if n_switch else 0.0
    move_chance = chance_move / n_move if n_move else 0.0
    move_chance_legal = chance_move_legal / n_move if n_move else 0.0
    switch_chance = chance_switch / n_switch if n_switch else 0.0
    top1 = correct_total / n
    overall_chance = chance_overall / n
    pred_switch_rate = pred_switch / n
    human_switch_rate = human_switch / n
    # D81: the first-use / repeat move-accuracy split (None when a split has no
    # samples). shortcut_gap = first_use_acc - repeat_acc (None if either is).
    first_use_acc = first_correct / n_first if n_first else None
    repeat_acc = repeat_correct / n_repeat if n_repeat else None
    return {
        "n": n, "n_move": n_move, "n_switch": n_switch,
        "move_acc": move_acc, "move_chance": move_chance,
        "move_chance_legal": move_chance_legal,
        "move_const_acc": move_const,
        "move_beats_chance": bool(n_move) and move_acc > move_chance_legal,
        "move_beats_const": bool(n_move) and move_acc > move_const,
        "switch_acc": switch_acc, "switch_chance": switch_chance,
        "switch_const_acc": sw_const,
        "switch_beats_chance": bool(n_switch) and switch_acc > switch_chance,
        "switch_beats_const": bool(n_switch) and switch_acc > sw_const,
        "top1_acc": top1, "overall_chance": overall_chance,
        "top1_const_acc": top1_const,
        "top1_beats_chance": top1 > overall_chance,
        "top1_beats_const": top1 > top1_const,
        "pred_switch_rate": pred_switch_rate,
        "human_switch_rate": human_switch_rate,
        "switch_rate_gap": pred_switch_rate - human_switch_rate,
        "n_first_use": n_first, "n_repeat": n_repeat,
        "first_use_move_acc": first_use_acc,
        "repeat_move_acc": repeat_acc,
        "shortcut_gap": (first_use_acc - repeat_acc)
                        if (first_use_acc is not None and repeat_acc is not None)
                        else None,
    }


def verdict(m: Dict) -> Tuple[bool, List[str]]:
    """The S4.3 "always-switch" gate. Returns ``(passed, failures)``.

    A net that only gets switches right FAILS (move-accuracy collapses to ~0,
    below chance). A net that only gets moves right also FAILS (switch-rate
    collapses to ~0, far from the human rate, and switch-accuracy ~0).

    Every accuracy gate is two-sided: the net must beat BOTH the random
    chance baseline (1/|legal|) AND the position-bias baseline (the best
    single constant action index). The second is the real "learned nothing"
    ceiling: with short action lists the human target sits at index 0 most
    of the time, so a constant-index net can clear chance without learning
    anything (that is exactly how the first S4.3 N=50 run "passed").
    """
    fails: List[str] = []
    if not m.get("n"):
        return False, ["no held-out samples to evaluate"]

    # HARD gate 1 — move-accuracy must beat chance AND the constant-index
    # position baseline (anti collapse-to-switch, anti index-0 bias).
    if not m.get("n_move"):
        fails.append("no move-decisions in the held-out set")
    elif not m["move_beats_chance"]:
        fails.append(f"move-accuracy {m['move_acc']:.3f} <= chance "
                     f"{m.get('move_chance_legal', m.get('move_chance', 0.0)):.3f}  "
                     f"(collapse to switches?)")
    elif not m["move_beats_const"]:
        fails.append(f"move-accuracy {m['move_acc']:.3f} <= constant-index "
                     f"baseline {m['move_const_acc']:.3f}  (net is a fixed "
                     f"action — it learned nothing)")

    # HARD gate 2 — switch-accuracy must beat chance AND its constant-index
    # baseline.
    if not m.get("n_switch"):
        fails.append("no switch-decisions in the held-out set")
    elif not m["switch_beats_chance"]:
        fails.append(f"switch-accuracy {m['switch_acc']:.3f} <= chance "
                     f"{m['switch_chance']:.3f}")
    elif not m["switch_beats_const"]:
        fails.append(f"switch-accuracy {m['switch_acc']:.3f} <= constant-"
                     f"index baseline {m['switch_const_acc']:.3f}  (net is a "
                     f"fixed action — it learned nothing)")

    # HARD gate 3 — predicted switch-rate must sit near the human rate
    # (catches ANY "always one action type" collapse, both directions).
    if abs(m.get("switch_rate_gap", 0.0)) > SWITCH_RATE_MARGIN:
        fails.append(f"predicted switch-rate {m['pred_switch_rate']:.3f} far "
                     f"from human {m['human_switch_rate']:.3f} "
                     f"(gap {m['switch_rate_gap']:+.3f} > {SWITCH_RATE_MARGIN})")

    # overall top-1 must beat overall chance AND the overall constant-index
    # baseline.
    if not m.get("top1_beats_chance"):
        fails.append(f"held-out top-1 {m['top1_acc']:.3f} <= chance "
                     f"{m['overall_chance']:.3f}")
    elif not m.get("top1_beats_const"):
        fails.append(f"held-out top-1 {m['top1_acc']:.3f} <= constant-index "
                     f"baseline {m['top1_const_acc']:.3f}  (net is a fixed "
                     f"action — it learned nothing)")

    # HARD gate 4 — the first-use/repeat leak (D81, the friend's "is it
    # learning or just grabbing the new move?" check). A net that keys on
    # "is this a NEW move?" scores high on first-use and low on repeats even
    # though its overall move-accuracy can still beat chance — the earlier
    # gates do not catch that. D82 — a MIX of a RELATIVE tell (first-use far
    # above repeat) and an ABSOLUTE baseline (first-use clear of a floor), so
    # a genuinely good net (both high, small gap) and a weak net (both at
    # chance) are both NOT flagged. Enforce only when BOTH splits are large.
    n_fu = m.get("n_first_use") or 0
    n_rp = m.get("n_repeat") or 0
    fu = m.get("first_use_move_acc")
    rp = m.get("repeat_move_acc")
    if (n_fu >= MIN_SPLIT_SAMPLES and n_rp >= MIN_SPLIT_SAMPLES
            and fu is not None and rp is not None
            and fu >= ABS_FIRST_USE_FLOOR
            and fu > rp * (1.0 + REL_SHORTCUT_MARGIN)):
        fails.append(
            f"first-use move-acc {fu:.3f} >> repeat move-acc {rp:.3f} "
            f"({fu / max(rp, 1e-9):.2f}x > {1.0 + REL_SHORTCUT_MARGIN:.2f}x AND "
            f"first-use >= {ABS_FIRST_USE_FLOOR}) — the net is using "
            f"\"new move\" as a shortcut, not the state")

    return (not fails), fails


# ── training loop (early stopping on held-out top-1) ────────────────────
def train(net: PokeNet, split: Split, device: str = "cpu",
          batch_size: int = 24, lr: float = 3e-3, epochs: int = 15,
          max_norm: float = 1.0, patience: int = DEFAULT_PATIENCE,
          seed: int = 0, log=print,
          stop_check: Optional[Callable[[], bool]] = None,
          heartbeat: Optional[Callable[[Dict], None]] = None) -> Dict:
    """Train ``net`` on ``split.train_samples``, early-stopping on the
    held-out top-1. Returns the final held-out metrics + the best seen.

    D86 hooks (optional; ``None`` = the old synchronous behavior):
      * ``stop_check()`` — checked BEFORE every epoch; when true the epoch
        loop breaks (the epochs already done keep their best-state restore);
      * ``heartbeat(info)`` — after every epoch with
        ``{"phase": "train", "epoch": e, "total_epochs": n, "top1": t}``.
    The result dict gains ``"stopped"`` (True iff the loop ended via the
    stop hook, not patience)."""
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

    stopped = False
    for epoch in range(1, epochs + 1):
        if stop_check is not None and stop_check():
            log(f"stop requested — ending training after epoch {epoch - 1}")
            stopped = True
            break
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
        if heartbeat is not None:
            heartbeat({"phase": "train", "epoch": epoch, "total_epochs": epochs,
                       "top1": round(float(top1), 5)})

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return {"best_top1": best_top1, "best_metrics": best_metrics,
            "final_metrics": evaluate(net, split.heldout_samples, device=device),
            "stopped": stopped}


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
        out_dir: str = "scratch", ckpt_name: Optional[str] = None,
        initial_ckpt: Optional[str] = None,
        log=print,
        stop_check: Optional[Callable[[], bool]] = None,
        heartbeat: Optional[Callable[[Dict], None]] = None) -> Dict:
    """Build the corpus split, train, evaluate, save the best checkpoint, and
    return the full result dict (metrics + the verdict).

    D86: optional ``stop_check`` / ``heartbeat`` hooks are forwarded to
    :func:`make_split` (per battle) and :func:`train` (per epoch) — the GUI's
    graceful stop + progress. A stop during REPLAY ends the run with no
    checkpoint (``{\"stopped\": True, \"phase\": \"replay\"}``); a stop during
    TRAIN keeps the work so far (checkpoint saved, ``stopped: True`` in the
    metadata). Both ``None`` = the old synchronous CLI behavior."""
    # D99: default checkpoint name = {n}b_{type}_{id}.pt (battles, type, unique id)
    # No date — avoids same-day collisions; the id is a 4-char uuid4 hex.
    # League-friendly: stable, referenceable, sortable by battle count.
    if ckpt_name is None:
        short_id = uuid.uuid4().hex[:4]
        ckpt_name = f"{n_battles}b_bc_{short_id}.pt"

    t0 = time.time()
    log(f"== S4.3 BC warm-start (increment 1) ==")
    log(f"N={n_battles} fixture={fixture} device={device} epochs={epochs} "
        f"batch={batch_size} lr={lr} seed={seed} ckpt={ckpt_name}"
        f"{'  FINE-TUNING from ' + str(initial_ckpt) if initial_ckpt else ''}")

    vocabs = build_vocabs()
    log(f"vocabs: {vocabs.sizes()}")

    log(f"[1/4] loading {n_battles} battles + replaying (cold node spawns, "
        f"~20-25 s/battle) ...")
    battles = load_local(fixture, n_battles)
    try:
        split = make_split(battles, vocabs, max_decisions=max_decisions,
                           heldout_frac=heldout_frac, seed=seed,
                           stop_check=stop_check, heartbeat=heartbeat)
    except BCStopped as exc:
        log(f"STOPPED: {exc} — nothing trained, no checkpoint")
        return {"stopped": True, "phase": "replay", "error": str(exc),
                "elapsed_s": round(time.time() - t0, 1)}
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
    if initial_ckpt:
        # continue / fine-tune from an existing checkpoint (the GUI's
        # "Start from ckpt" option). load_checkpoint verifies the vocab sizes
        # + layout checksum match (a loud error, not silent shape drift), and
        # — like the random-init path below — builds on CPU; train() moves it
        # to the device.
        net = load_checkpoint(initial_ckpt, vocabs)
        log(f"      continuing from existing checkpoint: {initial_ckpt}")
    else:
        net = PokeNet(vocabs)
        log(f"      (random init)")
    n_params = sum(p.numel() for p in net.parameters())
    log(f"      {n_params/1e6:.1f}M parameters")

    log(f"[3/4] training (early stopping on held-out top-1) ...")
    res = train(net, split, device=device, batch_size=batch_size, lr=lr,
                epochs=epochs, seed=seed, log=log,
                stop_check=stop_check, heartbeat=heartbeat)

    log(f"[4/4] final held-out verdict ...")
    final = res["final_metrics"]
    ok, fails = verdict(final)
    log("\n" + "-" * 56)
    log(f"held-out n          : {final['n']}  (moves {final['n_move']}, "
        f"switches {final['n_switch']})")
    log(f"move-accuracy       : {final['move_acc']:.3f}   chance "
        f"{final['move_chance_legal']:.3f}  const-idx {final['move_const_acc']:.3f}   "
        f"beats-both {'YES' if final['move_beats_chance'] and final['move_beats_const'] else 'NO'}")
    log(f"switch-accuracy     : {final['switch_acc']:.3f}   chance "
        f"{final['switch_chance']:.3f}   const-idx {final['switch_const_acc']:.3f}   "
        f"beats-both {'YES' if final['switch_beats_chance'] and final['switch_beats_const'] else 'NO'}")
    log(f"predicted switch-rate: {final['pred_switch_rate']:.3f}   human "
        f"{final['human_switch_rate']:.3f}   gap {final['switch_rate_gap']:+.3f}")
    # D81: the first-use/repeat leak check (the "grab the new move?" signal).
    _fu = final.get("first_use_move_acc")
    _rp = final.get("repeat_move_acc")
    if _fu is not None and _rp is not None:
        log(f"move first-use/repeat: first {_fu:.3f} (n={final['n_first_use']})   "
            f"repeat {_rp:.3f} (n={final['n_repeat']})   "
            f"ratio {(_fu / _rp if _rp > 0 else float('inf')):.2f}x   "
            f"gap {final['shortcut_gap']:+.3f}")
        log(f"  shortcut gate (D81/D82): FAIL if first-use >= {ABS_FIRST_USE_FLOOR} "
            f"AND first-use > {1.0 + REL_SHORTCUT_MARGIN:.2f}x repeat "
            f"(both splits >= {MIN_SPLIT_SAMPLES})")
    else:
        log(f"move first-use/repeat: n/a (first n={final.get('n_first_use', 0)}, "
            f"repeat n={final.get('n_repeat', 0)} — too few to gate)")
    log(f"held-out top-1      : {final['top1_acc']:.3f}   chance "
        f"{final['overall_chance']:.3f}   const-idx {final['top1_const_acc']:.3f}   "
        f"beats-both {'YES' if final['top1_beats_chance'] and final['top1_beats_const'] else 'NO'}")
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
        "verdict_pass": ok, "stopped": bool(res.get("stopped", False)),
        "elapsed_s": round(time.time() - t0, 1),
    })
    log(f"best checkpoint -> {ckpt_path}  "
        f"(best held top-1 {res['best_top1']:.3f})")

    return {
        "stopped": bool(res.get("stopped", False)),
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
