"""Hermetic tests for net/bc_warm.py — the (c) Arm B BC trainer.

Pins the CONTRACT between the two halves of Arm B:

  * ``tools.bc_collect.save_bc_dataset`` -> ``net.bc_warm.load_bc_dataset``
    (the npz+meta the collector writes is exactly what the trainer reads —
    a drift between them is a loud failure here)
  * the net batch shape (the ``steps_to_batch`` T=1 contract)
  * the "never learns" guard: a CONSTANT teacher action on IDENTICAL
    states must be imitable to ~0 NLL / 100% accuracy (same discipline as
    the S4.2 overfit guard, for the BC path)
  * checkpoint save/load through ``PokeNet.load_file`` (vocab +
    layout-checksum verified)

The PPOTrainer Arm B ``ref_kl`` behaviour is pinned in
``net/tests/test_ppo.py``. No server, no GPU, fast (small nets, CPU).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from net.bc_warm import (bc_batch, load_bc_dataset, save_bc_checkpoint,
                         train_bc)
from net.agent import make_random_net
from net.model import PokeNet
from net.registry import LAYOUT
from net.tests.test_ppo import make_steps
from net.tests.test_s41 import make_ctx, make_vocabs
from tools.bc_collect import save_bc_dataset


# ── helpers ────────────────────────────────────────────────────────────────────
# NOTE: the int64 LAYOUT fields are VOCAB INDICES — random values are invalid
# (out-of-range embedding indices crash the forward). These tests therefore
# build REAL observations via the shared encoder (make_steps -> encode_context)
# instead of random arrays — the same discipline test_ppo.py uses.
def _steps(n: int, seed: int = 3):
    ctx = make_ctx()
    vocabs = make_vocabs(ctx)
    return ctx, vocabs, make_steps(vocabs, ctx, n=n, seed=seed)


def _fields_from_steps(steps) -> dict:
    """(N, *shape) per LAYOUT field + the steps' (legal) action indices."""
    fields = {f.name: np.stack([np.ascontiguousarray(s.obs[f.name])
                                for s in steps]) for f in LAYOUT}
    ai = np.array([s.action_index for s in steps], dtype=np.int64)
    return fields, ai


def _rec(action_index: int, seed: int) -> dict:
    # a valid single obs (encoder-built, indices in-vocab); the first two
    # actions of the encoder's legal list are the legal ones
    _, _, steps = _steps(1, seed=seed)
    obs = {fld.name: np.ascontiguousarray(steps[0].obs[fld.name])
           for fld in LAYOUT}
    legal = int((obs["action_mask"] > 0).sum())
    idx = min(action_index, legal - 1) if action_index >= 0 else -1
    return {"obs": obs, "action_index": idx,
            "action": {"kind": "move", "move": "x"}, "sig": f"s{seed}",
            "n_legal": legal}


# ── batch shape contract ──────────────────────────────────────────────────────
def test_bc_batch_shape_contract():
    n = 7
    ctx, vocabs, steps = _steps(n, seed=3)
    fields, ai = _fields_from_steps(steps)
    batch, target = bc_batch(fields, ai, 0, n, "cpu")
    for f in LAYOUT:
        assert batch[f.name].shape == (n, 1, *f.shape), f.name
    assert batch["battle_mask"].shape == (n, 1)
    assert bool((batch["battle_mask"] == 1).all())
    assert target.shape == (n,) and target.dtype == torch.int64
    # slicing: a sub-batch equals the corresponding rows of the full batch
    batch2, target2 = bc_batch(fields, ai, 2, 3, "cpu")
    for f in LAYOUT:
        assert torch.equal(batch2[f.name], batch[f.name][2:5]), f.name
    assert torch.equal(target2, target[2:5])


# ── the "never learns" guard ──────────────────────────────────────────────────
def test_train_bc_imitates_constant_teacher():
    """A CONSTANT teacher action on IDENTICAL states must be driven to ~0 NLL
    / 100% accuracy — if this fails, the BC path is structurally broken
    (the net cannot even imitate a constant, so it cannot imitate the
    strategy bot)."""
    ctx, vocabs, steps = _steps(1, seed=1)
    net = make_random_net(vocabs, "cpu")
    obs0 = {f.name: np.ascontiguousarray(steps[0].obs[f.name]) for f in LAYOUT}
    n = 64
    fields = {name: np.stack([obs0[name] for _ in range(n)]) for name in obs0}
    # the teacher always plays the SAME legal action (step 0's own choice)
    ai = np.full(n, int(steps[0].action_index), dtype=np.int64)
    ds = {"fields": fields, "action_index": ai,
          "battle_id": np.zeros(n, dtype=np.int32), "meta": {}}
    res = train_bc(net, ds, epochs=30, batch_size=32, lr=3e-3, seed=7)
    assert res["final_val_accuracy"] == pytest.approx(1.0)
    assert res["final_val_nll"] < 0.1
    assert res["history"]["train_nll"][-1] < res["history"]["train_nll"][0]


def test_train_bc_deterministic_for_seed():
    n = 32
    ctx, vocabs, steps = _steps(n, seed=1)
    fields, ai = _fields_from_steps(steps)
    ds = {"fields": fields, "action_index": ai,
          "battle_id": np.zeros(n, dtype=np.int32), "meta": {}}

    def once(seed):
        torch.manual_seed(5)
        net = make_random_net(vocabs, "cpu")
        return train_bc(net, ds, epochs=3, batch_size=16, lr=1e-3, seed=seed)

    a, b = once(9), once(9)
    assert a["history"] == b["history"]
    c = once(10)
    assert c["history"] != a["history"]
    # small-n edge: n < 10 -> the whole set is the val split
    assert a["val_size"] == min(256, max(1, n // 10))


# ── the collector -> trainer contract ────────────────────────────────────────
def test_load_roundtrip_from_bc_collect(tmp_path):
    # indices 0/1 only: EVERY encoded obs has >= 2 legal actions, so these
    # are always legal targets (the bc_loss premise)
    per_battle = [
        {"battle_id": 0,
         "result": {"n_decisions": 3, "n_kept": 3, "n_dropped": 0,
                    "kept": [_rec(0, 1), _rec(1, 2), _rec(0, 3)]}},
        {"battle_id": 1,
         "result": {"n_decisions": 2, "n_kept": 2, "n_dropped": 0,
                    "kept": [_rec(1, 4), _rec(0, 5)]}},
    ]
    out = str(tmp_path / "bc")
    save_bc_dataset(out, per_battle, {"fmt": "gen9ou", "seed": 7})
    ds = load_bc_dataset(out)
    n = 5
    for f in LAYOUT:
        assert ds["fields"][f.name].shape == (n, *f.shape), f.name
    assert list(ds["action_index"]) == [0, 1, 0, 1, 0]
    assert list(ds["battle_id"]) == [0, 0, 0, 1, 1]
    assert ds["meta"]["fmt"] == "gen9ou"
    assert ds["meta"]["seed"] == 7


def test_load_rejects_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_bc_dataset(str(tmp_path / "does-not-exist"))


def test_load_rejects_teacher_anomaly_rows(tmp_path):
    # a dataset containing action_index < 0 (a teacher anomaly the
    # collection records but never re-maps) must be rejected LOUDLY —
    # never silently trained on
    per_battle = [
        {"battle_id": 0,
         "result": {"n_decisions": 2, "n_kept": 2, "n_dropped": 0,
                    "kept": [_rec(0, 1), _rec(-1, 2)]}},
    ]
    out = str(tmp_path / "bc")
    save_bc_dataset(out, per_battle, {"fmt": "gen9ou"})
    with pytest.raises(ValueError, match="action_index < 0"):
        load_bc_dataset(out)


# ── checkpoint round-trip (the Arm B start point) ─────────────────────────────
def test_checkpoint_roundtrip(tmp_path):
    ctx = make_ctx()
    vocabs = make_vocabs(ctx)
    net = make_random_net(vocabs, "cpu")
    path = str(tmp_path / "bc_init.pt")
    save_bc_checkpoint(net, path)
    loaded = PokeNet.load_file(path, vocabs, device="cpu")
    sd0, sd1 = net.state_dict(), loaded.state_dict()
    assert set(sd0) == set(sd1)
    for k in sd0:
        assert torch.equal(sd0[k], sd1[k]), k
