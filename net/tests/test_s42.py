"""S4.2 — network core + heads gate tests (PLAN.md §3c S4.2, D60).

The S4.2 gate (user spec):
  (1) **illegal actions = exactly 0 probability** — mask test over 500
      sampled states (real-corpus decisions);
  (2) **permutation invariance** — a bench slot swap leaves the logits/value
      unchanged (the per-side set-pool is a set function);
  (3) **overfit-32** — memorize a 32-battle subset -> train loss < 0.05
      (the "never learns" disaster guard);
  (4) **save/load round-trip** — identical logits; a vocab/layout mismatch
      on load RAISES (no silent drift);
  (5) **no NaN/inf grads** on a real-corpus batch;
  (6) baseline green — the rest of the suite, run separately.

Cost model: tests (1), (3), (5) share ONE module-scoped corpus fixture that
replays 32 battles through the belief/calc stack. Cold node spawns make this
~10-15 min ONCE per pytest session (budget minutes, not seconds — the same
cost class as the D68 200-battle gate). Tests (2), (4) and the synthetic
edge test are pure torch and run in seconds.

Interpretation notes (recorded, to confirm with the user):
  * "overfit-32" = a subset of 32 BATTLES (the fixture replays 32); each
    battle contributes up to 2 (state, action) samples to the memorization
    set, so the loss target spans every battle in the subset.
  * "500 sampled states" = 500 distinct (sample, supervised-timestep) pairs
    from the replayed corpus (the states the net is actually trained on).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "bot" / "data" / "replays" / "gen9ou.jsonl"
HAS_FIXTURE = FIXTURE.exists()
needs_fixture = pytest.mark.skipif(
    not HAS_FIXTURE, reason="gen9ou.jsonl human-ladder fixture missing")

from net import (
    LAYOUT,
    Vocabs,
    build_vocabs,
    encode_context,
    PokeNet,
    bc_loss,
    overfit_train,
    build_samples,
    samples_to_batch,
    batch_to_torch,
)
# the S4.1 gate test's context builders (a realistic gen9-ou turn-start ctx +
# DB-free vocabs built from the ids present in it) — reused, not duplicated.
from net.tests.test_s41 import make_ctx, make_vocabs
from corpus.pull import load_local
from corpus.trajectories import replay_many

# ── shared corpus fixture (the expensive part, built ONCE per session) ────
N_BATTLES = 32          # the overfit-32 subset size (the gate's unit is battles)
MAX_DECISIONS_PER_BATTLE = 40   # the corpus's legal-decision cap


@dataclass
class CorpusBundle:
    vocabs: Vocabs
    n_replayed: int
    per_battle: list          # List[List[Sample]] — one inner list per battle


@pytest.fixture(scope="module")
def corpus() -> CorpusBundle:
    torch.manual_seed(0)
    v = build_vocabs()
    battles = load_local(str(FIXTURE), N_BATTLES)
    traces = replay_many(battles)
    ok = [t for t in traces if t.skipped is None]
    assert len(ok) >= 8, f"only {len(ok)}/{len(traces)} battles replayed"
    per_battle = [build_samples(tr, v, max_decisions=MAX_DECISIONS_PER_BATTLE)
                  for tr in ok]
    return CorpusBundle(vocabs=v, n_replayed=len(ok), per_battle=per_battle)


def _all_samples(corpus: CorpusBundle) -> list:
    out = []
    for per in corpus.per_battle:
        out.extend(per)
    return out


# ── synthetic helpers (fast, DB-free) ─────────────────────────────────────
def random_batch(vocabs: Vocabs, B: int, T: int, seed: int = 0) -> dict:
    """A random-but-VALID model batch (every id index inside its vocab, each
    action row has >=1 legal position). For round-trip / edge-case tests."""
    rng = np.random.default_rng(seed)
    batch = {}
    for fld in LAYOUT:
        if fld.name == "action_mask":
            A = fld.shape[0]   # MAX_ACTIONS
            arr = np.zeros((B, T) + tuple(fld.shape), dtype=np.float32)
            for i in range(B):
                for t in range(T):
                    k = int(rng.integers(1, A + 1))
                    arr[i, t, rng.choice(A, k, replace=False)] = 1.0
            batch[fld.name] = arr
            continue
        if fld.dtype == "float32":
            batch[fld.name] = rng.uniform(0.0, 1.0,
                                          size=(B, T) + tuple(fld.shape)
                                          ).astype(np.float32)
            continue
        # int64 index channels — bound each to its OWN vocab's size
        name = fld.name
        if "species" in name:
            n = len(vocabs.species)
        elif "move" in name:
            n = len(vocabs.move)
        elif "ability" in name:
            n = len(vocabs.ability)
        elif "item" in name:
            n = len(vocabs.item)
        elif name == "global_weather":
            n = len(vocabs.weather)
        elif name == "global_terrain":
            n = len(vocabs.terrain)
        elif name == "action_kind":
            n = 8
        else:
            n = 1
        # n is the embedding row count (vocab size incl. <pad>, always >= 1);
        # rng.integers' upper bound is exclusive, so 0..n-1 is exactly in range.
        batch[fld.name] = rng.integers(0, n,
                                      size=(B, T) + tuple(fld.shape)
                                      ).astype(np.int64)
    batch["battle_mask"] = np.ones((B, T), dtype=np.float32)
    return batch


def _stack_contexts(encs: list) -> dict:
    """A list of encoded contexts (same vocabs) -> a T=1 batch (B=len)."""
    B = len(encs)
    batch = {}
    for fld in LAYOUT:
        dt = {"float32": np.float32, "int64": np.int64}[fld.dtype]
        arr = np.zeros((B, 1) + tuple(fld.shape), dtype=dt)
        for i, e in enumerate(encs):
            arr[i, 0] = e[fld.name]
        batch[fld.name] = arr
    batch["battle_mask"] = np.ones((B, 1), dtype=np.float32)
    return batch


# ── (2) permutation invariance: bench slot swap -> same logits ────────────
def test_permutation_invariance_bench_swap():
    ctx = make_ctx()
    v = make_vocabs(ctx)
    enc1 = encode_context(ctx, v)
    ctx2 = copy.deepcopy(ctx)
    # swap the two BENCH own slots (p1a stays active) — pure row permutation
    ctx2["own"]["p1b"], ctx2["own"]["p1c"] = ctx["own"]["p1c"], ctx["own"]["p1b"]
    enc2 = encode_context(ctx2, v)
    batch = batch_to_torch(_stack_contexts([enc1, enc2]))

    torch.manual_seed(0)
    net = PokeNet(v)
    net.eval()
    with torch.no_grad():
        out = net(batch)
    l1, l2 = out["logits"][0, 0], out["logits"][1, 0]
    v1, v2 = out["value"][0, 0], out["value"][1, 0]
    # float32 sum-order noise is the only legitimate difference (the set-pool
    # is a set function); the net must not amplify it into a decision change.
    assert torch.allclose(l1, l2, atol=1e-3, rtol=1e-3), \
        f"bench swap changed logits: max delta {(l1 - l2).abs().max()}"
    assert torch.allclose(v1, v2, atol=1e-4, rtol=1e-3), \
        f"bench swap changed value: {float(v1)} vs {float(v2)}"


# ── (4) save/load round-trip = identical logits (+ loud mismatch) ─────────
def test_save_load_roundtrip_identical_logits():
    v = make_vocabs(make_ctx())          # DB-free: the net only needs the ids
    torch.manual_seed(123)
    net = PokeNet(v)
    ctx_b = make_ctx()
    ctx_b["turn"] = 9                      # a second, different state
    batch = batch_to_torch(_stack_contexts([
        encode_context(make_ctx(), v), encode_context(ctx_b, v)]))

    net.eval()
    with torch.no_grad():
        out1 = net(batch)

    payload = net.state_for_save()
    net2 = PokeNet.load(payload, make_vocabs(make_ctx()))  # fresh, same ids
    net2.eval()
    with torch.no_grad():
        out2 = net2(batch)

    assert torch.equal(out1["logits"], out2["logits"])
    assert torch.equal(out1["value"], out2["value"])
    assert torch.equal(out1["prob"], out2["prob"])


def test_load_with_wrong_vocab_raises():
    v = make_vocabs(make_ctx())
    torch.manual_seed(0)
    net = PokeNet(v)
    payload = net.state_for_save()
    # a DIFFERENT vocab (one extra species row) must be rejected loudly
    v_bad = make_vocabs(make_ctx())
    v_bad.species.add("someotherspecies")
    with pytest.raises(ValueError, match="vocab size mismatch"):
        PokeNet.load(payload, v_bad)


# ── D102 device regression: load() must honour the device arg ─────────────
# The league pass-fest bug: torch.load(map_location="cuda") + load_state_dict
# into a fresh CPU net SILENTLY left the net on CPU while a CUDA agent fed it
# CUDA tensors. A fresh net is always built on CPU, so this test proves the
# device is applied AFTER load_state_dict — and that omitting it stays CPU.
def test_load_default_device_is_cpu():
    v = make_vocabs(make_ctx())
    torch.manual_seed(0)
    net = PokeNet(v)
    path = Path(__file__).with_name("_devload_test.pt")
    torch.save(net.state_for_save(), path)
    try:
        reloaded = PokeNet.load(torch.load(path, weights_only=True),
                                make_vocabs(make_ctx()))
        assert all(p.device.type == "cpu" for p in reloaded.parameters())
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="no CUDA device (CPU-only box)")
def test_load_with_device_moves_to_cuda():
    v = make_vocabs(make_ctx())
    torch.manual_seed(0)
    net = PokeNet(v)
    path = Path(__file__).with_name("_devload_test.pt")
    torch.save(net.state_for_save(), path)
    try:
        payload = torch.load(path, map_location="cuda", weights_only=True)
        reloaded = PokeNet.load(payload, make_vocabs(make_ctx()),
                                device="cuda")
        assert all(p.device.type == "cuda" for p in reloaded.parameters())
    finally:
        path.unlink(missing_ok=True)


# ── synthetic edge: an all-illegal action row must not produce NaN ────────
def test_all_masked_state_fallback_finite():
    v = make_vocabs(make_ctx())
    torch.manual_seed(0)
    net = PokeNet(v)
    net.eval()
    batch = batch_to_torch(random_batch(v, B=2, T=3, seed=7))
    batch["action_mask"][:] = 0.0          # every position illegal
    with torch.no_grad():
        out = net(batch)
    assert torch.isfinite(out["logits"]).all()
    assert torch.isfinite(out["value"]).all()
    assert torch.isfinite(out["prob"]).all()
    assert torch.allclose(out["prob"].sum(dim=-1),
                          torch.ones_like(out["prob"].sum(dim=-1)), atol=1e-5)


# ── (1) illegal actions = EXACTLY 0 probability, over 500 sampled states ──
@needs_fixture
def test_illegal_actions_exactly_zero_probability(corpus: CorpusBundle):
    samples = _all_samples(corpus)[:500]
    assert len(samples) >= 500, f"corpus too small for the 500-state gate: {len(samples)}"
    torch.manual_seed(0)
    net = PokeNet(corpus.vocabs)
    net.eval()
    checked = 0
    with torch.no_grad():
        for i in range(0, len(samples), 64):
            chunk = samples[i:i + 64]
            info = samples_to_batch(chunk)
            b = batch_to_torch(info["batch"])
            ts = info["ts"]
            out = net(b)
            for row in range(len(chunk)):
                t = int(ts[row])
                mask = b["action_mask"][row, t]
                prob = out["prob"][row, t]
                n_legal = int(mask.sum())
                assert n_legal >= 1, "every sampled state has a legal action"
                illegal = (mask == 0)
                if bool(illegal.any()):
                    mx = float(prob[illegal].max())
                    assert mx == 0.0, f"illegal action got prob {mx} (must be exactly 0)"
                s = float(prob.sum())
                assert abs(s - 1.0) < 1e-5, f"prob row sums to {s}, not 1"
                checked += 1
    assert checked >= 500


# ── (3) overfit-32: the "never learns" disaster guard ─────────────────────
@needs_fixture
def test_overfit32_train_loss_below_005(corpus: CorpusBundle):
    # up to 2 samples per battle -> every battle in the 32-battle subset
    # contributes to the memorization target.
    samples = [s for per in corpus.per_battle for s in per[:2]]
    assert len(samples) >= 16, f"only {len(samples)} samples to memorize"

    batches, targets, tss = [], [], []
    for i in range(0, len(samples), 16):
        chunk = samples[i:i + 16]
        info = samples_to_batch(chunk)
        batches.append(batch_to_torch(info["batch"]))
        targets.append(torch.from_numpy(np.ascontiguousarray(info["targets"])))
        tss.append(np.ascontiguousarray(info["ts"]))

    torch.manual_seed(0)
    net = PokeNet(corpus.vocabs)
    # budget: ~600 Adam steps at lr=1e-2. Calibrated 2026-11-17: a
    # deterministic input->action rule on this net reaches loss <0.05 in
    # <30 steps (scratch/probe_s42_overfit_cal.py), so this is a wide margin
    # for structured (real-battle) data — if it STILL fails, the net cannot
    # learn and the guard has done its job.
    final = overfit_train(net, batches, targets, tss, epochs=200, lr=1e-2)
    assert final < 0.05, (
        f"overfit-32 failed: final train loss {final:.4f} >= 0.05 — "
        "the net cannot memorize a small corpus; something is structurally "
        "wrong (the 'never learns' guard)")


# ── (3b) STATE-CONDITIONING guard (fast, deterministic, no replay) ──────
# The overfit-32 gate above only ever used size-1 / constant targets, so it
# could NOT catch a net that collapses to a single DEFAULT action (the
# "memorized default = switch" failure the user hit in the old system). This
# test pins the VARYING-target form: two distinct states, the SAME action list,
# DIFFERENT targets -> the net must be able to pick a different action per state.
def test_net_state_conditions_not_constant_action():
    from net.batch import Sample
    moves4 = ["earthquake", "icebeam", "roar", "toxic"]
    ctx = make_ctx(); v = make_vocabs(ctx)
    def enc(species, hp, turn):
        c = copy.deepcopy(ctx)
        c["own"]["p1a"]["name"] = species; c["own"]["p1a"]["hp_frac"] = hp
        c["own"]["p1a"]["moves"] = moves4; c["turn"] = turn
        return encode_context(c, v, [{"kind": "move", "move": m} for m in moves4])
    samples = [Sample("a", 1, "move", 0, moves4, [enc("garchomp", 0.9, 1)]),
               Sample("b", 2, "move", 2, moves4, [enc("blastoise", 0.4, 2)])]
    torch.manual_seed(0)
    net = PokeNet(v)
    sb = samples_to_batch(samples)
    b = batch_to_torch(sb["batch"])
    targets = torch.as_tensor(sb["targets"], dtype=torch.int64)
    ts = torch.as_tensor(sb["ts"], dtype=torch.int64)
    final = overfit_train(net, [b], [targets], [ts], epochs=400, lr=1e-2)
    assert final < 0.05, (
        f"the net cannot state-condition (final loss {final:.4f} ~ ln(k)) — it "
        "collapses to a constant/default action policy (the 'never learns' "
        "disaster). Check the GRU residual skip in PokeNet.forward.")


# ── (5) no NaN/inf grads on a real-corpus batch ───────────────────────────
@needs_fixture
def test_no_nan_inf_grads_real_corpus_batch(corpus: CorpusBundle):
    samples = []
    for per in corpus.per_battle[:4]:
        samples.extend(per[:2])
    samples = samples[:8]
    assert len(samples) >= 4

    torch.manual_seed(0)
    net = PokeNet(corpus.vocabs)
    net.train()
    total = None
    for s in samples:
        info = samples_to_batch([s])
        b = batch_to_torch(info["batch"])
        out = net(b)
        loss = bc_loss(out["logits"], b["action_mask"],
                       torch.tensor(s.target, dtype=torch.long),
                       len(s.seq) - 1)
        total = loss if total is None else total + loss
    assert torch.isfinite(total).all()
    total.backward()
    bad = [name for name, p in net.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    assert not bad, f"non-finite grads in: {bad}"
    assert any(p.grad is not None for p in net.parameters()), "no grads at all?"
