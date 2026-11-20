"""D103 Step 1 — ``league.learner.LearningBot`` (the bot that learns).

FAST + offline (no fork, no pool, no GPU): pins the container contract the
Steps 2–3 loop runs on —
  * the two seeding paths (random / snapshot) and their invariants
    (diversity, reproducibility under a seed, pristine snapshots);
  * the buffer plumbing (append / capacity / clear);
  * save → load round-trips through THE ONE load path (``PokeNet.load_file``,
    both checkpoint formats);
  * roster materialization (registry records → live bots).

Run:  python -m pytest league/tests/test_learner.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.learner import LearningBot, fresh_bot_name, materialize_roster
from net.model import PokeNet
from net.ppo import RolloutStep

from net.tests.test_s41 import make_ctx, make_vocabs


def _vocabs():
    return make_vocabs(make_ctx())


def _param_sum(net) -> float:
    """One scalar fingerprint of the weights (enough to tell nets apart)."""
    return float(sum(p.detach().float().sum().item() for p in net.parameters()))


def _step() -> RolloutStep:
    return RolloutStep(obs={}, action_index=0, old_logprob=-1.0,
                       old_value=0.5, reward=0.0, done=True)


# ── seeding: random ─────────────────────────────────────────────────────────

def test_random_bot_has_live_net_and_empty_buffer():
    v = _vocabs()
    bot = LearningBot.random("alpha", "scratch/league_alpha.pt", v)
    assert _param_sum(bot.net) != 0.0
    assert len(bot.buffer) == 0
    assert bot.origin == "random"
    assert bot.ckpt_path == Path("scratch/league_alpha.pt")
    assert bot.n_updates == 0 and bot.n_saved == 0


def test_random_bots_are_diverse():
    """Unseeded random bots must give DIFFERENT nets — the league's diversity
    (D103 step 4's cull+refill) depends on this."""
    v = _vocabs()
    a = LearningBot.random("a", "scratch/a.pt", v)
    b = LearningBot.random("b", "scratch/b.pt", v)
    assert _param_sum(a.net) != _param_sum(b.net)


def test_random_bot_seed_is_reproducible():
    v = _vocabs()
    a = LearningBot.random("a", "scratch/a.pt", v, seed=7)
    b = LearningBot.random("b", "scratch/b.pt", v, seed=7)
    assert _param_sum(a.net) == _param_sum(b.net)


# ── the ONE load path: both formats ────────────────────────────────────────

def test_load_file_both_formats_match_original(tmp_path):
    v = _vocabs()
    net = PokeNet(v)
    s0 = _param_sum(net)

    # BC format: the raw state_for_save payload
    bc_path = tmp_path / "bc.pt"
    torch.save(net.state_for_save(), str(bc_path))
    # self-play format: the {"net": …, "round": …} wrapper
    sp_path = tmp_path / "selfplay.pt"
    torch.save({"net": net.state_for_save(), "round": 3}, str(sp_path))

    for p in (bc_path, sp_path):
        loaded = PokeNet.load_file(p, v)
        assert _param_sum(loaded) == s0, f"weights drifted loading {p.name}"

    # unwrap_payload: the two-format logic, shared with gui.trainer_main
    sp_payload = torch.load(str(sp_path), map_location="cpu", weights_only=True)
    assert "round" in sp_payload                      # outer payload kept
    assert PokeNet.unwrap_payload(sp_payload) is sp_payload["net"]
    x_path = tmp_path / "x.pt"
    torch.save(net.state_for_save(), str(x_path))
    bc_payload = torch.load(str(x_path), map_location="cpu", weights_only=True)
    assert PokeNet.unwrap_payload(bc_payload) is bc_payload  # BC passes through


# ── seeding: snapshot (principle: the snapshot is pristine) ────────────────

def test_from_snapshot_loads_the_source_weights(tmp_path):
    v = _vocabs()
    src = PokeNet(v)
    s0 = _param_sum(src)
    snap = tmp_path / "snapshot.pt"
    torch.save(src.state_for_save(), str(snap))

    bot = LearningBot.from_snapshot("champ", snap, tmp_path / "champ_out.pt", v)
    assert _param_sum(bot.net) == s0
    assert bot.origin.startswith("snapshot:")
    assert bot.ckpt_path != Path(snap)          # its OWN save path


def test_snapshot_stays_pristine_after_bot_save(tmp_path):
    v = _vocabs()
    src = PokeNet(v)
    snap = tmp_path / "snapshot.pt"
    torch.save(src.state_for_save(), str(snap))
    before = snap.read_bytes()

    bot = LearningBot.from_snapshot("champ", snap, tmp_path / "champ_out.pt", v)
    bot.save({"elo": 1600})

    assert snap.read_bytes() == before, "the source snapshot must never be written"
    assert (tmp_path / "champ_out.pt").is_file()
    # and the saved bot is loadable with the same weights
    reloaded = PokeNet.load_file(tmp_path / "champ_out.pt", v)
    assert _param_sum(reloaded) == _param_sum(bot.net)


def test_save_writes_meta_sidecar(tmp_path):
    v = _vocabs()
    bot = LearningBot.random("meta_bot", tmp_path / "m.pt", v)
    bot.add_step(_step())
    bot.save({"elo": 1550})
    meta = (tmp_path / "m.pt.meta.json").read_text(encoding="utf-8")
    assert '"meta_bot"' in meta and '"elo": 1550' in meta
    assert '"n_steps_buffered": 1' in meta
    assert bot.n_saved == 1 and bot.last_saved is not None


def test_from_snapshot_missing_file_raises_loudly(tmp_path):
    v = _vocabs()
    try:
        LearningBot.from_snapshot("ghost", tmp_path / "nope.pt",
                                  tmp_path / "out.pt", v)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "ghost" in str(exc)


# ── buffer plumbing (Step 2 will fill this) ─────────────────────────────────

def test_add_step_grows_buffer_and_capacity_caps():
    v = _vocabs()
    bot = LearningBot.random("buf", "scratch/buf.pt", v, buffer_capacity=3)
    for _ in range(5):
        bot.add_step(_step())
    assert bot.n_steps == 3, "the capacity cap must hold (RolloutBuffer semantics)"
    bot.clear_buffer()
    assert bot.n_steps == 0


# ── the per-battle agent ────────────────────────────────────────────────────

def test_agent_wraps_the_current_net():
    v = _vocabs()
    bot = LearningBot.random("ag", "scratch/ag.pt", v)
    ag = bot.agent(seed=123)
    assert ag.net is bot.net, "the agent must read THIS bot's live weights"


# ── naming + roster materialization ─────────────────────────────────────────

def test_fresh_bot_names_are_unique_and_wellformed():
    a, b = fresh_bot_name(), fresh_bot_name()
    assert a != b
    assert a.suffix == ".pt" and a.stem.startswith("league_")
    assert len(a.stem.split("_")[-1]) == 4


def test_fresh_bot_name_skips_existing_files(tmp_path, monkeypatch):
    """A 4-hex id is only 16 bits: over time (Step 4's refill loop) an unlucky
    draw CAN land on an existing bot's file. fresh_bot_name must redraw until
    the path is free — a silent overwrite of another bot's weights is the
    footgun this pins. Deterministic: the first draw collides, the second
    does not."""
    import league.learner as learner
    draws = iter([type("U", (), {"hex": "a1b2"})(),
                  type("U", (), {"hex": "c3d4"})()])
    monkeypatch.setattr(learner.uuid, "uuid4", lambda: next(draws))
    (tmp_path / "league_a1b2.pt").touch()
    got = fresh_bot_name(out_dir=str(tmp_path))
    assert got == tmp_path / "league_c3d4.pt"   # redrew past the collision
    assert (tmp_path / "league_a1b2.pt").is_file()   # existing file untouched


def test_materialize_roster_maps_every_bot(tmp_path):
    from league.registry import BotRegistry
    v = _vocabs()

    # two roster bots with two distinct checkpoints
    reg = BotRegistry(path=tmp_path / "bots.json")
    for name in ("alpha", "beta"):
        ckpt = tmp_path / f"{name}.pt"
        net = PokeNet(v)
        torch.save(net.state_for_save(), str(ckpt))
        reg.add(name, str(ckpt))

    bots = materialize_roster(reg, v)
    assert set(bots) == {"alpha", "beta"}
    for name, bot in bots.items():
        src = PokeNet.load_file(tmp_path / f"{name}.pt", v)
        assert _param_sum(bot.net) == _param_sum(src), name
        assert bot.ckpt_path == Path(str(tmp_path / f"{name}.pt"))


def test_materialize_roster_rejects_shared_checkpoint_file(tmp_path):
    """Two roster bots on ONE file = silent clobber (first bot to save
    overwrites the other's weights; the other plays stale in-memory weights).
    Must be a LOUD error naming both bots, before any battle."""
    from league.registry import BotRegistry
    v = _vocabs()
    shared = tmp_path / "shared.pt"
    net = PokeNet(v)
    torch.save(net.state_for_save(), str(shared))

    reg = BotRegistry(path=tmp_path / "bots.json")
    reg.add("alpha", str(shared))
    reg.add("beta", str(shared))

    try:
        materialize_roster(reg, v)
        raise AssertionError("expected ValueError for a shared checkpoint file")
    except ValueError as exc:
        assert "alpha" in str(exc) and "beta" in str(exc)
        assert "ONE FILE PER BOT" in str(exc)


def test_materialize_roster_missing_checkpoint_names_the_bot(tmp_path):
    from league.registry import BotRegistry
    v = _vocabs()
    reg = BotRegistry(path=tmp_path / "bots.json")
    reg.add("ghost", str(tmp_path / "missing.pt"))
    try:
        materialize_roster(reg, v)
        raise AssertionError("expected FileNotFoundError")
    except FileNotFoundError as exc:
        assert "ghost" in str(exc)
