"""D103 Step 1 — a league bot that LEARNS: a live ``PokeNet`` + its experience
``RolloutBuffer``.

The split (design decision, see OPEN_DESIGN_CHOICES.md):

* ``league.registry.Bot`` stays the **ROSTER RECORD** — name, checkpoint path,
  ELO, W/L — JSON-backed and **torch-free** (the GUI reads the roster without
  paying for torch; the registry module's no-torch contract is unchanged).
* ``LearningBot`` (this module) is the **LIVE object** the league engine will
  drive: it OWNS the net the bot plays with and the buffer the experience
  lands in. One ``LearningBot`` per roster entry.

Seeding (user-approved in D103 — "start a new random net or choose a snapshot
to continue training"):

* ``LearningBot.random(...)``        — a fresh ``make_random_net`` (unseeded =
  a DIFFERENT net per bot — the league's diversity comes from here);
* ``LearningBot.from_snapshot(...)`` — continue training an existing ``.pt``
  checkpoint (BC or self-play format — ``PokeNet.load_file`` handles both).

**Principle: a snapshot is PRISTINE.** A bot seeded from
``scratch/s43_bc_n2000.pt`` keeps its UPDATED weights in its OWN save path —
it never overwrites the source snapshot (or any file another bot shares).

Steps 2–3 (next) collect ``(state, action, reward, logprob, value)`` into
``buffer`` and run the ``net/ppo.py`` update over it; Step 1 is the container
that makes that loop possible, plus the ONE checkpoint load path
(``PokeNet.load_file``) and roster materialization.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from net.ppo import RolloutBuffer, RolloutStep

__all__ = ["LearningBot", "fresh_bot_name", "materialize_roster"]


def fresh_bot_name(kind: str = "league", out_dir: str = "scratch") -> Path:
    """A D99-style unique checkpoint path for a NEW bot: ``{kind}_{id4}.pt``.

    No date (same-day collisions were the D99 problem); a 4-char uuid4 hex id
    makes same-league additions unique and league-friendly. Used by Step 4's
    cull+refresh (injecting fresh random bots) and by GUI "add random bot".

    **Never returns a path that already exists** — Step 4's refill loop can
    mint hundreds of bots over time, and a 4-hex id is only 16 bits: an
    unlucky draw could land on an existing bot's file, whose weights a new
    bot would then clobber on its first save. We redraw until the path is
    free (bounded — a full 65k-slot space never fills up in practice)."""
    out = Path(out_dir)
    for _ in range(64):
        candidate = out / f"{kind}_{uuid.uuid4().hex[:4]}.pt"
        if not candidate.exists():
            return candidate
    raise RuntimeError(
        f"fresh_bot_name: {64} draws all collided under {out} for kind "
        f"{kind!r} — the space is unexpectedly full")


class LearningBot:
    """One league bot that learns: a LIVE ``PokeNet`` + its ``RolloutBuffer``.

    ``ckpt_path`` is this bot's OWN save path (where :meth:`save` writes its
    updated weights) — for a snapshot-seeded bot that is a DIFFERENT file than
    the snapshot it was loaded from.

    The net is shared read-only by the per-battle agents (the engine builds a
    FRESH ``PokeAgent`` per battle for RNG isolation — ``bot.agent(seed)``);
    only a PPO update (Step 3) mutates the weights, and only between battles.
    """

    def __init__(self, name: str, ckpt_path, net: Any, vocabs: Any,
                 device: str = "cpu", buffer_capacity: Optional[int] = None,
                 origin: str = "random") -> None:
        self.name = name
        self.ckpt_path = Path(ckpt_path)
        self.net = net
        self.vocabs = vocabs
        self.device = device
        self.buffer = RolloutBuffer(capacity=buffer_capacity)
        self.origin = origin          # "random" | "snapshot:<path>"
        self.n_updates = 0            # Step 3: incremented per PPO update
        self.n_saved = 0              # incremented per :meth:`save`
        self.last_saved: Optional[str] = None   # ISO timestamp

    # ── factories (the two user-approved seeding paths) ────────────────

    @classmethod
    def random(cls, name: str, ckpt_path, vocabs: Any,
               device: str = "cpu", seed: Optional[int] = None,
               buffer_capacity: Optional[int] = None) -> "LearningBot":
        """A bot with a FRESH random-init net.

        ``seed=None`` (default) = truly random — consecutive calls give
        DIFFERENT nets (the league's diversity, D103 step 4's refill relies
        on this). ``seed=int`` = reproducible (tests)."""
        from net.agent import make_random_net
        net = make_random_net(vocabs, device=device, seed=seed)
        net.eval()
        return cls(name=name, ckpt_path=ckpt_path, net=net, vocabs=vocabs,
                   device=device, buffer_capacity=buffer_capacity,
                   origin="random")

    @classmethod
    def from_snapshot(cls, name: str, snapshot_path, ckpt_path, vocabs: Any,
                      device: str = "cpu",
                      buffer_capacity: Optional[int] = None) -> "LearningBot":
        """A bot continuing training from an existing checkpoint.

        ``snapshot_path`` — the ``.pt`` to LOAD (BC or self-play format;
        :meth:`PokeNet.load_file` verifies vocab + layout). ``ckpt_path`` —
        this bot's OWN save path for its UPDATED weights. The snapshot is
        never written (pristine-snapshot principle)."""
        from net.model import PokeNet
        snapshot_path = Path(snapshot_path)
        if not snapshot_path.is_file():
            raise FileNotFoundError(
                f"snapshot checkpoint not found: {snapshot_path} "
                f"(bot {name!r}) — add the correct path to the roster")
        net = PokeNet.load_file(snapshot_path, vocabs, device=device)
        net.eval()
        return cls(name=name, ckpt_path=ckpt_path, net=net, vocabs=vocabs,
                   device=device, buffer_capacity=buffer_capacity,
                   origin=f"snapshot:{snapshot_path}")

    # ── experience (Step 2 will fill the buffer; Step 3 will drain it) ──

    def add_step(self, step: RolloutStep) -> None:
        """Record one decision (a PPO transition) in this bot's buffer."""
        self.buffer.append(step)

    def clear_buffer(self) -> None:
        """Drop all buffered experience (the Step-3 contract after an update)."""
        self.buffer.clear()

    @property
    def n_steps(self) -> int:
        return len(self.buffer)

    # ── weights ────────────────────────────────────────────────────────

    def save(self, extra: Optional[Dict[str, Any]] = None) -> str:
        """Write the CURRENT weights to ``self.ckpt_path`` (BC format — the
        plain ``state_for_save`` payload, loadable by every path) + an
        optional JSON meta sidecar. Returns the path."""
        import torch
        self.ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_for_save(), str(self.ckpt_path))
        meta = {"name": self.name, "origin": self.origin,
                "device": self.device, "n_updates": self.n_updates,
                "n_steps_buffered": len(self.buffer),
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        if extra:
            meta.update(extra)
        with open(str(self.ckpt_path) + ".meta.json", "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        self.n_saved += 1
        self.last_saved = meta["saved_at"]
        return str(self.ckpt_path)

    def agent(self, seed: Optional[int] = None) -> Any:
        """A FRESH ``PokeAgent`` over the CURRENT weights (per-battle RNG
        isolation — the engine's contract: one agent per battle, fresh
        generator, shared net read-only)."""
        from net.agent import PokeAgent
        return PokeAgent(self.net, seed=seed, device=self.device)

    # ── roster materialization ─────────────────────────────────────────

    @classmethod
    def from_roster_entry(cls, bot: Any, vocabs: Any, device: str = "cpu",
                          buffer_capacity: Optional[int] = None) -> "LearningBot":
        """A roster RECORD (``registry.Bot``) -> a live bot.

        The roster entry's ``ckpt_path`` is BOTH the snapshot to load and the
        bot's save path — the league contract is ONE FILE PER BOT (a shared
        file would be clobbered by the first bot's save; the eval-only engine
        supported sharing because it never wrote)."""
        return cls.from_snapshot(bot.name, bot.ckpt_path, bot.ckpt_path,
                                 vocabs, device=device,
                                 buffer_capacity=buffer_capacity)


def materialize_roster(registry: Any, vocabs: Any, device: str = "cpu",
                       buffer_capacity: Optional[int] = None) -> Dict[str, LearningBot]:
    """The whole roster (``BotRegistry``) -> live :class:`LearningBot`s.

    ``{name: LearningBot}`` — the Step-2 entry point (the engine will drive
    these instead of its own net cache). Raises on the FIRST missing
    checkpoint with the bot name named (loud, before any battle).

    **ONE FILE PER BOT is enforced here, loudly:** two roster entries
    pointing at the same checkpoint would let the first bot to save
    clobber the other's weights (while the other keeps playing stale
    in-memory weights) — a silent data-loss footgun, so it's a hard
    error instead."""
    bots: Dict[str, LearningBot] = {}
    owner_by_file: Dict[str, str] = {}
    for bot in registry.all():
        resolved = str(Path(bot.ckpt_path).resolve())
        if resolved in owner_by_file:
            raise ValueError(
                f"roster error: {owner_by_file[resolved]!r} and {bot.name!r} "
                f"both point at {resolved} — the league contract is ONE FILE "
                f"PER BOT (the first bot to save would clobber the other's "
                f"weights). Give each bot its own checkpoint file.")
        owner_by_file[resolved] = bot.name
        bots[bot.name] = LearningBot.from_roster_entry(
            bot, vocabs, device=device, buffer_capacity=buffer_capacity)
    return bots
