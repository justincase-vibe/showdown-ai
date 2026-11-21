"""Option A / A2 — the fitness/evaluation harness (the "core").

``evaluate`` plays a **bot** (a checkpoint, or an agent factory) against a
**ladder** of references for ``n_battles`` battles each and aggregates
wins/losses/draws + win-rate per reference. This answers *"how good is this
bot at P\u00f4kemon?"* with an ABSOLUTE number — win-rate against a ladder of
increasingly-strong FIXED opponents — not just relative ELO within the league.

**Reuse, don't rewrite** (the project's one-workflow rule): the battle
primitive is ``league.driver.play_agent_battle_on`` (agent_a = the bot on p1,
agent_b = the reference on p2) and the server source is
``pool.ResourcePool.battle_slot()`` (a ``ServerConfiguration`` context
manager). The harness adds only the battle loop + the win/loss/draw
aggregation.

CPU-first discipline: importing this module never starts a server (the pool is
injected) and never pulls torch (the net/agent is built lazily, and only when
a REAL battle is played — the hermetic test injects a fake env + a fixed
decision so no net, no server, no GPU is touched).

Sequential-first (A2). Concurrency is an A5 concern; the loop is written so a
worker-thread front can be added later without touching the aggregation.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

__all__ = ["PerRefResult", "EvalResult", "evaluate", "compute_fitness"]

#: a bot = a checkpoint path (str/Path) OR a ``seed -> PokeAgent`` factory
#: OR a ``Reference`` (reuse a ladder entry as the bot, e.g. ref-vs-ref).
BotSpec = Union[str, os.PathLike, Callable[[Optional[int]], Any], Any]


@dataclass
class PerRefResult:
    """The bot's record against ONE reference (the ladder's one rung)."""
    ref: str
    wins: int = 0      # bot (p1) won, reference (p2) lost
    losses: int = 0    # reference (p2) won, bot (p1) lost
    draws: int = 0     # neither won (stall to the step cap / double K.O.)
    steps: int = 0     # total battle steps (diagnostic: how long they last)

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.draws

    @property
    def win_rate(self) -> float:
        return self.wins / self.n if self.n else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref": self.ref, "wins": self.wins, "losses": self.losses,
            "draws": self.draws, "n": self.n, "win_rate": self.win_rate,
            "steps": self.steps,
        }


@dataclass
class EvalResult:
    """The full evaluation: per-reference records + the aggregate score.

    ``fitness_score`` is the weighted aggregate the user ranks bots by (A4):
    the weighted AVERAGE of the per-reference win-rates (see
    :func:`compute_fitness`), normalized to [0, 1]. Weights come from each
    :class:`~fitness.ladder.Reference`'s ``.weight`` (stronger rungs count
    more), so a bot that beats the floor but loses to the strong rungs lands
    mid-range, and one that beats every rung lands near 1.0."""
    format: str
    n_battles: int
    per_ref: Dict[str, PerRefResult] = field(default_factory=dict)
    fitness_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "format": self.format,
            "n_battles": self.n_battles,
            "per_ref": {name: r.to_dict() for name, r in self.per_ref.items()},
            "fitness_score": self.fitness_score,
        }


def _resolve_bot(bot: BotSpec, vocabs: Any, device: str) -> Callable[[Optional[int]], Any]:
    """Normalize ``bot`` into a ``seed -> PokeAgent`` factory.

    * a checkpoint **path** (str/Path) -> ``PokeNet.load_file`` (THE ONE load
      path, D103 step 1) + ``PokeAgent``;
    * a ``Reference`` (A1) -> its ``.agent(seed)`` (ref-vs-ref / ref-vs-ref);
    * a **callable** -> assumed ``seed -> agent`` (already a factory).
    """
    if isinstance(bot, (str, os.PathLike)):
        from net.model import PokeNet
        from net.agent import PokeAgent

        def make(seed: Optional[int] = None) -> Any:
            net = PokeNet.load_file(bot, vocabs, device=device)
            return PokeAgent(net, seed=seed, device=device)

        return make
    if hasattr(bot, "agent") and hasattr(bot, "make_agent"):  # a Reference
        return bot.agent
    if callable(bot):
        return bot
    raise TypeError(
        f"bot must be a checkpoint path, a Reference, or a seed->agent "
        f"factory (got {type(bot).__name__})")


def _normalize_refs(refs: Any) -> List[Any]:
    """A ``Ladder`` / list / single ``Reference`` -> a list of references."""
    if refs is None:
        raise ValueError("refs must be a non-empty Ladder or list of Reference")
    if hasattr(refs, "all"):          # a Ladder
        out = list(refs.all())
    elif hasattr(refs, "agent") and hasattr(refs, "make_agent"):  # one Reference
        out = [refs]
    else:                              # a list/sequence of references
        out = list(refs)
    if not out:
        raise ValueError("refs must contain at least one reference")
    return out


@contextmanager
def _server(pool: Any, server_cfg: Any):
    """A ready-to-battle ``ServerConfiguration`` for ONE battle.

    * an injected ``server_cfg`` (tests) -> yield it as-is;
    * a started ``pool`` -> reserve a slot (back-pressure) and yield it;
    * neither -> yield a placeholder (ONLY valid with a fake ``env_factory`` —
      the driver never connects in that case; a real battle without a server is
      caught earlier with a clear error).
    """
    if server_cfg is not None:
        yield server_cfg
    elif pool is not None:
        with pool.battle_slot() as cfg:
            yield cfg
    else:
        yield {}


def evaluate(
    bot: BotSpec,
    refs: Any,
    n_battles: int,
    fmt: str = "gen9ou",
    *,
    vocabs: Any = None,
    device: str = "cpu",
    seed: int = 0,
    max_steps: int = 200,
    greedy: bool = True,
    pool: Any = None,
    server_cfg: Any = None,
    log: Optional[Callable[[str], None]] = None,
    decide: Optional[Callable[..., Any]] = None,
    env_factory: Optional[Callable[[], Any]] = None,
) -> EvalResult:
    """Play ``n_battles`` of ``bot`` (p1) vs EACH reference (p2) and aggregate.

    :param bot: a checkpoint **path** (str/Path), a ``Reference``, or a
                ``seed -> PokeAgent`` factory.
    :param refs: a ``Ladder`` or a non-empty list of ``Reference``.
    :param n_battles: battles to play AGAINST EACH reference.
    :param fmt: the battle format (e.g. ``gen9ou``, ``gen9bss3v3``).
    :param vocabs: the net's vocab db (needed to build the bot / references).
    :param device: ``cpu`` by default (the harness is CPU-first).
    :param seed: base seed; each (reference, battle) gets a unique derived seed
                (deterministic under a fixed seed).
    :param max_steps / greedy: forwarded to the battle primitive.
    :param pool: a started ``ResourcePool`` (reserves a slot per battle) — for
                real battles. Provide ``pool`` XOR ``server_cfg``.
    :param server_cfg: an injected ``ServerConfiguration`` (tests) — use instead
                of ``pool`` to avoid starting servers.
    :param decide / env_factory: TEST-INJECTION seams forwarded to the driver.
                With a fake ``env_factory`` + a fixed ``decide``, no server and
                no torch are touched (the hermetic aggregation test).
    :return: an :class:`EvalResult` (per-ref wins/losses/draws/win-rate).
    """
    if n_battles <= 0:
        raise ValueError(f"n_battles must be > 0 (got {n_battles})")
    refs_list = _normalize_refs(refs)
    if pool is None and server_cfg is None and env_factory is None:
        raise ValueError(
            "evaluate() needs a pool (real battles), an injected server_cfg, "
            "or a fake env_factory (tests) to play on")
    bot_make = _resolve_bot(bot, vocabs, device)
    # Real battles build a real observation (net.env.build_obs), which REQUIRES
    # vocabs. Checked AFTER bot validation (so a bad bot still reports the bot
    # error) but before ANY battle (so a missing vocabs fails LOUD at the API
    # boundary instead of a blind zero-obs deep in the encoder — the A3
    # self-check bug: heuristic vs random ran 3/3 with BOTH sides playing blind).
    # A fake `decide` (test injection) bypasses that path, so only then may
    # vocabs be absent.
    if vocabs is None and decide is None:
        raise ValueError(
            "evaluate(): `vocabs` is required for real battles "
            "(net.vocab_db.build_vocabs()); it is only optional when a fake "
            "`decide` (test injection) bypasses the real observation path. "
            "Playing without it would make the bot blind (zero-obs every turn).")

    # THE battle primitive (one-workflow): agent_a = bot (p1), agent_b = ref (p2).
    from league.driver import play_agent_battle_on

    result = EvalResult(format=fmt, n_battles=n_battles)
    battle_idx = 0
    for ref in refs_list:
        pr = PerRefResult(ref=ref.name)
        for _ in range(n_battles):
            b_seed = seed + battle_idx
            battle_idx += 1
            bot_agent = bot_make(b_seed)
            ref_agent = ref.agent(b_seed)
            with _server(pool, server_cfg) as scfg:
                a1_won, a2_won, n_steps = play_agent_battle_on(
                    scfg, agent_a=bot_agent, agent_b=ref_agent,
                    fmt=fmt, seed=b_seed, vocabs=vocabs, max_steps=max_steps,
                    greedy=greedy, log=log, decide=decide, env_factory=env_factory)
            if a1_won and not a2_won:
                pr.wins += 1
                outcome = "W"
            elif a2_won and not a1_won:
                pr.losses += 1
                outcome = "L"
            else:
                pr.draws += 1
                outcome = "D"
            pr.steps += n_steps
            if log is not None:
                log(f"[fitness] {ref.name} #{_ + 1}: {outcome} (steps={n_steps})")
        result.per_ref[ref.name] = pr
    # A4: the weighted aggregate. Weights already live on each Reference.
    result.fitness_score = compute_fitness(
        result.per_ref, {r.name: r.weight for r in refs_list})
    return result


def compute_fitness(per_ref: Any,
                    weights: Optional[Dict[str, float]] = None) -> float:
    """The weighted aggregate fitness score, normalized to [0, 1].

    A weighted AVERAGE of per-reference win-rates::

        sum_r(w_r * win_rate_r) / sum_r(w_r)

    Stronger references (higher ``w_r``) move the score more — a bot that beats
    the weak floor but loses to the strong rungs lands mid-range, a bot that
    beats every rung lands near 1.0. ``per_ref`` maps name ->
    :class:`PerRefResult` (or a raw win-rate ``float``); ``weights`` maps
    name -> weight (any reference not named defaults to weight 1.0).
    """
    num = 0.0
    den = 0.0
    for name, pr in (per_ref or {}).items():
        wr = float(pr) if isinstance(pr, (int, float)) else float(getattr(pr, "win_rate", 0.0))
        w = float((weights or {}).get(name, 1.0))
        num += w * wr
        den += w
    if den <= 0:
        return 0.0
    return max(0.0, min(1.0, num / den))
