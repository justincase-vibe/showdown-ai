"""Concrete, instantiable poke-env battle environment.

poke-env's ``SinglesEnv`` is still *abstract*: it never implements ``embed_battle`` /
``calc_reward`` and never assigns ``observation_spaces``, so it cannot be instantiated
and its ``SingleAgentWrapper`` crashes. This class fills exactly that gap on top of the
(already working) ``PokeEnv`` battle engine:

  * ``embed_battle`` -> our fixed-length feature vector (see ``obs.py``).
  * ``calc_reward``  -> +1 win / -1 loss / 0 otherwise (the paper's reward).
  * ``observation_spaces`` / ``action_spaces`` assigned so the Gym contract is valid.
  * **Unified move-identity action space** (963) -- move ids + 6 switches + 3 gimmick
    transforms -- with per-situation boolean masks, so a SINGLE shared policy plays
    gen6-9. Actions are a move *by identity* (not a slot), so train and live never drift.

The class intentionally keeps poke-env's PettingZoo two-agent (``A``/``B``) stepping so
we reuse its correct battle/order/queue machinery. The self-play training loop (train.py)
drives it directly.
"""

from __future__ import annotations

import asyncio
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from gymnasium import spaces

from poke_env.environment.singles_env import SinglesEnv
from poke_env.ps_client import LocalhostServerConfiguration, ServerConfiguration
from poke_env.teambuilder import Teambuilder

from action_space import get_action_space, canon_move, TRANSFORMS
from obs import OBS_DIM, embed_battle as _embed_raw
from state_from_log import reconstruct_current_state as _reconstruct_current

# Unified *move-identity* action space (one policy across gens 6-9):
#   MOVE 0..N-1 | SWITCH N..N+5 | TRANSFORM N+6..N+8 (mega/dynamax/tera) -> N_ACTIONS total.
UNIFIED_ACTION_SIZE = get_action_space().N_ACTIONS


class ShowdownEnv(SinglesEnv):
    def __init__(
        self,
        *,
        battle_format: str = "gen7ou",
        team: Optional[Union[str, Teambuilder]] = None,
        server_configuration: Optional[ServerConfiguration] = LocalhostServerConfiguration,
        save_replays: Union[bool, str] = False,
        log_level: Optional[int] = None,
        account_configuration1: Optional[Any] = None,
        account_configuration2: Optional[Any] = None,
        **kwargs: Any,
    ):
        super().__init__(
            account_configuration1=account_configuration1,
            account_configuration2=account_configuration2,
            battle_format=battle_format,
            team=team,
            server_configuration=server_configuration,
            save_replays=save_replays,
            log_level=log_level,
            **kwargs,
        )
        # --- Fix the action space to a unified size (one policy across gens) ---
        self.action_spaces = {
            agent: spaces.Discrete(UNIFIED_ACTION_SIZE)
            for agent in self.possible_agents
        }
        # --- Provide the observation space (this also auto-builds the action_mask) ---
        self.observation_spaces = {
            agent: spaces.Box(
                low=-1e9, high=1e9, shape=(OBS_DIM,), dtype=np.float32
            )
            for agent in self.possible_agents
        }

    # ------------------------------------------------------------------ obs
    def embed_battle(self, battle: Any) -> Any:
        """Build the observation the policy actually sees.

        **Train/live equivalence.** poke-env populates the trainee's *own* mons with
        their full moveset, real item and real ability, and shows the opponent's mons
        with empty moves. The imitation-training observation, by design, is
        *revealed-only* on both sides (PLAN.md §3.1) so the bot must learn to adapt
        rather than rely on a hard-coded moveset. Encoding the two differently would
        be a silent train/live mismatch that poisons every downstream policy.

        So instead of embedding the raw live mon objects, we rebuild the observation
        from the live battle's own protocol stream (``battle._build_replay_log()``) with
        the *same* reconstruction code (``state_from_log``) used to build the training
        data. Same handlers, same layout, same revealed-only rules -> the live and
        training observations are identical for a given battle state.
        """
        try:
            log = battle._build_replay_log()
            gen = 8
            for ln in log.splitlines():
                if ln.startswith("|gen|"):
                    try:
                        gen = int(ln.split("|")[2])
                    except (ValueError, IndexError):
                        pass
                    break
            # Per-agent perspective: poke-env calls embed_battle(battle1) for a1 and
            # embed_battle(battle2) for a2, and a1 (the challenger) is P1 in the log.
            # Both battle objects carry the full protocol stream, so the same log is
            # rebuilt from whichever side is being served. Without this, the P2
            # opponent acted on a P1-perspective observation (own/opponent swapped)
            # and was systematically handicapped -> 20/0 self-play instead of ~50/50.
            side = "p2" if battle is getattr(self, "battle2", None) else "p1"
            return _reconstruct_current(log, gen, side)
        except Exception:
            # Defensive fallback: raw encoding (pre-mask) rather than crashing a battle.
            # _embed_raw reads the per-agent battle object directly, so it is already
            # perspective-correct.
            return _embed_raw(battle)

    # ---------------------------------------------------------------- reward
    # Simple terminal-only reward. Zero-sum by construction: a win on one side is the
    # loss on the other. No per-step shaping (trading is a valid strategy, switch count
    # doesn't matter -- only the outcome does).
    #
    #   Win:  +1.0
    #   Loss: -1.0
    #
    # There is deliberately NO forfeit reward. A "forfeit" in this stack is not an action
    # the policy can take (the 963 space is moves+switches+transforms only) -- it is
    # poke-env sending /forfeit when a battle is reset()/close()d before it finished, i.e.
    # a harness artifact, not a learned choice. We therefore treat a forfeited battle as
    # an ordinary loss (our team did not win), which keeps the reward zero-sum and stops
    # us from "shaping" a button the net never had. If your stats show forfeits, that is a
    # BUG to fix in the harness (train.py), not a reward to tune.
    REWARD_WIN = 1.0
    REWARD_LOSS = -1.0

    def calc_reward(self, battle: Any) -> float:
        if battle.won:
            return self.REWARD_WIN
        if battle.lost:
            return self.REWARD_LOSS
        return 0.0

    # ------------------------------------------------------- unified action mask
    @staticmethod
    def get_action_mask(battle: Any) -> np.ndarray:
        """963-space legal-action mask (move-identity).

        Mirrors poke-env's legality (available_moves / available_switches / can_mega / can_tera)
        but emits actions as *move identities* rather than slots, so training and live play
        never drift (the whole point of the redesign). Gimmicks are single TRANSFORM actions;
        the paired move is chosen at step-time (v1 simplification).
        """
        as_ = get_action_space()
        mask = np.zeros(as_.N_ACTIONS, dtype=bool)
        if getattr(battle, "_wait", False):
            return mask
        active = getattr(battle, "active_pokemon", None)
        force_switch = bool(getattr(battle, "force_switch", False))
        try:
            team = list((getattr(battle, "team", None) or {}).values())
        except Exception:
            team = []

        if active is not None and not force_switch:
            # --- moves (identity) ---
            try:
                known = list(active.moves.values())[:4]
                avail = {getattr(m, "id", None)
                         for m in (getattr(battle, "available_moves", None) or [])}
            except Exception:
                known, avail = [], set()
            for m in known:
                mid = getattr(m, "id", None)
                if mid is None:
                    continue
                if avail and mid not in avail:
                    continue  # move currently restricted (Encore, etc.)
                i = as_.move_id(mid)
                if i is not None:
                    mask[i] = True
            # --- transforms (mega / dynamax / tera) ---
            if getattr(battle, "can_mega_evolve", False):
                t = as_.transform("mega")
                if t is not None:
                    mask[t] = True
            if getattr(battle, "can_dynamax", False):
                t = as_.transform("dynamax")
                if t is not None:
                    mask[t] = True
            if getattr(battle, "can_tera", False):
                t = as_.transform("tera")
                if t is not None:
                    mask[t] = True

        # --- switches (only the ones the server says are available) ---
        # ROOT-CAUSE FIX: unmask a switch ONLY if that mon is in `available_switches`.
        # The old code used `if avail_sw and bs not in avail_sw: continue`, so an EMPTY
        # available_switches (whole bench fainted / no switchable mon) fell through and
        # allowed ALL six slots. The policy would then pick a switch the server rejects
        # -> the "re-pick / forced into an unintended mon" bug. An empty set must mean
        # "no switches", exactly like poke-env's own get_action_mask.
        if not bool(getattr(battle, "trapped", False)):
            try:
                avail_sw = {getattr(p, "base_species", None)
                            for p in (getattr(battle, "available_switches", None) or [])}
                avail_sw.discard(None)
            except Exception:
                avail_sw = set()
            for slot, mon in enumerate(team):
                bs = getattr(mon, "base_species", None)
                if bs is None or bs not in avail_sw:
                    continue
                sid = as_.switch_id(slot)
                if sid is not None:
                    mask[sid] = True
        return mask

    @staticmethod
    def get_action_space_size(gen: int) -> int:
        return UNIFIED_ACTION_SIZE

    # ------------------------------------------------------------------ teardown
    def close(self, force: bool = True, wait: bool = True):
        """Tear down WITHOUT ever sending a forfeit (root-cause fix).

        poke-env's ``PokeEnv.close()`` does two things for a battle that is not
        finished yet: it sends a ``ForfeitBattleOrder`` (the *sole* source of the
        spurious 'X forfeited / win Y' replays -- one per env, at the end of every
        run), and it then blocks on ``_challenge_task.result()``, which only completes
        *because* that forfeit ended the battle.

        A forfeit is a harness artifact, not an action the policy can take (our 963
        space is moves + switches + transforms only), so we must never emit one. Instead
        we abandon the live battle cleanly: no forfeit, no blocking wait (the challenge
        task is still pending for a live battle), and no ``reset_battles()`` (it raises
        while a battle is live). We just drop the references and drain the queues; the
        abandoned local self-play battle is cancelled with the event loop at process
        exit. When a battle has genuinely finished (the normal end-of-run path after
        draining), ``reset_battles()`` runs safely and the queue drain is a no-op.
        """
        if self._challenge_task is not None and self._challenge_task.done():
            self.reset_battles()  # only safe once the battle has actually finished
        self._challenge_task = None
        self.battle1 = None
        self.battle2 = None
        self.agent1.battle = None
        self.agent2.battle = None
        for q in (
            self.agent1.order_queue, self.agent2.order_queue,
            self.agent1.battle_queue, self.agent2.battle_queue,
        ):
            try:
                while not q.empty():
                    q.get()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 963-space (move-identity) action  ->  poke-env slot-based int
# ---------------------------------------------------------------------------
def _find_move_idx(battle, name) -> int:
    """Index of ``name`` (canonical) in the active mon's known moves (<=4), else -1."""
    c = canon_move(name)
    if c is None:
        return -1
    active = getattr(battle, "active_pokemon", None)
    if active is None:
        return -1
    try:
        known = list(active.moves.values())[:4]
    except Exception:
        return -1
    for i, m in enumerate(known):
        if canon_move(getattr(m, "id", None)) == c:
            return i
    for i, m in enumerate(known):
        if canon_move(getattr(m, "name", None)) == c:
            return i
    return -1


def _to_pe_action(my_action: int, battle) -> int:
    """Translate a move-identity action to poke-env's int, before ``env.step``.

    My space:   MOVE 0..N-1 | SWITCH N..N+5 | TRANSFORM N+6..N+8 (mega/dynamax/tera)
    poke-env:   0..5 switch | 6..9 move | 10..13 mega | 14..17 z | 18..21 dynamax | 22..25 tera

    A gimmick (transform) is paired with the first known move (v1: the "which move" signal
    is dropped for mega/tera/dynamax, per the approved design). poke-env validates the
    resulting order against ``valid_orders`` (strict=False -> falls back to a random move),
    so a stale identity can never crash a battle.
    """
    if my_action < 0:
        return my_action  # -2 default / -1 forfeit pass straight through
    as_ = get_action_space()
    band = as_.band(my_action)
    if band == "switch":
        return my_action - as_.SWITCH_BASE  # 0..5
    if band == "transform":
        kind = TRANSFORMS[my_action - as_.TRANSFORM_BASE]
        base = {"mega": 10, "dynamax": 18, "tera": 22}[kind]
        return base + 0  # pair with the first known move
    # MOVE band (v1: plain moves; Z / G-Max identities are reserved for v2)
    move_name = as_.id_to_move[my_action]
    idx = _find_move_idx(battle, move_name)
    idx = 0 if idx < 0 else idx
    return 6 + (idx % 4)


def _pe_valid(pe: int, battle) -> bool:
    """Whether the poke-env int ``pe`` converts to a currently valid order."""
    try:
        SinglesEnv.action_to_order(np.int64(pe), battle, strict=True)
        return True
    except Exception:
        return False


def _order_to_our_action(order: Any, battle: Any) -> Optional[int]:
    """Map a poke-env BattleOrder back into our 963 space (last-resort fallback)."""
    pe = int(SinglesEnv.order_to_action(order, battle, strict=True))
    as_ = get_action_space()
    if 0 <= pe <= 5:
        return as_.SWITCH_BASE + pe
    if 6 <= pe <= 9:
        known = list(getattr(battle, "active_pokemon", None).moves.values())[:4]
        return as_.move_id(getattr(known[pe - 6], "id", None))
    if 10 <= pe <= 13:
        return as_.transform("mega")
    if 18 <= pe <= 21:
        return as_.transform("dynamax")
    if 22 <= pe <= 25:
        return as_.transform("tera")
    return 0


# ---------------------------------------------------------------------------
# Single-agent wrapper (torch-free: usable for env verification too)
# ---------------------------------------------------------------------------
class TraineeEnv:
    """Exposes agent1 (the trainee) of a two-agent ``ShowdownEnv`` as a clean
    single-agent interface. ``opponent`` is either the string ``"random"`` or any
    object with ``.act(obs, mask) -> (action, logprob, value)`` (frozen / live policy).
    Handles the rare ``wait`` state by sending the default (pass) order."""

    def __init__(self, env: ShowdownEnv, env_id: int, fmt: str):
        self.env = env
        self.id = env_id
        self.fmt = fmt
        self.a1 = env.possible_agents[0]
        self.a2 = env.possible_agents[1]
        self.last_obs: Optional[np.ndarray] = None
        self.last_mask: Optional[np.ndarray] = None
        self.obs2: Optional[np.ndarray] = None
        self.mask2: Optional[np.ndarray] = None
        self.done = True
        self.re_picks = 0  # == illegal picks the policy made and we had to re-ask about
        self.illegal_total = 0  # lifetime illegal picks (for logging)
        self._ill1 = 0  # P1 illegal-pick count this episode (logging only)
        self._ill2 = 0  # P2 illegal-pick count this episode (logging only)
        # Both-sides self-play: each side's decision, captured in step() so a SHARED
        # policy can learn from BOTH sides (zero-sum reward). See _decide.
        self.last_self: Optional[dict] = None  # P1 (trainee) decision
        self.last_opp: Optional[dict] = None   # P2 (opponent) decision
        self.last_opp_rew: float = 0.0

    def reset(self) -> np.ndarray:
        try:
            obs, _ = self.env.reset()
        except asyncio.TimeoutError as exc:
            # poke-env's ``"Agent is not challenging"`` timeout is almost always a
            # *misleading* message: the challenge never completes because the server
            # REJECTED the team (e.g. a random bag drew two of the same species ->
            # Species Clause) and bounced it. Surface the real cause so it is not
            # mistaken for a torch/asyncio/threading bug.
            raise RuntimeError(
                "reset() timed out waiting for the battle to start. Two usual causes: "
                "(1) the server REJECTED the team at battle start -- check the log for a "
                "poke_env WARNING like 'Your team was rejected ... Species Clause' (for "
                "random-bag teams ensure the pool is deduplicated by species: "
                "teams.load_verified_pool does this); (2) the server is not running at "
                "all -- if the log shows 'ConnectionRefusedError' / 'WinError 1225', "
                "start it first: cd pokemon-showdown && node pokemon-showdown start. "
                "Re-raising the original timeout for the stack trace."
            ) from exc
        self.last_obs = obs[self.a1]["observation"]
        self.last_mask = obs[self.a1]["action_mask"]
        self.obs2 = obs[self.a2]["observation"]
        self.mask2 = obs[self.a2]["action_mask"]
        self.done = False
        self._ill1 = 0
        self._ill2 = 0
        return self.last_obs

    # A pick the server rejects (rare, only if the mask is momentarily stale) is NOT
    # penalized in reward -- terminal rewards stay purely win/loss (+1/-1). We simply
    # mask that pick out and ASK THE AGENT AGAIN until it emits a legal one, so its own
    # choice is honored and forced-switch states (U-turn / Volt Switch / Roar / Dragon
    # Tail / Whirlwind) stay in the training signal. The mask should match the server,
    # so re-asks should fire ~0 times; the per-episode counts are kept for logging only.
    MAX_REASK = 8

    def _decide(self, obs, mask, actor, battle, max_tries: int = MAX_REASK):
        """Have the agent pick a SERVER-LEGAL action, honoring its own choice.

        * If the mask is empty (Showdown 'wait'), return None (caller sends the -2 pass).
        * Otherwise sample from ``actor`` (a policy, or 'random') under the current mask
          and check the resulting order against the server. If it is legal, use it.
          If it is illegal, mask that pick out and ASK AGAIN (the agent keeps choosing)
          until it is legal (the count is logged only, with no reward effect). This is
          what makes forced-switch states (U-turn / Volt Switch / Roar / Dragon Tail /
          Whirlwind) work without shoving an arbitrary mon in: the agent's actual pick is
          used whenever it is legal.
        * If we exhaust re-asks (shouldn't happen once the mask is exact), fall back to a
          server-valid order so the battle can never stall/forfeit.

        Returns ``(action|None, logprob, value, n_illegal)`` -- a consistent
        ``(action, logprob)`` pair (the logprob is from the draw that produced the action),
        which is what keeps PPO's importance ratio correct.
        """
        m = np.array(mask, dtype=bool).copy() if mask is not None else np.zeros(UNIFIED_ACTION_SIZE, dtype=bool)
        if int(m.sum()) == 0:
            return None, 0.0, 0.0, 0  # 'wait': no decision this turn
        n_illegal = 0
        for _ in range(max_tries):
            if actor is None or actor == "random":
                legal = np.flatnonzero(m)
                a = int(legal[np.random.randint(len(legal))]) if len(legal) else 0
                lp = 0.0
                v = 0.0
            else:
                a, lp, v = actor.act(obs, m)
                a = int(a)
            if _pe_valid(_to_pe_action(a, battle), battle):
                return a, float(lp), float(v), n_illegal
            # Illegal pick: the agent gets to choose again (this pick is masked out).
            m[a] = False
            n_illegal += 1
            if int(m.sum()) == 0:
                break
        # Exhausted re-asks: take a server-valid order so the battle cannot stall.
        a = self._first_valid_from_orders(battle)
        return a, 0.0, 0.0, n_illegal + 1

    @staticmethod
    def _first_valid_from_orders(battle: Any) -> int:
        """Last-resort: first action the server itself lists as valid (valid_orders)."""
        try:
            for order in battle.valid_orders:
                a = _order_to_our_action(order, battle)
                if a is not None:
                    return int(a)
        except Exception:
            pass
        return get_action_space().SWITCH_BASE  # absolute fallback

    def step(self, actor1, actor2) -> Tuple[np.ndarray, float, bool]:
        """Advance one battle. ``actor1``/``actor2`` are each a policy (with ``.act``) or
        the string/None 'random'. Both sides make their own, server-validated decision
        (see ``_decide``); a SHARED policy (``--opponent live``) plays both sides and both
        transitions train the model (zero-sum reward).
        """
        env = self.env
        # Capture each side's decision state BEFORE the post-step overwrite of obs/mask.
        obs1_pre, mask1_pre = self.last_obs, self.last_mask
        obs2_pre, mask2_pre = self.obs2, self.mask2
        actions: dict = {}
        a1 = lp1 = v1 = ill1 = None
        a2 = lp2 = v2 = ill2 = None

        # --- P1 (trainee) ---
        if env.agent1_to_move and self.last_mask is not None and int(np.sum(self.last_mask)) > 0:
            a1, lp1, v1, ill1 = self._decide(obs1_pre, self.last_mask, actor1, env.battle1)
            if a1 is not None:
                # numpy int64: poke-env's action_to_order calls ``action.item()``.
                actions[self.a1] = np.int64(_to_pe_action(a1, env.battle1))
        # --- P2 (opponent) ---
        if env.agent2_to_move and self.mask2 is not None and int(np.sum(self.mask2)) > 0:
            a2, lp2, v2, ill2 = self._decide(obs2_pre, self.mask2, actor2, env.battle2)
            if a2 is not None:
                actions[self.a2] = np.int64(_to_pe_action(a2, env.battle2))

        # Not our turn to move (Showdown 'wait'): send the default pass order.
        actions.setdefault(self.a1, np.int64(-2))
        actions.setdefault(self.a2, np.int64(-2))

        # Illegal-pick accounting (drives the slight terminal penalty + logging).
        ill1 = int(ill1) if ill1 is not None else 0
        ill2 = int(ill2) if ill2 is not None else 0
        self.re_picks += ill1 + ill2
        self.illegal_total += ill1 + ill2
        self._ill1 += ill1
        self._ill2 += ill2

        new_obs, rew, term, trunc, _info = env.step(actions)
        self.last_obs = new_obs[self.a1]["observation"]
        self.last_mask = new_obs[self.a1]["action_mask"]
        self.obs2 = new_obs[self.a2]["observation"]
        self.mask2 = new_obs[self.a2]["action_mask"]
        reward = float(rew[self.a1])
        reward2 = float(rew[self.a2])  # zero-sum: = -reward at a terminal win/loss
        done = bool(term[self.a1]) or bool(trunc[self.a1])
        self.done = done

        # Expose each side's decision so a shared policy can learn BOTH sides. Each dict
        # holds a consistent (action, logprob, value) pair under the mask that produced it.
        self.last_self = {
            "a": a1,
            "lp": float(lp1) if lp1 is not None else 0.0,
            "v": float(v1) if v1 is not None else 0.0,
            "obs": obs1_pre, "mask": mask1_pre,
            "ill": int(ill1) if ill1 is not None else 0,
            "made": a1 is not None,
        }
        self.last_opp = {
            "a": a2,
            "lp": float(lp2) if lp2 is not None else 0.0,
            "v": float(v2) if v2 is not None else 0.0,
            "obs": obs2_pre, "mask": mask2_pre,
            "ill": int(ill2) if ill2 is not None else 0,
            "made": a2 is not None,
        }
        self.last_opp_rew = reward2
        return self.last_obs, reward, done

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
