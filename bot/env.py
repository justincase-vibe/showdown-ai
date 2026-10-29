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

from typing import Any, Optional, Tuple, Union

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
            return _reconstruct_current(log, gen)
        except Exception:
            # Defensive fallback: raw encoding (pre-mask) rather than crashing a battle.
            return _embed_raw(battle)

    # ---------------------------------------------------------------- reward
    def calc_reward(self, battle: Any) -> float:
        if battle.won:
            return 1.0
        if battle.lost:
            return -1.0
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

        # --- switches (alive + available; always allowed unless trapped) ---
        if not bool(getattr(battle, "trapped", False)):
            try:
                avail_sw = [getattr(p, "base_species", None)
                            for p in (getattr(battle, "available_switches", None) or [])]
            except Exception:
                avail_sw = []
            for slot, mon in enumerate(team):
                bs = getattr(mon, "base_species", None)
                if bs is None:
                    continue
                if avail_sw and bs not in avail_sw:
                    continue
                sid = as_.switch_id(slot)
                if sid is not None:
                    mask[sid] = True
        return mask

    @staticmethod
    def get_action_space_size(gen: int) -> int:
        return UNIFIED_ACTION_SIZE


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

    def reset(self) -> np.ndarray:
        obs, _ = self.env.reset()
        self.last_obs = obs[self.a1]["observation"]
        self.last_mask = obs[self.a1]["action_mask"]
        self.obs2 = obs[self.a2]["observation"]
        self.mask2 = obs[self.a2]["action_mask"]
        self.done = False
        return self.last_obs

    def step(self, a1_action: int, opponent) -> Tuple[np.ndarray, float, bool]:
        env = self.env
        # poke-env's action_to_order calls ``action.item()``, so it needs a numpy scalar,
        # not a plain Python int (a plain int raises AttributeError, which its
        # ``except ValueError`` does NOT catch).
        #
        # CRITICAL: our actions live in the 963-space (move-identity); poke-env expects
        # its own slot-based ints. Translate *before* env.step -- passing a raw 963 id
        # straight through would index the wrong move (or a switch/transform) silently.
        actions = {}
        if env.agent1_to_move:
            actions[self.a1] = np.int64(-2) if env.battle1.wait else np.int64(_to_pe_action(a1_action, env.battle1))
        if env.agent2_to_move:
            pe_opp = _to_pe_action(self._opp_action(opponent), env.battle2)
            actions[self.a2] = np.int64(-2) if env.battle2.wait else np.int64(pe_opp)
        actions.setdefault(self.a1, np.int64(-2))
        actions.setdefault(self.a2, np.int64(-2))

        new_obs, rew, term, trunc, _info = env.step(actions)
        self.last_obs = new_obs[self.a1]["observation"]
        self.last_mask = new_obs[self.a1]["action_mask"]
        self.obs2 = new_obs[self.a2]["observation"]
        self.mask2 = new_obs[self.a2]["action_mask"]
        reward = float(rew[self.a1])
        done = bool(term[self.a1]) or bool(trunc[self.a1])
        self.done = done
        return self.last_obs, reward, done

    def _opp_action(self, opponent) -> int:
        if opponent is None or opponent == "random":
            legal = np.flatnonzero(self.mask2)
            if len(legal) == 0:
                legal = np.array([0])
            return int(np.random.choice(legal))
        a, _, _ = opponent.act(self.obs2, self.mask2)
        return int(a)

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
