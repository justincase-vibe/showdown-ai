"""Concrete, instantiable poke-env battle environment.

poke-env's ``SinglesEnv`` is still *abstract*: it never implements ``embed_battle`` /
``calc_reward`` and never assigns ``observation_spaces``, so it cannot be instantiated
and its ``SingleAgentWrapper`` crashes. This class fills exactly that gap on top of the
(already working) ``PokeEnv`` battle engine:

  * ``embed_battle`` -> our fixed-length feature vector (see ``obs.py``).
  * ``calc_reward``  -> +1 win / -1 loss / 0 otherwise (the paper's reward).
  * ``observation_spaces`` / ``action_spaces`` assigned so the Gym contract is valid.
  * **Unified action space** (26) with action masks padded per-generation, so a SINGLE
    shared policy can play gen6 (14 valid actions), gen7 (18), etc. -- the legal actions
    for a gen are always a *prefix* of the unified space, so masking is exact.

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

from obs import OBS_DIM, embed_battle as _embed_raw
from state_from_log import reconstruct_current_state as _reconstruct_current

# Covers gen6(14) .. gen9(26); gen-specific legal actions are a prefix of this space.
UNIFIED_ACTION_SIZE = 26


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
    def get_action_mask(battle: Any) -> list:
        mask = SinglesEnv.get_action_mask(battle)
        if len(mask) < UNIFIED_ACTION_SIZE:
            mask = mask + [0] * (UNIFIED_ACTION_SIZE - len(mask))
        elif len(mask) > UNIFIED_ACTION_SIZE:
            mask = mask[:UNIFIED_ACTION_SIZE]
        return mask

    @staticmethod
    def get_action_space_size(gen: int) -> int:
        # Keep the per-gen helper intact for introspection, but the *space* is unified.
        return UNIFIED_ACTION_SIZE


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
        # poke-env's action_to_order calls ``action.item()``, so it needs a numpy
        # scalar, not a plain Python int (a plain int raises AttributeError, which
        # its ``except ValueError`` does NOT catch).
        actions = {}
        if env.agent1_to_move:
            actions[self.a1] = np.int64(-2) if env.battle1.wait else np.int64(a1_action)
        if env.agent2_to_move:
            actions[self.a2] = np.int64(-2) if env.battle2.wait else np.int64(self._opp_action(opponent))
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
