"""P1 — the self-play battle environment (approved 6-phase plan, phase 1).

This is the LIVE counterpart of the BC observation pipeline. It reuses the SAME
proven building blocks the imitation path already validated, just driven from a
running fork battle instead of a static replay:

    fork battle log  ->  belief (LogAdapter -> BeliefState -> get_context)
                     ->  net.encoder.encode_context  ->  net.registry layout

So the observation an agent sees in self-play is built by the exact same
belief/encoder/registry code that the net's embedding tables were sized against
(net/registry.py is the single source of truth for the tensor layout). There is
no second, hand-maintained feature vector to drift — the D61 OBS_DIM bug class
cannot reappear here because the width is derived from the registry.

Design decisions (recorded for the user, see PLAN.md / handoff):
  * **Action space = the D60 current-legal-action list** (a variable list of
    ``{kind, move|species}`` dicts, capped at ``registry.MAX_ACTIONS``), NOT the
    legacy fixed 963 move-identity space. The net's pointer head scores each
    listed action by IDENTITY (position-independent), which is exactly what this
    list feeds (``action_kind`` / ``action_move`` / ``action_species`` channels).
    ``env.step`` still consumes poke-env-native ints (0..5 switch, 6..9 move,
    22..25 tera); ``action_to_pe`` is the bridge from a D60 list entry to that int.
  * **Reward = sparse terminal ±1** (win +1 / loss -1 / 0 otherwise), zero-sum by
    construction. No per-step shaping (a trade is a valid strategy; only the
    outcome matters). A forfeit is a harness artifact, not a learnable action, so
    a forfeited battle is simply a loss (our side did not win) — we must never
    *emit* a forfeit (see ``close``).
  * **POV-correct observation.** The belief pipeline is p1-POV by construction
    ("own" = p1). For the p2 agent we swap the ``p1``/``p2`` side/slot tokens in
    the protocol log before feeding it, so "own" becomes the real p2 team and
    "opponent" the real p1 team — the same layout, just re-perspectived. This is
    what dissolves the BC unknown-moveset bug for our own side: an agent's own
    six are its own, fully known, no "revealed" marker needed.
  * **Defensive observation.** A belief/encode failure (e.g. a species with no
    prior in the format) must NEVER crash a live battle — it degrades to a
    zero observation (logged) so the battle and the training loop survive.

Performance note (flagged for P3, not a P1 blocker): ``embed_battle`` rebuilds
the belief from the current full log each call. The belief SOFT path (3.4)
spawns a Node calc process per unique mon spec (``~250ms`` cold, then cached in
``belief.evidence._CACHE``), so the *first* observation of a battle is the
expensive one and later ones are warm. For a training loop this should become an
INCREMENTAL belief (feed only new protocol lines since the last step) — that is a
P3 optimization; P1 keeps the simple, obviously-correct rebuild.
"""
from __future__ import annotations

import logging
import random as _random
import re
from typing import Any, Dict, List, Optional

import numpy as np

from poke_env.environment.singles_env import SinglesEnv
from poke_env.ps_client import (
    LocalhostServerConfiguration,
    ServerConfiguration,
)
from poke_env.teambuilder import Teambuilder, TeambuilderPokemon

from belief.log_events import LogAdapter, Unknown
from belief.state import BeliefState
from net.encoder import encode_context
from net.registry import LAYOUT, MAX_ACTIONS, total_elements

log = logging.getLogger("net.env")


# ═══════════════════════════════════════════════════════════════════════════════
#  POV side-swap  (p2 observation)
# ═══════════════════════════════════════════════════════════════════════════════
# The belief stack is p1-POV ("own" = p1). To serve the p2 agent we swap the
# p1/p2 side+slot tokens in the protocol log so that, to the belief, the real p2
# team reads as "own". Showdown side tokens are exactly ``p1``/``p2`` (bare side)
# or ``p1[abcdef]``/``p2[abcdef]`` (a slot). No species/move id contains a
# standalone ``p1``/``p2`` token with word boundaries on both sides, so the
# boundary-anchored regex below is safe to apply across the whole log.
_SIDE_SLOT = re.compile(r"\b(p[12])([a-f]?)\b")


def swap_sides(log_text: str) -> str:
    """Swap p1<->p2 side/slot tokens in a Showdown protocol log (for p2-POV)."""
    def _sub(m: "re.Match") -> str:
        side, slot = m.group(1), m.group(2)
        return ("p2" if side == "p1" else "p1") + slot
    return _SIDE_SLOT.sub(_sub, log_text)


# ═══════════════════════════════════════════════════════════════════════════════
#  own-team extraction  (populates get_context()['own'])
# ═══════════════════════════════════════════════════════════════════════════════
def _own_team_from_battle(battle: Any) -> Dict[str, Dict]:
    """p1's own-team spec from the poke-env battle object — the AUTHORITATIVE
    source for the own side (D74).

    The ``|poke|`` team-preview log lines carry species+gender only — NO moves,
    item, ability — which is exactly why the log-derived spec below left the
    net's ``own_moves`` / ``own_ability`` / ``own_item`` channels ALL ZERO from
    turn 1. But poke-env seeds ``battle.team`` from our OWN Teambuilder
    (``Pokemon.from_teambuilder``), so every own mon is FULLY known from turn 1:
    species, the complete 4-move set, item, ability. That is the self-play
    premise (D73: "the agent's own team is FULLY KNOWN — it built it").

    Deliberately EXCLUDES the tera type: the calc bridge (``node_calc.js``
    ``effectiveTypes``) treats a mon's ``teraType`` as *currently terastallized*
    (it replaces the base types and grants Tera STAB in EVERY window). Passing
    the known-but-not-yet-revealed tera would therefore corrupt every pre-Tera
    damage window for the own mon. The revealed Tera lives in the belief's
    ``facts.tera`` (HARD evidence) and is the only tera signal the observation
    carries (``tera_revealed``).

    Returns ``slot p1a..p1f -> spec`` (the belief ``own_team`` shape, preview
    order); ``{}`` when ``battle.team`` is not populated yet, in which case the
    caller falls back to the log-derived name-only spec.
    """
    team = getattr(battle, "team", None) or {}
    out: Dict[str, Dict] = {}
    for i, mon in enumerate(team.values()):
        if i >= 6:
            break
        # ``species`` = the ACTUAL forme id (slowkinggalar, tornadustherian);
        # ``base_species`` strips the forme (slowking, tornadus) and must only
        # be a fallback — encoding the base form mislabels forme mons in the
        # own_species rows (caught by the p1_bug_hunt A probe).
        sp = getattr(mon, "species", None) or getattr(mon, "base_species", None)
        if not sp:
            continue
        moves_obj = getattr(mon, "moves", None) or {}
        if isinstance(moves_obj, dict):
            moves = [getattr(m, "id", None) or k for k, m in moves_obj.items()]
        else:
            moves = [str(m) for m in moves_obj]
        spec: Dict[str, Any] = {"name": str(sp), "moves": moves[:4]}
        item = getattr(mon, "item", None)
        if item:
            spec["item"] = str(item)
        ability = getattr(mon, "ability", None)
        if ability:
            spec["ability"] = str(ability)
        out[f"p1{'abcdef'[i]}"] = spec
    return out


def _own_team_from_log(lines: List[str]) -> Dict[str, Dict]:
    """DEFENSIVE FALLBACK (D74): p1's own-team spec from the log — name-only,
    because the ``|poke|`` lines carry species+gender only (no moves/item/
    ability). Used only when ``battle.team`` is unavailable, so the net still
    sees the own species rows instead of an empty own side.

    p1's own-team spec for the belief: ``slot -> {'name': species}``.

    Mirrors ``corpus.trajectories._own_team`` (deliberately re-implemented here so
    the self-play path does not import the abandoned BC ``corpus`` module). The
    belief API already supports an ``own_team`` spec; without it
    ``get_context()['own']`` is empty and the net is blind to its own bench.
    Only the species name is recoverable from the log (ability/item are hidden
    until revealed — a reveal updates the belief's posterior, not this static spec).

    ``|poke|`` team-preview lines come in two shapes (both occur; the fork emits
    the side shape):
      * side shape: ``|poke|p1|Lokix, M|``       — p1 slots a..f in list order
      * slot shape: ``|poke| p1a 1 | Kyurem |``   — explicit slot in field 2
    ``|switch`` / ``|drag`` / ``|replace`` fill any slot the preview did not name.
    """
    order = "abcdef"
    team: Dict[str, Dict] = {}
    n_side = 0
    for line in lines:
        if not isinstance(line, str):
            continue
        parts = line.split("|")
        if len(parts) < 4 or parts[1] not in ("poke", "switch", "drag", "replace"):
            continue
        key = (parts[2].strip().split() or [""])[0]
        name = parts[3].split(",")[0].strip()
        if not name:
            continue
        slot: Optional[str] = None
        if key == "p1":
            if n_side < 6:
                slot = f"p1{order[n_side]}"
                n_side += 1
        elif key.startswith("p1") and len(key) == 3 and key[2] in "abcdef":
            slot = key
        if slot is None or slot in team:
            continue
        team[slot] = {"name": name}
    return team


# ═══════════════════════════════════════════════════════════════════════════════
#  D60 legal-action list  (+ bridge to poke-env ints)
# ═══════════════════════════════════════════════════════════════════════════════
def build_action_list(battle: Any, max_actions: int = MAX_ACTIONS) -> List[Dict]:
    """The CURRENT legal actions for this agent, as D60 ``{kind, move|species}``
    dicts (the list the net's pointer head scores and the encoder emits into the
    ``action_*`` channels).

    * move decision (normal turn): the active mon's currently-legal moves; in
      gen9 each is ALSO offered as a ``tera`` variant when ``can_tera``.
    * force-switch (U-turn / Roar / ...): only switches are legal (no moves).
    * trapped (Arena Trap / Choice-adjacent): no switches.

    Legality is read straight from poke-env's battle object (``available_moves`` /
    ``available_switches`` / ``force_switch`` / ``trapped`` / ``can_tera``), so it
    matches what the server will accept. Order is deterministic (known-moves order,
    then bench order); the pointer head selects by IDENTITY, not position (D70).
    """
    acts: List[Dict] = []
    if getattr(battle, "_wait", False):
        return acts
    active = getattr(battle, "active_pokemon", None)
    force_switch = bool(getattr(battle, "force_switch", False))
    trapped = bool(getattr(battle, "trapped", False))
    can_tera = bool(getattr(battle, "can_tera", False))

    # --- moves (only when not force-switching) ---
    if active is not None and not force_switch:
        try:
            known = list(active.moves.values())[:4]
        except Exception:
            known = []
        try:
            avail = {getattr(m, "id", None)
                     for m in (getattr(battle, "available_moves", None) or [])}
        except Exception:
            avail = set()
        avail.discard(None)
        for m in known:
            mid = getattr(m, "id", None)
            if mid is None:
                continue
            if avail and mid not in avail:
                continue  # currently restricted (Encore / Taunt / PP, ...)
            acts.append({"kind": "move", "move": mid})
            if can_tera:
                acts.append({"kind": "tera", "move": mid})

    # --- switches (only when not trapped) ---
    if not trapped:
        for mon in (getattr(battle, "available_switches", None) or []):
            # the ACTUAL forme id (``species``), not the forme-stripped
            # ``base_species`` — the action_species channel and the opponent-
            # side rows both use full forme ids, so switch actions must too.
            sp = getattr(mon, "species", None) or getattr(mon, "base_species", None)
            if sp:
                acts.append({"kind": "switch", "species": sp})

    return acts[:max_actions]


def action_to_pe(act: Dict, battle: Any) -> int:
    """Bridge a D60 action dict to a poke-env-native int for ``env.step``.

    poke-env mapping (``SinglesEnv.action_to_order``): 0..5 switch (index into
    ``battle.team``), 6..9 move (index into the active mon's known moves), 22..25
    tera+move. Returns ``-2`` (default/pass) if the action is not currently
    resolvable — poke-env with ``strict=False`` then falls back to a valid move
    rather than crashing the battle.
    """
    kind = act.get("kind")
    if kind in ("move", "tera"):
        move_id = act.get("move")
        active = getattr(battle, "active_pokemon", None)
        if active is None:
            return -2
        try:
            known = list(active.moves.values())[:4]
        except Exception:
            return -2
        idx = -1
        for i, m in enumerate(known):
            if getattr(m, "id", None) == move_id:
                idx = i
                break
        if idx < 0:
            return -2
        return (22 if kind == "tera" else 6) + idx
    if kind == "switch":
        sp = act.get("species")
        try:
            team = list((getattr(battle, "team", None) or {}).values())
        except Exception:
            return -2
        for i, mon in enumerate(team):
            if getattr(mon, "base_species", None) == sp or getattr(mon, "species", None) == sp:
                return i
        return -2
    return -2


def random_action(battle: Any, rng: Optional[_random.Random] = None) -> Optional[Dict]:
    """P1 'random player': a uniformly random currently-legal action (or ``None``
    to pass/wait when there is no decision this turn). This is the stand-in policy
    until P2 wires the net in; it exercises the real legality path end to end.
    """
    acts = build_action_list(battle)
    if not acts:
        return None
    r = rng or _random
    return r.choice(acts)


# ═══════════════════════════════════════════════════════════════════════════════
#  observation fallback
# ═══════════════════════════════════════════════════════════════════════════════
def _zero_obs() -> Dict[str, np.ndarray]:
    """A valid registry-shaped all-zero observation (defensive fallback)."""
    out: Dict[str, np.ndarray] = {}
    for fld in LAYOUT:
        out[fld.name] = np.zeros(fld.shape, dtype=fld.dtype)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  the standalone observation builder  (POV-correct, reusable by a Player)
# ═══════════════════════════════════════════════════════════════════════════════
def build_obs(battle: Any, fmt: str, vocabs: Any, side: str = "p1") -> Dict[str, np.ndarray]:
    """The net's observation for ``battle`` from ``side``'s perspective.

    The SAME belief -> encoder -> registry pipeline ``ShowdownEnv.embed_battle``
    uses, factored into a free function so a poke-env ``Player`` (human-vs-bot
    play, D92) can build the EXACT observation the net was trained on, straight
    from a live ``battle`` object — no second, hand-maintained feature vector.

    ``side`` is the player's side in the ``_build_replay_log()`` protocol log
    ("p1"/"p2"). For "p2" the p1/p2 side tokens are swapped first, so the
    belief's "own" team is the real p2 team (the env's POV-correctness trick).
    Defensive: any belief/encode failure degrades to a zero observation (logged)
    so a live battle never dies on a data gap (a species with no prior, ...).
    """
    try:
        text = battle._build_replay_log()
        if side == "p2":
            text = swap_sides(text)
        lines = [ln for ln in text.split("\n") if ln.strip()]
        own_team = _own_team_from_battle(battle) or _own_team_from_log(lines)
        belief = BeliefState(fmt, own_team=own_team)
        adapter = LogAdapter()
        for line in lines:
            ev = adapter.feed_line(line)
            if ev is None or isinstance(ev, Unknown):
                continue
            belief.update(ev)
        ctx = belief.get_context()
        actions = build_action_list(battle)
        return encode_context(ctx, vocabs, actions)
    except Exception as exc:  # noqa: BLE001 — a live battle must never die on obs
        log.warning("build_obs degraded to zero obs (%s: %s)",
                    type(exc).__name__, exc)
        return _zero_obs()


def my_side(battle: Any, lines: List[str]) -> str:
    """Which side ("p1"/"p2") of the protocol log is ``battle`` (this player)
    on? The belief pipeline is p1-POV, so a bot that is really on p2 must swap.

    Detected by matching this player's team species against the ``|poke|``
    preview lines of each side — the side sharing more species is ours. Defaults
    to "p1", which is correct for the challenger (the human-vs-bot case and the
    training POV). Never raises — a detection failure must not break play.
    """
    try:
        mine = set()
        for mon in (getattr(battle, "team", None) or {}).values():
            sp = getattr(mon, "species", None) or getattr(mon, "base_species", None)
            if sp:
                mine.add(sp.lower())
        if not mine:
            return "p1"
        p1_sp, p2_sp = set(), set()
        for ln in lines:
            if not ln.startswith("|poke|"):
                continue
            # ``|poke|p1a|Garchomp|M|`` splits to ['', 'poke', 'p1a', 'Garchomp', ...]
            # so the side+slot token is parts[2] and the species is parts[3].
            parts = ln.split("|")
            if len(parts) >= 4 and parts[3]:
                (p1_sp if parts[2][:2] == "p1" else p2_sp).add(parts[3].lower())
        if len(mine & p2_sp) > len(mine & p1_sp):
            return "p2"
    except Exception:  # noqa: BLE001 — side-detect failure must never break play
        pass
    return "p1"


# ═══════════════════════════════════════════════════════════════════════════════
#  the environment
# ═══════════════════════════════════════════════════════════════════════════════
def _stop_ps_client(client: Any, timeout: float = 5.0) -> bool:
    """Stop a poke-env ``PSClient``'s websocket listener cleanly (best-effort).

    Why this exists: each ``PokeEnv`` spins up its OWN event-loop thread
    (``env._loop``), and BOTH of its ``_EnvPlayer`` agents own a ``PSClient``
    whose ``listen()`` coroutine runs on that loop. poke-env only reaps those
    at ``atexit`` (``poke_env.concurrency.__clear_loop``) — neither
    ``PokeEnv.close()`` nor ours ever stops them. So every self-play battle
    leaks a websocket + a coroutine until the process exits. If a Showdown
    server is killed FIRST, the still-live ``listen()`` dies as a raw
    ``ConnectionResetError`` (poke-env's ``listen()`` only treats a clean
    ``ConnectionClosedOK`` as non-exceptional) and logs it as a full
    traceback — that is the teardown noise.

    Calling poke-env's own ``PSClient.stop_listening()`` (which closes the
    websocket) while the server is still up lets ``listen()`` exit through the
    ``ConnectionClosedOK`` path (a one-line warning, no traceback) and frees
    the socket immediately (no accumulation over a long run).

    The clients live on the AGENTS (``agent1.ps_client`` / ``agent2.ps_client``),
    NOT on the env — ``env.ps_client`` does not exist (verified live). This
    helper takes one client and is safe to call on ``None``.

    Best-effort: returns ``True`` if a live client was asked to stop, ``False``
    if there was nothing to stop, and NEVER raises (teardown must not fail the
    battle that is being torn down).
    """
    import asyncio
    if client is None:
        return False
    if getattr(client, "websocket", None) is None:
        return False                    # never connected; nothing to close
    loop = getattr(client, "loop", None)
    stop = getattr(client, "stop_listening", None)
    if loop is None or stop is None:
        return False
    try:
        future = asyncio.run_coroutine_threadsafe(stop(), loop)
        future.result(timeout=timeout)
        return True
    except Exception:  # noqa: BLE001 — cleanup must never raise into the caller
        return False


class ShowdownEnv(SinglesEnv):
    """A concrete, instantiable singles self-play environment.

    ``SinglesEnv`` (and ``PokeEnv``) leave ``embed_battle`` / ``calc_reward``
    abstract and never assign ``observation_spaces`` / ``action_spaces``, so they
    cannot be instantiated and the gym contract is invalid. This class fills that
    gap on top of the (already working) ``PokeEnv`` battle engine, reusing the
    project's belief/encoder/registry observation pipeline (train/live
    equivalence — the net sees exactly the layout it was sized against).
    """

    def __init__(
        self,
        *,
        battle_format: str = "gen9ou",
        team: Optional[Teambuilder] = None,
        server_configuration: Optional[ServerConfiguration] = LocalhostServerConfiguration,
        save_replays: Any = False,
        log_level: Optional[int] = None,
        account_configuration1: Optional[Any] = None,
        account_configuration2: Optional[Any] = None,
        vocabs=None,
        strict: bool = False,
        **kwargs: Any,
    ):
        # strict=False: a momentarily-stale action falls back to a valid move
        # instead of raising, so a single bad pick can never abort a battle.
        super().__init__(
            account_configuration1=account_configuration1,
            account_configuration2=account_configuration2,
            battle_format=battle_format,
            team=team,
            server_configuration=server_configuration,
            save_replays=save_replays,
            log_level=log_level,
            strict=strict,
            **kwargs,
        )
        self._fmt = battle_format
        if vocabs is None:
            from net.vocab_db import build_vocabs
            vocabs = build_vocabs()
        self._vocabs = vocabs
        # gym contract (nominal here — the observation is a registry-shaped dict;
        # the loop drives reset()/step() directly and reads the dict).
        from gymnasium import spaces
        self.observation_spaces = {
            agent: spaces.Box(low=-1e9, high=1e9, shape=(total_elements(),))
            for agent in self.possible_agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(MAX_ACTIONS) for agent in self.possible_agents
        }

    # ------------------------------------------------------------------ reward
    REWARD_WIN = 1.0
    REWARD_LOSS = -1.0

    def calc_reward(self, battle: Any) -> float:
        """Sparse terminal reward: win +1 / loss -1 / 0 otherwise (zero-sum)."""
        if battle.won:
            return self.REWARD_WIN
        if battle.lost:
            return self.REWARD_LOSS
        return 0.0

    # ------------------------------------------------------------------ action
    def get_action_mask(self, battle: Any) -> np.ndarray:
        """A length-``MAX_ACTIONS`` mask over the CURRENT legal-action list
        (1.0 for a real action, 0.0 for a pad slot). Consistent with the
        registry's ``action_mask`` channel and the pointer-head design."""
        n = len(build_action_list(battle))
        mask = np.zeros(MAX_ACTIONS, dtype=np.float32)
        mask[:n] = 1.0
        return mask

    def get_action_space_size(self, gen: int) -> int:
        return MAX_ACTIONS

    # ------------------------------------------------------------------ obs
    def embed_battle(self, battle: Any) -> Dict[str, np.ndarray]:
        """The observation this agent sees, from ITS perspective.

        Built by the SAME belief -> encoder -> registry pipeline the net's
        embedding tables are sized against. POV-correct: for the p2 agent the
        protocol log's p1/p2 tokens are swapped first so "own" is the real p2
        team. Defensive: any belief/encode failure degrades to a zero
        observation (logged) so a live battle / training loop never crashes on a
        data gap (e.g. a species with no prior in the format).
        """
        side = "p2" if battle is self.battle2 else "p1"
        return build_obs(battle, self._fmt, self._vocabs, side)

    # ------------------------------------------------------------------ teardown
    def close(self, force: bool = True, wait: bool = True):
        """Tear down WITHOUT ever sending a forfeit.

        poke-env's default ``close()`` sends a ``ForfeitBattleOrder`` for any
        battle not yet finished — the sole source of the spurious
        "X forfeited / win Y" replays. A forfeit is a harness artifact, not an
        action the policy can take, so we abandon the live battle cleanly instead
        (no forfeit, no blocking wait on a still-pending challenge task). When the
        battle has genuinely finished (the normal end-of-run path),
        ``reset_battles()`` is safe and the queue drain is a no-op.
        """
        if self._challenge_task is not None and self._challenge_task.done():
            self.reset_battles()
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
        # stop EACH agent's websocket listener cleanly (see _stop_ps_client):
        # poke-env never does this (each PokeEnv spins its own loop thread and
        # only reaps it at atexit), so without it every battle leaks a socket +
        # a coroutine (and, when a server is killed first, the
        # ConnectionResetError teardown noise). The clients live on the agents,
        # not on the env — self.ps_client does not exist (verified live).
        for agent in (getattr(self, "agent1", None), getattr(self, "agent2", None)):
            _stop_ps_client(getattr(agent, "ps_client", None))


# ═══════════════════════════════════════════════════════════════════════════════
#  team building  (reuses the server-verified bot/team_pools.json data)
# ═══════════════════════════════════════════════════════════════════════════════
_STAT_IDX = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}


def _mon_to_tb(mon: Dict) -> TeambuilderPokemon:
    """A mon-dict (from team_pools.json) -> a poke-env TeambuilderPokemon."""
    evs = list(mon.get("evs") or [0, 0, 0, 0, 0, 0])
    ivs = list(mon["ivs"]) if mon.get("ivs") else [31] * 6
    tb = TeambuilderPokemon(
        species=mon.get("species"),
        item=mon.get("item") or None,
        ability=mon.get("ability") or None,
        moves=list(mon.get("moves") or []),
        nature=mon.get("nature") or None,
        evs=evs,
        ivs=ivs,
    )
    if mon.get("tera_type"):
        tb.tera_type = mon["tera_type"]
    return tb


class StrongTeamBuilder(Teambuilder):
    """Yields one coherent curated 6-mon team per ``yield_team()`` call.

    Reads the server-verified pools from ``bot/team_pools.json`` (reusable data —
    every team was accepted by the local fork's own TeamValidator at build time, so
    they are guaranteed legal). poke-env calls ``yield_team()`` once per agent, so
    each side independently draws a full coherent team.
    """

    def __init__(self, battle_format: str = "gen9ou",
                 pool_path: Optional[str] = None,
                 seed: Optional[int] = None):
        import json
        import os
        if pool_path is None:
            here = os.path.dirname(os.path.abspath(__file__))
            pool_path = os.path.join(here, os.pardir, "bot", "team_pools.json")
        pool_path = os.path.normpath(pool_path)
        if not os.path.exists(pool_path):
            raise RuntimeError(f"no team pools at {pool_path}")
        with open(pool_path, encoding="utf8") as fh:
            pools = json.load(fh)
        entry = pools.get(battle_format)
        if not isinstance(entry, dict) or not entry.get("teams"):
            raise RuntimeError(f"no coherent teams for '{battle_format}' in {pool_path}")
        self._teams = entry["teams"]
        self._rng = _random.Random(seed)

    def yield_team(self) -> str:
        team = self._rng.choice(self._teams)
        return Teambuilder.join_team([_mon_to_tb(m) for m in team])
