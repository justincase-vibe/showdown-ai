"""A3.1 + A3.2 — the ``HeuristicAgent``: a strong, DETERMINISTIC, CPU-first policy.

It is a ``PokeAgent``-compatible drop-in — ``.act(obs, acts, deterministic)``
returns a ``Decision``-shaped object — so it flows through the SAME
``league.driver.policy_action`` path the net uses. The harness therefore plays
exactly what the heuristic does: no second decision pipeline, no code twice.

Design (CPU-first / single source of truth):
* Decodes structured state from ``obs`` using the S4.1 **registry** indices
  (``own_slot`` / ``opp_slot`` active+hp, ``own_species`` / ``opp_species``) —
  never hand-maintained column numbers.
* Static move / species / type data from the belief state's OWN layer:
  ``legality.lookup.get_move`` / ``get_species`` / ``get_type_chart`` and the
  shared ``net.vocab`` species index — no second move table, no second type chart.
* **Lazy imports** (like ``fitness.ladder.random_reference``): importing this
  module pulls NO torch; the registry / vocab / legality DB are touched only
  inside ``.act`` (the module stays CPU-first; torch is only imported when a
  battle is actually played — the same discipline as the rest of ``fitness/``).

Behaviour (legal + sensible; the >70%-beats-random gate is A3.3's self-check):
* empty ``acts``                          -> pass (``action=None``)
* no moves legal (force-switch)           -> pick a switch
* own active HP <= ``switch_threshold`` and a switch is legal -> switch
* else                                    -> the best damaging move by
  ``A3.2``  score = power * type_effectiveness(move_type, opp_types)
                    * (1.5 if STAB else 1.0)
  (status / unknown-power moves score 0, ranked last; unknown types degrade to
  pure power — the A3.1 behaviour — so it never crashes on a partial ``obs``).

Deterministic: identical ``(obs, acts)`` always yields the identical action —
a strong reference should be a reproducible opponent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

__all__ = ["HeuristicAgent", "Decision"]

#: per-gen cached type data (chart + species vocab + lazy species->types map).
_TYPE_DATA: Dict[int, Dict[str, Any]] = {}


@dataclass
class Decision:
    """A ``net.agent.Decision``-SHAPED return (duck-typed by the driver).

    Defined here (not imported from ``net.agent``) so ``import fitness.heuristic``
    stays torch-free. The play path reads only ``.action`` (driver) and
    ``.action_index`` / ``.logprob`` / ``.value`` (``league.collect``); ``.prob`` /
    ``.entropy`` are present for shape-consistency with the net's Decision.
    """
    action: Optional[Dict]
    action_index: int
    logprob: float
    value: float
    entropy: float
    prob: np.ndarray


class HeuristicAgent:
    """A deterministic rule-based opponent for the fitness ladder.

    ``gen`` is the @smogon/calc ENGINE KEY passed to the legality lookups
    (0 = champions, 1-9 = gen) so move power / types are correct per format.
    Defaults to 9 (the primary format). ``seed`` is accepted to satisfy the
    ``Reference.make_agent(seed)`` contract but is unused — the policy is fully
    deterministic.
    """

    def __init__(self, gen: int = 9, switch_threshold: float = 0.30,
                 seed: Optional[int] = None) -> None:
        self.gen = int(gen)
        self.switch_threshold = float(switch_threshold)
        self.seed = seed

    # ── type data (cached per gen; lazily built on first use) ────────────
    def _type_data(self) -> Dict[str, Any]:
        d = _TYPE_DATA.get(self.gen)
        if d is not None:
            return d
        from net.vocab_db import build_vocabs
        from legality.lookup import get_type_chart
        d = {
            "chart": get_type_chart(self.gen),      # {def_type: {atk_type: factor}}
            "species": build_vocabs().species,      # Vocab: int idx -> id
            "species_types": {},                    # id -> tuple(display types)
        }
        _TYPE_DATA[self.gen] = d
        return d

    def _types_of(self, obs: Any, side: str) -> Tuple[str, ...]:
        """The active mon's display types for ``side`` ('own'/'opp'), or () if
        undecodable (degrades gracefully to "unknown" -> no STAB / neutral eff)."""
        if not isinstance(obs, dict):
            return ()
        slot = obs.get(f"{side}_slot")
        species = obs.get(f"{side}_species")
        if slot is None or species is None:
            return ()
        slot = np.asarray(slot)
        species = np.asarray(species)
        if slot.ndim != 2:
            return ()
        try:
            from net.registry import SLOT_INDEX
            active_i = SLOT_INDEX["active"]
        except Exception:
            return ()
        if slot.shape[1] <= active_i:
            return ()
        n = min(slot.shape[0], species.shape[0])
        active_rows = [i for i in range(n) if float(slot[i, active_i]) > 0.5]
        if not active_rows:
            return ()
        a = active_rows[0]
        idx = int(species[a])
        if idx <= 0:
            return ()
        d = self._type_data()
        sid = d["species"].id_of(idx)
        if not sid:
            return ()
        if sid not in d["species_types"]:
            from legality.lookup import get_species
            try:
                d["species_types"][sid] = tuple(get_species(sid, self.gen).get("types") or ())
            except Exception:
                d["species_types"][sid] = ()
        return d["species_types"][sid]

    def _effectiveness(self, move_type: str, defender_types: Tuple[str, ...]) -> float:
        """Product of the per-defending-type factors (0/0.5/1/2), or 1.0 unknown."""
        if not move_type or not defender_types:
            return 1.0
        chart = self._type_data()["chart"]
        factor = 1.0
        for dt in defender_types:
            row = chart.get(dt)
            if row and move_type in row:
                factor *= row[move_type]
        return float(factor)

    # ── obs decoding (defensive: any missing/malformed field just degrades) ──
    def _my_active_hp(self, obs: Any) -> Optional[float]:
        """The active own-mon's HP fraction in [0,1], or None if undecodable."""
        slot = obs.get("own_slot") if isinstance(obs, dict) else None
        if slot is None:
            return None
        slot = np.asarray(slot)
        if slot.ndim != 2:
            return None
        try:
            from net.registry import SLOT_INDEX
            active_i, present_i, hp_i = (SLOT_INDEX["active"],
                                        SLOT_INDEX["present"],
                                        SLOT_INDEX["hp_frac"])
        except Exception:
            return None
        if slot.shape[1] <= max(active_i, present_i, hp_i):
            return None
        active_rows = [i for i in range(slot.shape[0]) if float(slot[i, active_i]) > 0.5]
        if not active_rows:
            return None
        a = active_rows[0]
        if float(slot[a, present_i]) < 0.5:
            return None
        return float(slot[a, hp_i])

    # ── move scoring (the belief state's own lookups; no second table) ──
    def _move_score(self, act: Dict, own_types, opp_types) -> float:
        """``power * effectiveness(move_type, opp) * (1.5 if STAB else 1.0)``.

        Status / unknown-power moves score 0 (ranked last). Unknown types simply
        yield effectiveness 1.0 / no STAB, i.e. pure power (the A3.1 behaviour)."""
        mid = act.get("move")
        if not mid:
            return 0.0
        try:
            from legality.lookup import get_move
            m = get_move(mid, self.gen)
        except Exception:  # unknown move -> rank last, but still legal
            return 0.0
        if (m.get("category") or "").lower() == "status":
            return 0.0
        power = float(m.get("power") or 0)
        if power <= 0:
            return 0.0
        mt = m.get("type") or ""
        eff = self._effectiveness(mt, opp_types)
        stab = 1.5 if (mt and mt in own_types) else 1.0
        return power * eff * stab

    # ── A3.2b: hazard-aware switch target ─────────────────────────────
    def _threatening_hazards(self, obs: Any) -> Dict[str, bool]:
        """Hazards on the OPPONENT's side (p2) that hit MY switch-ins.

        The obs is always built for the acting agent, so p1 = me and p2 = the
        opponent; hazards the opponent set (p2-side) damage my switch-ins. Only
        type-interacting hazards are scored (Spikes/Sticky Web don't depend on
        type, so they're intentionally ignored here)."""
        g = obs.get("global") if isinstance(obs, dict) else None
        if g is None:
            return {}
        g = np.asarray(g)
        if g.ndim != 1:
            return {}
        try:
            from net.registry import GLOBAL_INDEX
            sr = GLOBAL_INDEX["stealthrock_p2"]
            ts = GLOBAL_INDEX["toxicspikes_p2"]
        except Exception:
            return {}
        if g.shape[0] <= max(sr, ts):
            return {}
        return {"stealthrock": bool(float(g[sr]) > 0.0),
                "toxicspikes": bool(float(g[ts]) > 0.0)}

    def _switch_score(self, act: Dict, hazards: Dict[str, bool]) -> float:
        """A switch-in's hazard safety: 0.0 if safe, <0 if weak to an active
        hazard. Higher is better."""
        if not hazards:
            return 0.0
        sp = act.get("species")
        if not sp:
            return 0.0
        try:
            from legality.lookup import get_species
            types = set(get_species(sp, self.gen).get("types") or ())
        except Exception:
            return 0.0
        penalty = 0.0
        if hazards.get("stealthrock") and (types & {"Rock", "Steel", "Flying", "Bug"}):
            penalty += 2.0
        if hazards.get("toxicspikes") and (types & {"Poison", "Flying"}):
            penalty += 1.0
        return -penalty

    def _pick_switch(self, acts: List[Dict], switches: List[Dict],
                     obs: Any) -> Dict:
        """The best legal switch (hazard-aware when hazards threaten my
        switch-ins). Falls back to the first legal switch when there's only one
        or no hazards are up (the A3.1 behaviour)."""
        if len(switches) == 1:
            return switches[0]
        hazards = self._threatening_hazards(obs)
        if not hazards:
            return switches[0]
        return max(switches, key=lambda a: self._switch_score(a, hazards))

    # ── the policy ─────────────────────────────────────────────────────
    def act(self, obs: Any, action_list: Optional[List[Dict]],
            deterministic: bool = False) -> Decision:
        """Choose one legal action (deterministic). ``deterministic`` is accepted
        for contract parity but the policy is always deterministic."""
        acts: List[Dict] = list(action_list or [])
        if not acts:
            return self._decision(acts, -1)

        moves = [a for a in acts if a.get("kind") in ("move", "tera")]
        switches = [a for a in acts if a.get("kind") == "switch"]

        # 1) force-switch: only switches are legal.
        if not moves:
            return self._decision(acts, acts.index(self._pick_switch(acts, switches, obs)))

        # 2) critically low + can switch -> bail before we faint for nothing
        #    (A3.2b: pick the switch-in least hurt by the opponent's hazards).
        hp = self._my_active_hp(obs)
        if switches and hp is not None and hp <= self.switch_threshold:
            return self._decision(acts, acts.index(self._pick_switch(acts, switches, obs)))

        # 3) play the best damaging move (A3.2: power * effectiveness * STAB).
        try:
            own_types = self._types_of(obs, "own")
            opp_types = self._types_of(obs, "opp")
        except Exception:  # never let a type-lookup glitch kill a decision
            own_types, opp_types = (), ()
        best = max(moves, key=lambda a: self._move_score(a, own_types, opp_types))
        return self._decision(acts, acts.index(best))

    # ── decision packing ─────────────────────────────────────────────────
    def _decision(self, acts: List[Dict], idx: int) -> Decision:
        if idx < 0:  # a pass/wait turn
            return Decision(None, -1, 0.0, 0.5, 0.0,
                            np.zeros(0, dtype=np.float32))
        prob = np.zeros(len(acts), dtype=np.float32)
        prob[idx] = 1.0  # a deterministic one-hot (logprob 0.0, entropy 0.0)
        return Decision(acts[idx], idx, 0.0, 0.5, 0.0, prob)


def heuristic_reference(gen: int = 9) -> "Reference":  # noqa: F821 (lazy)
    """Build the mid-ladder ``heuristic`` reference (weight 2, above the random
    floor). Lazy so this import stays cheap; mirrors ``random_reference``.
    """
    from .ladder import Reference

    def make_agent(seed: Optional[int] = None) -> HeuristicAgent:
        return HeuristicAgent(gen=gen, seed=seed)

    return Reference(
        name="heuristic", make_agent=make_agent, weight=2.0,
        description="deterministic rule-based opponent (type-aware move scoring "
                    "+ switch-when-low); a strong bot beats this most of the time")
