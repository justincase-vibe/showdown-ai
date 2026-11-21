"""M2a — the WorldModel bridge: the real ``PokeNet`` speaks the MCTS protocol.

WHY THIS EXISTS
---------------
M1 (``net/mcts.py``) is a net-agnostic PUCT search core. It needs a
:class:`~net.mcts.WorldModel` that can (a) list the legal actions at a state,
(b) score them (a policy prior), (c) evaluate a leaf (a value), and (d) branch
(``step``). This module provides that bridge for REAL Pokémon, wrapping
``net.model.PokeNet`` (through ``PokeAgent.forward_obs`` — the ONE forward
path) with the belief ``get_context()`` dict as the state.

M2a = the NON-BRANCHING adapters (``actions`` / ``value`` / ``policy``),
verified against the real net on a realistic belief context (CPU-only,
random-init net). ``step`` — the branching — is M2b and raises
``NotImplementedError`` until then. That keeps M2a free of ANY effect
bookkeeping / second source of truth (the handoff's explicit warning).

THE STATE
---------
A :class:`State` is the canonical belief context (``BeliefState.get_context()``,
always **p1-POV**) + a ``decider`` ("p1"/"p2") = who moves next. The canonical
context is the single shared belief; the ``decider`` tells the WorldModel whose
side to act/evaluate for. The root of a search is ``decider="p1"`` (the deciding
player, matching the net's p1-POV training).

PERSPECTIVE CONTRACT (the crux — see OPEN_DESIGN_CHOICES.md OC-7)
------------------------------------------------------------------
The net is a **p1-decider**: its value head outputs the *deciding player's*
win probability and its policy head acts for the *deciding player* (both read
from the obs's POV). MCTS, however, needs every value from the **ROOT**
player's perspective (M1's ``_backup`` sign-flips from there). So:

  * ``value(state)`` runs the head on the *decider's* POV (keeping every value
    head call IN-DISTRIBUTION), then maps to the root perspective:
    ``+(2V−1)`` if the decider IS the root, else ``−(2V−1)``.
  * ``policy(state, actions)`` always runs on the decider's POV over the
    decider's own legal set.

M2a only ever sees ``decider="p1"`` (the root): without ``step`` the search
cannot reach a p2-turn node. The p2-POV orientation (the field-aware
own<->opponent reflection + slot remap) is M2b — it is exercised and tested
once ``step`` actually produces p2 nodes. Shipping that fiddly swap now,
untested by the real search flow, is exactly the "untested fiddly code" risk
we avoid (see OC-7: the decider-aware design is the direction; M2a is its
correct, verifiable root case).

NO SECOND FORWARD PATH
----------------------
``value`` / ``policy`` both go through ``PokeAgent.forward_obs`` — the SAME
batch-build + forward ``net/agent.py`` uses for play — so there is ONE
net-forward code path (the "no code written twice" rule).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .agent import PokeAgent
from .encoder import encode_context


@dataclass(frozen=True)
class State:
    """One MCTS state = the canonical (p1-POV) belief context + who moves next.

    ``ctx`` is the ``BeliefState.get_context()`` dict (JSON-safe, p1-POV).
    ``decider`` is "p1" or "p2" — the player who acts at this node. The root of
    a search is "p1" (the deciding player). M2b's ``step`` will produce a new
    ``State`` (a new ctx, the flipped decider). Frozen so a node's state is
    immutable (MCTS never mutates a state in place).
    """
    ctx: Dict
    decider: str = "p1"


def _orient(ctx: Dict, decider: str) -> Dict:
    """Return ``ctx`` oriented to ``decider``'s POV.

    M2a: ``"p1"`` (the canonical root orientation) -> ``ctx`` as-is.
    ``"p2"`` is the opponent-reply case and is **M2b**: it needs the field-aware
    own<->opponent reflection + slot remap (own schema: name/ability/item;
    opponent schema: species/posterior/top_moves). It is exercised and tested
    once ``step`` actually produces p2-turn nodes. Raising here keeps M2a honest
    and avoids shipping an untested, fiddly swap into the search path.
    """
    if decider == "p1":
        return ctx
    raise NotImplementedError(
        f"PokeWorldModel: decider={decider!r} (opponent-POV) is M2b — paired with "
        "step and exercised once step produces p2-turn nodes. M2a covers the root "
        "(p1-POV) case only. See OPEN_DESIGN_CHOICES.md (OC-7).")


def _actions_from_own(ctx: Dict) -> List[Dict]:
    """The decider's legal actions (moves + bench switches), in the SAME order
    the net was trained on (``net/env.build_action_list``): the active mon's
    known moves first, then the alive benched mons (bench order).

    Reads the canonical p1-POV context's own side. The active mon is the slot
    in ``ctx["active"]["p1"]`` (the encoder derives the per-slot active flag
    from it, not from a per-block field). Tera variants are omitted (OC-8: the
    context does not yet track "tera-used," and a multi-ply search would
    otherwise tera repeatedly).
    """
    own = ctx.get("own", {}) or {}
    active_slot = (ctx.get("active", {}) or {}).get("p1")

    acts: List[Dict] = []
    if active_slot is not None:
        blk = own.get(active_slot) or {}
        for m in blk.get("moves", []) or []:
            acts.append({"kind": "move", "move": m})
    for slot in ("p1a", "p1b", "p1c", "p1d", "p1e", "p1f"):
        blk = own.get(slot)
        if not blk:
            continue
        if slot == active_slot:
            continue              # the active mon is not a switch target
        if blk.get("fainted"):
            continue              # a gone mon is not switchable bench
        name = blk.get("name") or blk.get("species") or ""
        if name:
            acts.append({"kind": "switch", "species": name})
    return acts


class PokeWorldModel:
    """M2a — the real :class:`PokeNet` as an MCTS :class:`~net.mcts.WorldModel`.

    ``actions`` / ``value`` / ``policy`` are live (the non-branching bridge);
    ``step`` is M2b (raises). ``agent`` is a ``PokeAgent`` (it owns the net and
    the ONE forward path); ``vocabs`` builds the obs from a context.
    """

    def __init__(self, agent: PokeAgent, vocabs, root_decider: str = "p1") -> None:
        self.agent = agent
        self.vocabs = vocabs
        self.root_decider = root_decider

    # ── ONE forward path (value + policy share it — no second batch builder) ──
    def _forward(self, ctx: Dict, actions: List[Dict]):
        obs = encode_context(ctx, self.vocabs, actions)
        return self.agent.forward_obs(obs)

    # ── the WorldModel protocol (net/mcts.py) ──────────────────────────────
    def actions(self, state: State) -> List[Dict]:
        """The legal actions at ``state`` (the D60 list the net scores over)."""
        return _actions_from_own(_orient(state.ctx, state.decider))

    def value(self, state: State) -> float:
        """The value-head leaf eval for ``state``, in (−1, +1), from the MCTS
        ROOT player's perspective (the contract M1's ``_backup`` relies on)."""
        ctx = _orient(state.ctx, state.decider)
        acts = _actions_from_own(ctx)
        out = self._forward(ctx, acts)
        v = float(out["value"][0, 0, 0].item())           # (0,1) decider-POV
        sign = 1.0 if state.decider == self.root_decider else -1.0
        return sign * (2.0 * v - 1.0)

    def policy(self, state: State, actions: List[Dict]) -> List[float]:
        """The policy-head priors over ``actions`` (the masked distribution, so
        they sum to ~1 over the legal set). MCTS re-normalizes either way."""
        ctx = _orient(state.ctx, state.decider)
        out = self._forward(ctx, list(actions))
        prob = out["prob"][0, 0].detach().cpu().numpy()
        return [float(prob[i]) for i in range(len(actions))]

    def step(self, state: State, action: Dict) -> Tuple["State", float, bool]:
        """Apply ``action`` -> (next_state, reward_from_root_view, done).

        M2b: apply the decider's move to a copied context — damage via the ONE
        ``damage/calc`` (never re-derived) + minimal HP/switch/faint
        bookkeeping — gated by the world-model-vs-sim agreement test so the sim
        stays the ground truth. Not built in M2a (see OPEN_DESIGN_CHOICES OC-6).
        """
        raise NotImplementedError(
            "PokeWorldModel.step is M2b (branching + the p2-POV orientation). "
            "M2a is the non-branching bridge only. See OPEN_DESIGN_CHOICES.md "
            "(OC-6/OC-8).")
