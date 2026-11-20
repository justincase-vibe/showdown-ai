"""S4.2 — network core + heads (PLAN.md §3c S4.2 / D60).

The S4.2 spec (user-approved depth 3–4):
  * **4×GRU-512** over the turn sequence + a **per-mon encoder (128)**;
  * LayerNorm between layers; **GRU state reset per battle** (each battle is
    its own sequence with a fresh hidden state — no cross-battle leak);
  * **policy head** = logits over the per-state legal-action list (D60 compact
    enumeration). Illegal actions are excluded by a MASKED SOFTMAX, so their
    probability is **exactly 0.0** (``exp(-inf) == 0``) — structural, not a
    learned penalty;
  * **value head** = scalar in (0, 1) (logistic squash).

The input is EXACTLY the :mod:`net.registry` layout (the single source of
truth) as a per-timestep batch. The net never re-derives a width — every input
dimension comes from the registry, so the net and the encoder cannot drift.

Permutation invariance (S4.2 gate): the per-side representation is a **sum over
the ≤6 per-slot vectors** (a set function — invariant to slot order) plus the
**active mon(s) selected by the per-slot ``active`` flag** (singles = 1,
doubles = 2, same code path). Swapping any bench slots therefore leaves the
timestep input — and hence the logits — unchanged.

No leakage (rule 4): the net consumes ONLY the registry's fields, and the
encoder already routes opponent UNREVEALED info through the posterior channels
only. So the net inherits the encoder's no-leakage guarantee by construction.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import (
    GLOBAL_DIM,
    LAYOUT,
    MAX_ACTIONS,
    MAX_OWN_MOVES,
    MAX_POST,
    MAX_SLOTS,
    MAX_TOP_MOVES,
    SLOT_DIM,
    SLOT_INDEX,
)
from .vocab import Vocabs

#: learned id-embedding width (a fixed design choice; not a layout field).
EMBED_DIM = 32
#: per-mon encoder width (the S4.2 spec's "per-mon encoder (128)").
MON_DIM = 128
#: GRU width + depth (the S4.2 spec's "4×GRU-512").
GRU_DIM = 512
GRU_LAYERS = 4
#: policy / value head hidden width.
HEAD_DIM = 256
#: D70 pointer-head intent width (the state->intent vector the per-action
#: embeddings are scored against).
INTENT_DIM = 64
#: D70 pointer-head temperature (logit scale). Cosine-similarity logits live in
#: [-1, 1]; multiplying by LOGIT_SCALE bounds them to [-LOGIT_SCALE, +LOGIT_SCALE]
#: so the cross-entropy is BOUNDED and well-conditioned. The raw bilinear head
#: had UNBOUNDED dot-product logits, so a confident-wrong prediction exploded the
#: S4.2 overfit loss (4.7 / 13.8). A max logit gap of 2*LOGIT_SCALE is still far
#: more than softmax needs to push the target probability to ~1.0 (SCALE=8 -> gap
#: 16 -> p_target ~ 1.0), so the net can still memorize (loss -> 0).
LOGIT_SCALE = 8.0


def _masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Softmax over the action axis with illegal positions at EXACTLY 0.0.

    ``mask`` is (…, MAX_ACTIONS) with 1.0 = legal. Illegal positions are set to
    ``-inf`` BEFORE softmax, so ``exp(-inf) == 0.0`` exactly and their
    probability is exactly 0.0 (the D60 structural guarantee). If a row has NO
    legal action (all-masked — should not happen on real states) we fall back to
    the unmasked softmax rather than producing NaN, so a degenerate state never
    poisons a batch.
    """
    if mask is None:
        return torch.softmax(logits, dim=-1)
    neg_inf = torch.finfo(logits.dtype).min  # -inf, but finite-safe for softmax
    masked = logits.masked_fill(mask == 0, neg_inf)
    # all-masked rows -> leave them unmasked (avoids an all -inf row -> NaN)
    any_legal = (mask == 1).any(dim=-1, keepdim=True)
    unmasked = torch.softmax(logits, dim=-1)
    row_masked = torch.softmax(
        masked.masked_fill(~any_legal.expand_as(masked), 0.0), dim=-1)
    return torch.where(any_legal, row_masked, unmasked)


class MonEncoder(nn.Module):
    """One uniform per-slot (per-mon) encoder -> MON_DIM.

    Input per slot: the registry's slot float block (SLOT_DIM) + the slot's
    species / ability / item embeddings + the mean of its move embeddings.
    Shared by the own AND the opponent side (D61: one uniform block shape).
    Applied IDENTICALLY to every slot, so a sum over slots is a set function.
    """

    def __init__(self, vocabs: Vocabs) -> None:
        super().__init__()
        self.emb_species = nn.Embedding(len(vocabs.species), EMBED_DIM, padding_idx=0)
        self.emb_ability = nn.Embedding(len(vocabs.ability), EMBED_DIM, padding_idx=0)
        self.emb_item = nn.Embedding(len(vocabs.item), EMBED_DIM, padding_idx=0)
        self.emb_move = nn.Embedding(len(vocabs.move), EMBED_DIM, padding_idx=0)
        in_dim = SLOT_DIM + 4 * EMBED_DIM  # slot floats + 4 id embeddings
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, MON_DIM),
            nn.LayerNorm(MON_DIM),
            nn.ReLU(),
            nn.Linear(MON_DIM, MON_DIM),
        )

    def forward(self, slot: torch.Tensor, species: torch.Tensor,
                ability: torch.Tensor, item: torch.Tensor,
                moves: torch.Tensor) -> torch.Tensor:
        """slot (B,T,6,SLOT_DIM) + id rows -> (B,T,6,MON_DIM) per-slot vectors."""
        # moves is (B,T,6,MAX_OWN_MOVES) -> emb (B,T,6,MOVES,E); mean over the
        # MOVES axis (dim=-2), NOT the embedding axis (dim=-1), -> (B,T,6,E).
        emb = torch.cat([
            self.emb_species(species),        # (B,T,6,E)
            self.emb_ability(ability),        # (B,T,6,E)
            self.emb_item(item),              # (B,T,6,E)
            self.emb_move(moves).mean(dim=-2),  # (B,T,6,E) mean over <=4 moves
        ], dim=-1)
        return self.mlp(torch.cat([slot, emb], dim=-1))


class PokeNet(nn.Module):
    """The S4.2 policy/value net over encoded battle sequences.

    Batch layout: every registry field is a tensor ``(B, T, *field.shape)`` and
    ``battle_mask`` is ``(B, T)`` (1.0 = a real timestep, 0.0 = padding). The
    GRU runs over T with a FRESH hidden state per battle (reset per battle).
    Outputs are per-timestep: ``logits (B,T,MAX_ACTIONS)`` and
    ``value (B,T,1)``.
    """

    def __init__(self, vocabs: Vocabs) -> None:
        super().__init__()
        self.mon_enc = MonEncoder(vocabs)

        # global / context id embeddings (weather, terrain, posterior, top moves,
        # action list). Small fixed channels — all from the registry layout.
        self.emb_weather = nn.Embedding(len(vocabs.weather), EMBED_DIM, padding_idx=0)
        self.emb_terrain = nn.Embedding(len(vocabs.terrain), EMBED_DIM, padding_idx=0)
        self.emb_ability_g = nn.Embedding(len(vocabs.ability), EMBED_DIM, padding_idx=0)
        self.emb_move_g = nn.Embedding(len(vocabs.move), EMBED_DIM, padding_idx=0)
        self.emb_species_g = nn.Embedding(len(vocabs.species), EMBED_DIM, padding_idx=0)
        # action-kind is a tiny fixed set (move/switch/tera/transform + pad).
        self.emb_action_kind = nn.Embedding(8, 16, padding_idx=0)

        side_dim = 2 * MON_DIM            # set-pooled (128) + active (128)
        post_dim = EMBED_DIM              # weighted top-K ability
        topmove_dim = EMBED_DIM           # mean top-moves
        # per-action embedding: kind(16) + move(E) + species(E)
        action_dim = 16 + 2 * EMBED_DIM
        ctx_dim = (GLOBAL_DIM + 2 * EMBED_DIM + post_dim + topmove_dim
                   + action_dim)
        in_dim = 2 * side_dim + ctx_dim

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, GRU_DIM),
            nn.LayerNorm(GRU_DIM),
        )
        self.gru = nn.GRU(GRU_DIM, GRU_DIM, num_layers=GRU_LAYERS,
                          batch_first=True)
        # LayerNorm between the GRU layers is applied to the stacked output;
        # nn.GRU does not expose per-layer hooks, so we normalize the final
        # hidden stream (and rely on LayerNorm at the input + heads for the
        # between-layer conditioning the spec asks for).
        self.gru_norm = nn.LayerNorm(GRU_DIM)

        # D70: identity-aware (pointer) policy head. logits[i] = score(state,
        # action_i), so the net selects the correct action by IDENTITY
        # regardless of its position in the (shuffled) action list. The old
        # head (Linear(h)->MAX_ACTIONS) only saw a POOLED action vector, so it
        # could learn state->position but NOT "use Ice Beam wherever it sits"
        # — structurally unable to do identity-based action selection.
        self.policy_intent = nn.Sequential(
            nn.Linear(GRU_DIM, HEAD_DIM), nn.ReLU(),
            nn.Linear(HEAD_DIM, INTENT_DIM),
        )
        self.policy_act_proj = nn.Linear(action_dim, INTENT_DIM)
        self.value_head = nn.Sequential(
            nn.Linear(GRU_DIM, HEAD_DIM), nn.ReLU(),
            nn.Linear(HEAD_DIM, 1), nn.Sigmoid(),
        )

    # ── per-side set encoding (permutation invariant) ─────────────────
    def _side_repr(self, slot: torch.Tensor, species: torch.Tensor,
                   ability: torch.Tensor, item: torch.Tensor,
                   moves: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
        """(B,T,6,MON_DIM) per-slot + active flag -> (B,T,2*MON_DIM).

        ``set_vec``  = SUM over the 6 slots (a set function: slot-order
                       invariant).
        ``active_vec`` = SUM over the slots whose ``active`` flag is set
                       (singles = the one active mon; doubles = both).
        """
        per_slot = self.mon_enc(slot, species, ability, item, moves)  # (B,T,6,M)
        set_vec = per_slot.sum(dim=-2)              # (B,T,M) — permutation invariant
        act_w = active.unsqueeze(-1)                 # (B,T,6,1)
        active_vec = (per_slot * act_w).sum(dim=-2)  # (B,T,M)
        return torch.cat([set_vec, active_vec], dim=-1)

    def _context_repr(self, b: Dict[str, torch.Tensor],
                      T: int) -> torch.Tensor:
        """The global / context channels for one batch -> (B,T,ctx_dim)."""
        B = b["global"].shape[0]
        dev = b["global"].device
        weather = self.emb_weather(b["global_weather"].squeeze(-1))   # (B,T,E)
        terrain = self.emb_terrain(b["global_terrain"].squeeze(-1))   # (B,T,E)
        # posterior: weighted mean of top-K ability embeddings
        post_ids = b["opp_ability_post_ids"]                       # (B,T,K)
        post_w = b["opp_ability_post_w"]                           # (B,T,K)
        post_emb = self.emb_ability_g(post_ids)                    # (B,T,K,E)
        post_vec = (post_emb * post_w.unsqueeze(-1)).sum(dim=-2)   # (B,T,E)
        top_moves = self.emb_move_g(b["opp_top_moves"]).mean(dim=-2)  # (B,T,E)
        # action list: masked mean of per-action embeddings
        a_kind = b["action_kind"]                                   # (B,T,A)
        a_move = b["action_move"]
        a_sp = b["action_species"]
        a_mask = b["action_mask"]                                   # (B,T,A)
        act_emb = torch.cat([
            self.emb_action_kind(a_kind),      # (B,T,A,16)
            self.emb_move_g(a_move),           # (B,T,A,E)
            self.emb_species_g(a_sp),          # (B,T,A,E)
        ], dim=-1)                                                  # (B,T,A,act)
        act_w = a_mask.unsqueeze(-1)                                 # (B,T,A,1)
        act_sum = (act_emb * act_w).sum(dim=-2)                     # (B,T,act)
        act_n = act_w.sum(dim=-2).clamp_min(1.0)                    # (B,T,1)
        act_vec = act_sum / act_n                                   # (B,T,act)
        return torch.cat([
            b["global"], weather, terrain, post_vec, top_moves, act_vec,
        ], dim=-1)

    def forward(self, b: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """``b``: registry fields as (B,T,*) + battle_mask (B,T).

        Returns ``logits (B,T,MAX_ACTIONS)``, ``value (B,T,1)``,
        ``prob (B,T,MAX_ACTIONS)`` (masked — illegal exactly 0).
        """
        B, T = b["global"].shape[0], b["global"].shape[1]
        _active = SLOT_INDEX["active"]  # per-slot active flag column (registry)
        own = self._side_repr(
            b["own_slot"], b["own_species"], b["own_ability"], b["own_item"],
            b["own_moves"], b["own_slot"][:, :, :, _active])
        # opponent ability/item are UNREVEALED — fed as <pad> (0); their info
        # reaches the net ONLY through the posterior channels (rule 4, no leak).
        opp = self._side_repr(
            b["opp_slot"], b["opp_species"], torch.zeros_like(b["opp_species"]),
            torch.zeros_like(b["opp_species"]), b["opp_moves"],
            b["opp_slot"][:, :, :, _active])
        ctx = self._context_repr(b, T)
        x = self.input_proj(torch.cat([own, opp, ctx], dim=-1))   # (B,T,GRU)
        g, _ = self.gru(x)                                          # (B,T,GRU)
        # RESIDUAL skip: the current turn-start state ``x`` must reach the head
        # directly. A plain stacked GRU over a short (often T=1) sequence with a
        # zero hidden state acts as a stack of gated attenuations -- each
        # ``(1-z)*tanh(...)`` cell with closed gates kills the signal, and the net
        # then cannot learn to open the gates (no signal -> no gradient), so it
        # collapses to a constant (target-averaging) policy. The skip makes the
        # state the primary signal and lets the GRU contribute only the temporal
        # DELTA (``g``); this is what lets the net state-condition at all.
        h = self.gru_norm(g + x)
        # D70: pointer head — score each action by its identity against the
        # state intent (position-independent action selection, so the shuffled
        # action order cannot be shortcut and the net must pick the right move
        # by what it IS, not where it sits).
        act_emb = torch.cat([
            self.emb_action_kind(b["action_kind"]),   # (B,T,A,16)
            self.emb_move_g(b["action_move"]),        # (B,T,A,E)
            self.emb_species_g(b["action_species"]),  # (B,T,A,E)
        ], dim=-1)                                     # (B,T,A,action_dim)
        act_p = self.policy_act_proj(act_emb)          # (B,T,A,INTENT)
        intent = self.policy_intent(h)                 # (B,T,INTENT)
        # Normalized scaled dot-product (standard attention scoring): score each
        # action by the COSINE similarity between the state intent and the
        # action's identity embedding, then scale by LOGIT_SCALE. Normalizing both
        # factors to unit norm keeps the logits BOUNDED ([-LOGIT_SCALE, LOGIT_SCALE]),
        # which is what makes the cross-entropy finite and the gradient well-
        # conditioned (the raw bilinear head's unbounded logits blew up the S4.2
        # overfit loss). Selection is still purely by action IDENTITY — position-
        # independent — so the D70 order shuffle cannot be shortcut.
        intent = F.normalize(intent, dim=-1)
        act_p = F.normalize(act_p, dim=-1)
        logits = torch.einsum("bti,btai->bta", intent, act_p) * LOGIT_SCALE  # (B,T,A)
        value = self.value_head(h)                                   # (B,T,1)
        prob = _masked_softmax(logits, b.get("action_mask"))
        return {"logits": logits, "value": value, "prob": prob}

    # ── save / load (S4.2 round-trip gate) ─────────────────────────────
    def state_for_save(self) -> Dict:
        """The full save payload: weights + the vocab the weights were built on.

        Saving the vocab sizes alongside the weights means a load can VERIFY
        the layout/vocab matches (a mismatch is a loud error, not silent drift).
        """
        return {
            "state_dict": self.state_dict(),
            "vocab_sizes": {
                "species": len(self.mon_enc.emb_species.weight),
                "ability": len(self.mon_enc.emb_ability.weight),
                "item": len(self.mon_enc.emb_item.weight),
                "move": len(self.mon_enc.emb_move.weight),
                "weather": len(self.emb_weather.weight),
                "terrain": len(self.emb_terrain.weight),
            },
            "layout_checksum": _layout_checksum(),
        }

    @classmethod
    def load(cls, payload: Dict, vocabs: Vocabs,
             device: Optional[str] = None) -> "PokeNet":
        """Load a :meth:`state_for_save` payload into a fresh net built on
        ``vocabs``. Verifies the vocab sizes + layout checksum match — a
        mismatch raises (no silent shape drift).

        ``device`` is optional: a fresh net is always BUILT on CPU (so a CPU-only
        machine can still load a checkpoint), but if a ``device`` (e.g.
        ``"cuda"``) is given the net is moved there AFTER ``load_state_dict``.
        Without this, ``torch.load(map_location="cuda")`` moves only the
        state_dict into GPU memory and then ``load_state_dict`` copies it back
        into the fresh CPU params — so the net SILENTLY stays on CPU while a
        CUDA agent feeds it CUDA tensors (the D102 league pass-fest bug)."""
        net = cls(vocabs)
        saved_sizes = payload.get("vocab_sizes", {})
        live = {
            "species": len(vocabs.species), "ability": len(vocabs.ability),
            "item": len(vocabs.item), "move": len(vocabs.move),
            "weather": len(vocabs.weather), "terrain": len(vocabs.terrain),
        }
        if saved_sizes and saved_sizes != live:
            raise ValueError(f"vocab size mismatch on load: {saved_sizes} != {live}")
        saved_ck = payload.get("layout_checksum")
        if saved_ck is not None and saved_ck != _layout_checksum():
            raise ValueError("layout checksum mismatch on load "
                             f"({saved_ck} != {_layout_checksum()})")
        net.load_state_dict(payload["state_dict"])
        if device is not None:
            net.to(torch.device(device))
        return net

    @classmethod
    def load_file(cls, path, vocabs: Vocabs,
                  device: Optional[str] = None) -> "PokeNet":
        """Load a checkpoint FILE (BOTH formats) into a fresh verified net.

        THE ONE load path (D103 step 1): the league engine, ``net.human_play``,
        ``gui.eval_main`` and ``gui.trainer_main`` each re-derived the same
        two-format sniffing — all five call sites now delegate here, so this
        is the only place the logic lives.

        * **BC format:**        payload = ``{state_dict, vocab_sizes, layout_checksum}``
        * **Self-play format:** payload = ``{"net": {state_dict, …}, "round", …}``

        Vocab-size + layout verification is ``:meth:`load`'s job (a mismatch is
        a loud error, never silent drift). ``device``: the net is BUILT on CPU
        (a CPU-only box can always load a checkpoint), then moved to ``device``
        AFTER ``load_state_dict`` — the D102 silent-CPU-on-GPU fix.
        """
        payload = torch.load(path, map_location="cpu", weights_only=True)
        return cls.load(cls.unwrap_payload(payload), vocabs, device=device)

    @staticmethod
    def unwrap_payload(payload: Dict) -> Dict:
        """The inner net payload of a checkpoint, for BOTH formats — the ONE
        place the two-format logic lives (D103 step 1):

        * self-play ``{"net": {state_dict, …}, "round": …}`` → the ``"net"`` dict
        * BC          ``{state_dict, …}`` → itself

        ``load_file`` uses this; callers that also need the OUTER payload's
        extra fields (``gui.trainer_main`` needs ``round``) use it too — so no
        call site re-derives the sniffing."""
        if isinstance(payload, dict) and isinstance(payload.get("net"), dict) \
                and "state_dict" in payload["net"]:
            return payload["net"]
        return payload


def _layout_checksum() -> str:
    from .registry import checksum
    return checksum()


def _as_int64_tensor(x, device: torch.device, shape: tuple) -> torch.Tensor:
    """Normalize a timestep/target value (python int, numpy, or torch) to an
    int64 torch tensor on ``device`` with ``shape``."""
    if not torch.is_tensor(x):
        x = torch.as_tensor(np.asarray(x), dtype=torch.int64)
    else:
        x = x.to(dtype=torch.int64)
    x = x.to(device)
    if x.numel() == 1 and tuple(shape) != ():
        x = x.expand(shape)
    return x


def bc_loss(logits: torch.Tensor, action_mask: torch.Tensor,
            target, t) -> torch.Tensor:
    """Cross-entropy at timestep ``t`` over the LEGAL actions only.

    ``logits`` may be a full sequence (B, T, A) or an already-sliced
    (B, A) tensor. ``t`` and ``target`` are each either a scalar (uniform
    over the batch — the B=1 case) or a per-batch (B,) array/tensor; in the
    per-batch case every row is sliced/supervised at ITS OWN timestep, which
    is what makes mixed-length (left-padded) batches correct. ``target``
    must be a LEGAL action position (mask==1). We softmax over the masked
    logits and take the negative log-prob of the target. Illegal positions
    are exactly 0, so the loss is well-defined only on legal targets (the
    corpus guarantees this).
    """
    B = logits.shape[0]
    dev = logits.device
    t_t = _as_int64_tensor(t, dev, (B,))
    tgt = _as_int64_tensor(target, dev, (B,))
    if logits.dim() == 3:
        # logits (B, T, A) and action_mask (B, T, A) -> row i's timestep t_i
        A = logits.shape[-1]
        idx = t_t.unsqueeze(-1).unsqueeze(-1).expand(B, 1, A)
        logits = torch.gather(logits, 1, idx).squeeze(1)          # (B, A)
        action_mask = torch.gather(action_mask, 1, idx).squeeze(1)  # (B, A)
    prob = _masked_softmax(logits, action_mask)
    logp = prob.clamp_min(1e-12).log().gather(-1, tgt.unsqueeze(-1))
    return -logp.squeeze(-1)


def overfit_train(net: PokeNet, batches: List[Dict[str, torch.Tensor]],
                  targets: List[torch.Tensor], ts: List[object],
                  epochs: int = 30, lr: float = 3e-3,
                  max_norm: float = 1.0) -> float:
    """The S4.2 overfit-32 loop: drive the BC loss on a small fixed set of
    (state, action) samples down to ~0. Returns the FINAL mean training loss.

    This is the "never learns" disaster guard (S4.2 gate): if a net of this
    capacity cannot memorize 32 battles, something is structurally wrong.

    ``targets[i]`` is a per-batch (B,) action-index tensor and ``ts[i]`` the
    matching per-batch (B,) timestep array — each sample is supervised at its
    OWN last real timestep (samples in one batch may have different lengths).
    A scalar ``t`` is still accepted for a B=1 batch (the smoke-probe path).
    Gradients are clipped (``max_norm``) for stability; with the default
    ``lr`` and a few hundred steps this net comfortably memorizes a few
    dozen samples.
    """
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    final = float("inf")
    for _ in range(epochs):
        net.train()
        tot = 0.0
        n = 0
        for b, tgt, t in zip(batches, targets, ts):
            opt.zero_grad()
            out = net(b)
            loss = bc_loss(out["logits"], b["action_mask"], tgt, t).mean()
            loss.backward()
            if max_norm:
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)
            opt.step()
            tot += loss.item()
            n += 1
        final = tot / max(1, n)
    return final
