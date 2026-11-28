"""Team brain — the league builder's learned team-synergy ranker (step 3).

Role (approved design 2026-11-28): the living team pool (``league/teams.py``)
breeds candidate teams by mutation; the TEAM BRAIN ranks those candidates so
the pool keeps its best lineups. It learns TEAM SYNERGY (typing / moveset
combinations) from the league's OWN battle outcomes — the user's ask:
"the NN could learn team synergies like typing and moveset synergies".

Architecture (small — the battle nets are 12 MB total, this must stay in the
same league):
  * ``TeamEncoder`` — one token per mon (6 tokens), reusing the battle
    net's LEARNED EMBEDDING RECIPE (``net/model.py::MonEncoder``):
    species / ability / item embeddings + mean over ≤4 move embeddings
    (``EMBED_DIM=32``) -> a per-mon vector; a 2-layer attention stack over
    the 6 tokens (interactions between mons — the synergy) -> mean pool
    (a set function: slot order must not matter) -> team vector.
  * ``TeamBrain`` — a scalar quality logit per team + a Bradley-Terry
    PREFERENCE head: ``P(a better than b) = σ(score(a) − score(b))``.
    Trained on (team_a, team_b, a_better) pairs mined from the team
    registry's ELO (the league's own results, refreshed every cull
    boundary). NO move tags / hand-designed features (project ruling):
    meaning lives in the learned embeddings + attention.

Hermetic guarantee (pinned by ``net/tests/test_team_brain.py``): an
overfit test on a SYNTHETIC synergy rule (a count-interaction pattern a
per-mon linear model cannot express) must be learned — if the brain
cannot overfit an obvious synergy, it cannot be trusted with the 10k.

Torch at import (consistent with ``net/model.py``); the league engine
pulls it lazily at the cull boundary.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vocab import Vocabs

#: learned id-embedding width (SAME as net.model.EMBED_DIM — the shared
#: recipe the handoff approves; not a new constant)
EMBED_DIM = 32
#: per-mon token width
MON_DIM = 96
#: attention stack over the 6 mon tokens
N_HEADS = 4
N_LAYERS = 2
FFN_DIM = 192
#: ≤4 moves per mon (the exchange format's cap)
MAX_MOVES = 4


def _to_id(name) -> str:
    """Showdown toID (same 2-line convention as net/encoder.py::_to_id —
    codebase convention, not a new source of truth)."""
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


def _idx(vocab, ident) -> int:
    """Vocab lookup accepting an id OR a display name (net/encoder::_idx)."""
    return vocab.index(_to_id(ident))


# ═══════════════════════════════════════════════════════════════════════════
#  tokenization (the team_pools exchange format -> tensors)
# ═══════════════════════════════════════════════════════════════════════════

def team_to_ids(mons: Sequence[Dict], vocabs: Vocabs,
                n_mons: int = 6) -> Dict[str, torch.Tensor]:
    """One team (≤6 mon-dicts) -> int id rows for the encoder.

    ``mons`` is the team_pools exchange format (species/item/ability/moves).
    Slots beyond ``len(mons)`` are zero-padded (the <pad> row — the same
    OOV-as-<pad> discipline the battle encoder uses, so a 6-mon team is the
    norm but nothing breaks if a pool entry is shorter)."""
    n = min(max(1, int(n_mons)), 6)
    sp = torch.zeros(n, dtype=torch.long)
    ab = torch.zeros(n, dtype=torch.long)
    it = torch.zeros(n, dtype=torch.long)
    mv = torch.zeros(n, MAX_MOVES, dtype=torch.long)
    for i, m in enumerate(mons[:n]):
        sp[i] = _idx(vocabs.species, m.get("species"))
        ab[i] = _idx(vocabs.ability, m.get("ability"))
        it[i] = _idx(vocabs.item, m.get("item"))
        moves = [mv_id for mv_id in
                 (_idx(vocabs.move, x) for x in (m.get("moves") or [])[:MAX_MOVES])]
        mv[i, :len(moves)] = torch.tensor(moves, dtype=torch.long)
    return {"species": sp, "ability": ab, "item": it, "moves": mv}


def teams_to_batch(teams: Sequence[Sequence[Dict]], vocabs: Vocabs) -> Dict[str, torch.Tensor]:
    """A list of teams -> a batched id dict (B, ...) for :meth:`TeamBrain.score`."""
    out: Dict[str, torch.Tensor] = {}
    for t in teams:
        ids = team_to_ids(t, vocabs)
        for k, v in ids.items():
            out.setdefault(k, []).append(v)
    return {k: torch.stack(v) for k, v in out.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  encoder + brain
# ═══════════════════════════════════════════════════════════════════════════

class TeamMonEncoder(nn.Module):
    """One mon-dict -> a MON_DIM token.

    Reuses the battle net's learned-embedding recipe (MonEncoder): species /
    ability / item embeddings + mean over ≤4 move embeddings, then a small MLP.
    Slot order is meaningless here (a team is a SET of mons) — the attention
    stack + mean pool below keep it that way."""

    def __init__(self, vocabs: Vocabs) -> None:
        super().__init__()
        self.emb_species = nn.Embedding(len(vocabs.species), EMBED_DIM, padding_idx=0)
        self.emb_ability = nn.Embedding(len(vocabs.ability), EMBED_DIM, padding_idx=0)
        self.emb_item = nn.Embedding(len(vocabs.item), EMBED_DIM, padding_idx=0)
        self.emb_move = nn.Embedding(len(vocabs.move), EMBED_DIM, padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(4 * EMBED_DIM, MON_DIM),
            nn.LayerNorm(MON_DIM),
            nn.GELU(),
        )

    def forward(self, ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """ids (B,S,*) -> (B,S,MON_DIM)."""
        emb = torch.cat([
            self.emb_species(ids["species"]),          # (B,S,E)
            self.emb_ability(ids["ability"]),          # (B,S,E)
            self.emb_item(ids["item"]),                # (B,S,E)
            self.emb_move(ids["moves"]).mean(dim=-2),  # (B,S,E) mean over ≤4 moves
        ], dim=-1)
        return self.mlp(emb)


class TeamEncoder(nn.Module):
    """6 mon tokens -> one team vector (MON_DIM).

    2 attention layers over the 6 tokens (the mons INTERACT — typing /
    moveset synergy is a property of the COMBINATION, not of any single
    mon) + mean pool (a set function: shuffling the team's slots must not
    change the vector — the same permutation-invariance discipline the
    battle net's side pooling uses)."""

    def __init__(self, vocabs: Vocabs) -> None:
        super().__init__()
        self.mon = TeamMonEncoder(vocabs)
        layer = nn.TransformerEncoderLayer(
            d_model=MON_DIM, nhead=N_HEADS, dim_feedforward=FFN_DIM,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.att = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.out_norm = nn.LayerNorm(MON_DIM)

    def forward(self, ids: Dict[str, torch.Tensor]) -> torch.Tensor:
        """ids (B,S,*) -> (B,MON_DIM)."""
        tokens = self.mon(ids)                       # (B,S,MON_DIM)
        h = self.att(tokens)                         # (B,S,MON_DIM)
        return self.out_norm(h.mean(dim=1))          # (B,MON_DIM) mean pool


class TeamBrain:
    """The learned team ranker (quality logit + Bradley-Terry preference).

    ``score(teams)`` -> one quality logit per team.
    ``rank(parent, candidates)`` -> candidate indices best-first (the
    builder's entry point: keep the best mutation of the parent).
    ``train_step(pairs)`` -> one preference pass (pairs = (a_mons, b_mons,
    a_better) mined from the team registry's ELO).
    Save/load: ``save`` / ``load_file`` (state_dict; the vocabs are fixed
    for a league run, so they are NOT serialized — load with the SAME
    vocabs, exactly like the battle nets).
    """

    def __init__(self, vocabs: Vocabs, device: str = "cpu",
                 seed: Optional[int] = None) -> None:
        self.vocabs = vocabs
        self.device = device
        if seed is not None:
            g = torch.Generator()
            g.manual_seed(int(seed))
            torch.manual_seed(int(seed))
        self.encoder = TeamEncoder(vocabs).to(device)
        self.head = nn.Linear(MON_DIM, 1).to(device)
        self._opt = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=1e-3)

    # ── inference ────────────────────────────────────────────────────────
    def score(self, teams: Sequence[Sequence[Dict]]) -> torch.Tensor:
        """Quality logit per team: (n_teams,) on self.device.

        ``teams`` = list of mon-dict lists (the exchange format)."""
        self.encoder.eval()
        self.head.eval()
        ids = teams_to_batch(list(teams), self.vocabs)
        ids = {k: v.to(self.device) for k, v in ids.items()}
        with torch.no_grad():
            v = self.encoder(ids)                    # (B,MON_DIM)
            out = self.head(v).squeeze(-1)           # (B,)
        return out

    @torch.no_grad()
    def rank(self, candidates: Sequence[Sequence[Dict]],
             parent: Optional[Sequence[Dict]] = None) -> List[int]:
        """Candidate indices best-first. With ``parent`` given, candidates
        whose score does NOT beat the parent are ranked LAST (a mutation
        that does not improve the parent is no better than keeping it) —
        the builder only replaces a team when a candidate beats it."""
        teams = list(candidates)
        s = self.score(teams)
        parent_score = self.score([list(parent)])[0] if parent is not None else None
        idx = torch.argsort(s, descending=True).tolist()
        if parent_score is not None:
            idx.sort(key=lambda i: (bool(s[i] > parent_score), float(s[i])),
                     reverse=True)
        return idx

    # ── training ─────────────────────────────────────────────────────────
    def train_step(self, pairs: Sequence[Tuple[Sequence[Dict], Sequence[Dict], bool]],
                   epochs: int = 1, lr: Optional[float] = None) -> Dict[str, float]:
        """One preference training pass over (a, b, a_better) pairs.

        Bradley-Terry: ``P(a ≻ b) = σ(score(a) − score(b))`` — the SAME
        relative formulation the league uses to compare teams (ELO is an
        expectation of exactly this). ``a_better=False`` means b ≻ a."""
        self.encoder.train()
        self.head.train()
        if lr is not None:
            for g in self._opt.param_groups:
                g["lr"] = float(lr)
        n = len(pairs)
        if n == 0:
            return {"n_pairs": 0, "loss": float("nan")}
        a_ids = teams_to_batch([a for a, _, _ in pairs], self.vocabs)
        b_ids = teams_to_batch([b for _, b, _ in pairs], self.vocabs)
        targets = torch.tensor([1.0 if ab else 0.0 for _, _, ab in pairs],
                               dtype=torch.float32, device=self.device)
        a_ids = {k: v.to(self.device) for k, v in a_ids.items()}
        b_ids = {k: v.to(self.device) for k, v in b_ids.items()}
        total = 0.0
        for _ in range(max(1, int(epochs))):
            self._opt.zero_grad()
            sa = self.head(self.encoder(a_ids)).squeeze(-1)
            sb = self.head(self.encoder(b_ids)).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(sa - sb, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.head.parameters()),
                max_norm=1.0)
            self._opt.step()
            total = float(loss.detach())
        return {"n_pairs": n, "loss": total}

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "head": self.head.state_dict(),
            "mon_dim": MON_DIM, "embed_dim": EMBED_DIM,
            "n_heads": N_HEADS, "n_layers": N_LAYERS, "ffn": FFN_DIM,
        }, p)

    @classmethod
    def load_file(cls, path, vocabs: Vocabs, device: str = "cpu") -> "TeamBrain":
        brain = cls(vocabs, device=device)
        blob = torch.load(path, map_location=device)
        brain.encoder.load_state_dict(blob["encoder"])
        brain.head.load_state_dict(blob["head"])
        brain.encoder.eval()
        brain.head.eval()
        return brain


# ═══════════════════════════════════════════════════════════════════════════
#  mining preference pairs from the team registry (the league's own results)
# ═══════════════════════════════════════════════════════════════════════════

def team_pairs_from_registry(reg, min_battles: int = 2,
                             max_pairs: Optional[int] = None
                             ) -> List[Tuple[List[Dict], List[Dict], bool]]:
    """(a_mons, b_mons, a_better) preference pairs from a TeamRegistry.

    A pair is eligible once the two teams have decided at least
    ``min_battles`` combined (W+L+D) — one battle of ELO noise is not a
    preference. ``a_better`` = the higher-ELO side (the league's own
    verdict). Pairs with EQUAL ELO are dropped (no preference signal).

    Parent attribution is preserved FOR FREE: a generated team's ``parent``
    pairs appear here whenever parent and child have both fought — the
    (parent, child) matchups are the most controlled synergy experiments
    (exactly one slot changed) and the ELO comparison ranks them like any
    other pair."""
    teams = reg.all()
    by_tid = {t.tid: t for t in teams}
    out: List[Tuple[List[Dict], List[Dict], bool]] = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            ta, tb = teams[i], teams[j]
            decided_a = ta.wins + ta.losses + ta.draws
            decided_b = tb.wins + tb.losses + tb.draws
            if min(decided_a, decided_b) < max(1, int(min_battles)):
                continue
            if ta.elo == tb.elo:
                continue
            a_better = ta.elo > tb.elo
            a_mons, b_mons = list(ta.mons), list(tb.mons)
            if a_better:
                out.append((a_mons, b_mons, True))
            else:
                out.append((b_mons, a_mons, True))
    if max_pairs is not None and len(out) > int(max_pairs):
        # deterministic cap (registry order is insertion-stable)
        out = out[:int(max_pairs)]
    return out
