"""S4.1 — id vocabularies: legality-DB string ids -> stable int indices.

The legality DB keys entities by Showdown ``toID`` strings (``garchomp``,
``stealthrock``, ``choiceband``, ``sturdy``, ...). The S4.2 network wants
integer row indices for its embedding tables, so this module is the single
mapping between the two.

Guarantees:
  * index 0 is ALWAYS reserved for ``<pad>`` / out-of-vocabulary;
  * indices are assigned in SORTED id order, so the mapping is deterministic
    and identical across processes and machines (Python's built-in ``hash``
    is salted per-process and must NOT be used for this);
  * S4.2 builds one :class:`Vocab` per domain from the legality DB (the
    ``pokemon`` / ``ability`` / ``item`` / ``move`` tables) and passes the SAME
    :class:`Vocabs` object to both the encoder (here) and the embedding tables
    — so the encoder's int indices and the net's embedding rows can never
    disagree (the "two sources of truth" the user forbids).

For S4.1 the encoder is DB-free: tests build a :class:`Vocabs` from the ids
present in the context. The production build path (DB -> Vocabs) is a one-line
S4.2 concern and does not change any tensor shape.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List


class Vocab:
    """A deterministic ``id -> index`` map. Index 0 is reserved for <pad>/OOV."""

    __slots__ = ("_id2idx", "_idx2id", "size")

    def __init__(self, ids: Iterable[str] = ()):
        self._id2idx: Dict[str, int] = {}
        self._idx2id: List[str] = []
        for ident in sorted(set(ids)):
            self._add_sorted(ident)
        self.size = len(self._idx2id) + 1  # + the reserved <pad> slot

    def _add_sorted(self, ident: str) -> int:
        self._id2idx[ident] = len(self._idx2id) + 1
        self._idx2id.append(ident)
        self.size = len(self._idx2id) + 1   # keep size live (add() must grow it)
        return self._id2idx[ident]

    def index(self, ident) -> int:
        """Map an id to its index; unknown / empty -> 0 (<pad>/OOV)."""
        if not ident:
            return 0
        return self._id2idx.get(ident, 0)

    def add(self, ident: str) -> int:
        """Register a new id at runtime (tests / streaming). Stable: returns the
        existing index if already known. (Production builds the Vocab up front
        from the DB; this is only for incremental use.)"""
        if not ident:
            return 0
        if ident in self._id2idx:
            return self._id2idx[ident]
        return self._add_sorted(ident)

    def id_of(self, index: int) -> str:
        """Inverse map (for debugging / tests). 0 -> '' (the <pad> slot)."""
        if index <= 0 or index > len(self._idx2id):
            return ""
        return self._idx2id[index - 1]

    def __len__(self) -> int:
        # dynamic, not the init-time snapshot: a runtime add() must be visible
        return len(self._idx2id) + 1

    def __contains__(self, ident) -> bool:
        return bool(ident) and ident in self._id2idx


@dataclass
class Vocabs:
    """The per-domain vocab bundle the encoder (and S4.2's embeddings) share.

    ``action_kind`` is a tiny FIXED set (move/switch/tera/transform) — the
    encoder uses hard-coded indices for it (see ``ACTION_KIND_IDX``), so it is
    kept here only for documentation / size accounting, not for lookups.
    """
    species: Vocab = field(default_factory=Vocab)
    ability: Vocab = field(default_factory=Vocab)
    item: Vocab = field(default_factory=Vocab)
    move: Vocab = field(default_factory=Vocab)
    weather: Vocab = field(default_factory=Vocab)
    terrain: Vocab = field(default_factory=Vocab)

    def sizes(self) -> Dict[str, int]:
        return {
            "species": len(self.species),
            "ability": len(self.ability),
            "item": len(self.item),
            "move": len(self.move),
            "weather": len(self.weather),
            "terrain": len(self.terrain),
        }
