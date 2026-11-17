"""S4.2 — build the shared :class:`net.vocab.Vocabs` from the legality DB.

PLAN.md §3c S4.2: the S4.2 network's id-embeddings are keyed by legality-DB ids
and share the SAME :class:`net.vocab.Vocabs` the encoder uses, so the encoder's
int indices and the net's embedding rows can never disagree.

``net.vocab`` stays DB-free (S4.1 is pure/numpy); this module is the ONE place
that reads the DB to build the production vocabs. It is deterministic (the
:class:`Vocab` assigns indices in sorted-id order), so a fresh build is
bitwise-identical across processes — the same guarantee the S4.1 tests rely on.

Domains (each -> one ``Vocab``; index 0 is always ``<pad>``/OOV):
  * species  <- ``form.id``
  * move     <- ``move.id``
  * ability  <- ``ability.id``
  * item     <- ``item.id``
  * weather / terrain <- a small FIXED set (the field conditions that appear in
    ``get_context()['field']``); they are not DB tables, so we enumerate them.
"""
from __future__ import annotations

from typing import List

from .vocab import Vocab, Vocabs

#: the field weather / terrain ids that appear in ``get_context()['field']``
#: (Showdown ids). A FIXED small set — not DB-sourced, so the encoder and the
#: net agree on the same tiny vocabulary regardless of the DB contents.
WEATHER_IDS: List[str] = [
    "raindance", "primordialsea", "sun", "strongwinds", "sandstorm",
    "hail", "snowscape", "electricterrain",
]
TERRAIN_IDS: List[str] = [
    "electricterrain", "grassyterrain", "mistyterrain", "psychicterrain",
]


def _ids_from_table(conn, table: str) -> List[str]:
    return [r[0] for r in conn.execute(f"SELECT id FROM {table} ORDER BY id")]


def build_vocabs(db_path: str = None) -> Vocabs:
    """Build the production :class:`Vocabs` from the legality DB at ``db_path``.

    ``db_path`` defaults to the legality DB's own default. Deterministic: the
    same DB contents always produce the same index assignment (sorted order).
    """
    from legality import db  # lazy: keeps net/ importable without the DB

    conn = db.connect(db_path) if db_path else db.connect()
    try:
        v = Vocabs(
            species=Vocab(_ids_from_table(conn, "form")),
            move=Vocab(_ids_from_table(conn, "move")),
            ability=Vocab(_ids_from_table(conn, "ability")),
            item=Vocab(_ids_from_table(conn, "item")),
            weather=Vocab(WEATHER_IDS),
            terrain=Vocab(TERRAIN_IDS),
        )
        return v
    finally:
        conn.close()
