"""legality — a local source of truth for Pokémon Showdown legality.

Per-format legal species/forms/abilities/moves/items (gens 1–9), Smogon-parity
reference data (types, stats, tiers, learnsets, strategies), and wipeable
win rates.

    from legality import api
    api.status()
    api.legal_forms("gen9ou")

See ``PLAN.md`` for the mission, decision log and phase gates, and
``ingest_protocol.md`` for how data enters the DB.
"""
from . import api, db, formats

__all__ = ["api", "db", "formats"]
