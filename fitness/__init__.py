"""fitness/ — measure a bot's ABSOLUTE strength (not just relative ELO).

The ladder of reference opponents + the win-rate harness that scores any bot
against it. CPU-first: importing this package never pulls torch — references
are factories, called only when a battle is actually played (same discipline
as ``pool`` / ``league``).

See ``OPEN_DESIGN_CHOICES.md`` -> "Option A — fitness/eval harness" for the
design + build order (A1..A5).
"""
from .ladder import Reference, Ladder, random_reference

__all__ = ["Reference", "Ladder", "random_reference"]
