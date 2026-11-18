"""S4.1 — observation encoder (PLAN.md §3c, D61/D60).

Turns the belief stack's ``get_context()`` snapshot (JSON-safe dict) into the
FIXED float/int tensor layout the S4.2 network consumes.

Modules
-------
``registry``  — the SINGLE SOURCE OF TRUTH for the tensor layout: an ordered
                list of named fields with fixed shape + dtype, and a
                checksum-pinned drift-catcher (``checksum()``). The width is
                DERIVED from here — never hand-maintained (the D61 OBS_DIM fix).
``vocab``     — legality-DB string ids -> stable int indices (index 0 = <pad>).
                Shared by the encoder AND the S4.2 embedding tables, so they
                can never disagree.
``encoder``   — the pure, deterministic ``get_context() -> dict[str, ndarray]``
                mapping that fills exactly the registry's fields.

Design (D61/D60, doubles-robust):
  * each side = a SET of <=6 UNIFORM slot blocks; ``active``/``fainted`` are
    PER-SLOT FLAGS (columns), not block boundaries — singles and doubles share
    the same tensor shape + the same net (no ``if doubles:`` branch);
  * own and opponent share ONE uniform block shape (differing only in
    known-vs-posterior content);
  * identity = int id channels for learned embeddings (NOT one-hot, NOT the
    old lossy multi-probe hash);
  * opponent UNREVEALED info reaches the net ONLY via the posterior channels
    (rule 4, no leakage);
  * D60: the current legal-action list is emitted as indexed inputs so the net
    sees what each action position IS.

Deliberately deferred (flagged for the user, see ``encoder`` docstring):
  * the 7 curated volatiles + G1–G6 (turns-left / revealed? / PP / static dex)
    require ``belief/`` data additions (331-baseline risk) — the NEXT decision;
  * per-slot ``fainted`` is a RESERVED column left 0.0 (get_context() cannot
    yet tell "benched, untouched" from "fainted" — both read hp_frac 0 — and the
    raw evidence is in the slot block). A dedicated ``belief/`` flag will fill it
    with no shape change — next decision.
"""
from .registry import (
    LAYOUT,
    FIELDS,
    SHAPES,
    DTYPES,
    Field,
    checksum,
    total_elements,
    summary,
)
from .vocab import Vocab, Vocabs
from .encoder import encode_context, encode_sequence
from .vocab_db import build_vocabs
from .model import PokeNet, bc_loss, overfit_train
from .batch import build_samples, samples_to_batch, batch_to_torch
from .agent import PokeAgent, Decision, make_random_net
from .ppo import (
    gae,
    RolloutStep,
    RolloutBuffer,
    steps_to_batch,
    chosen_logprobs,
    PPOTrainer,
)

__all__ = [
    "LAYOUT", "FIELDS", "SHAPES", "DTYPES", "Field",
    "checksum", "total_elements", "summary",
    "Vocab", "Vocabs", "build_vocabs",
    "encode_context", "encode_sequence",
    "PokeNet", "bc_loss", "overfit_train",
    "build_samples", "samples_to_batch", "batch_to_torch",
    "PokeAgent", "Decision", "make_random_net",
    "gae", "RolloutStep", "RolloutBuffer", "steps_to_batch",
    "chosen_logprobs", "PPOTrainer",
]
