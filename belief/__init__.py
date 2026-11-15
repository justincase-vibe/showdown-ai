"""Step 3 — Opponent belief sampler (build plan step 3; execution plan in PLAN.md §3).

Bayesian belief over the opponent's HIDDEN info (set / item / ability / nature /
EV spread / Tera type) per opponent slot, initialised from Step-1 set data and
updated with observable events. Hidden info is NEVER leaked into observations —
this module is the only place it exists (rule 4).

Sub-steps (one at a time, each pytest-gated):
  3.1  state + event model + prior + log→event adapter   (belief/state.py,
       belief/events.py, belief/log_events.py)
  3.2  evidence engine (HARD/SOFT likelihoods, ε floor)  (belief/evidence.py)
  3.3  update(): revealed move / ability / Choice-lock / Tera
  3.4  damage + speed-order evidence (via the Step 2 calc, KO-clip D40)
  3.5  sample(n) + move_probabilities()
  3.6  max-HP / EV-bucket inference
  3.7  whole-battle fuzz bug-catcher + readable belief report
  3.8  perf bench + wrap-up

Sources of truth (PLAN.md, D37): damage numbers → @smogon/calc (Step 2) ·
raw game data → the fork's Dex (Step 1 DB) · battle observations → the fork.
"""
from .state import BeliefState, SlotBelief, build_prior  # noqa: F401
