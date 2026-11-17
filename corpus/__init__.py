"""S4.0 — BC corpus pipeline (PLAN.md §3c).

Replays real human Showdown ladder battles through the EXISTING belief stack
(``belief/log_events.py`` adapter + ``belief.state.BeliefState.get_context``)
at each decision point and emits ``(context_seq, action)`` trajectories for
both POVs. This is the data groundwork for the BC warm-start (S4.3) and it
measures the human switch-rate the S4.3 gate targets.

Reuse-first (user: "i dont want code being written twice"):
  * the network puller wraps ``bot.datamine.fetch.fetch_top`` (no rewrite);
  * the event stream is ``belief.log_events.LogAdapter`` (the 3.1 adapter);
  * the observation is ``belief.state.BeliefState.get_context()`` (the 3.4 API).
Only the decision-point walker + the report are new.

Modules:
  ``pull``        — deterministic, resumable puller + local loader.
  ``trajectories``— the decision-point walker (log -> events -> belief ->
                    decisions + turn-start observations). The bug-catcher: it
                    counts/samples ``Unknown`` (unmapped) log lines so a
                    human-ladder tag the fork-fuzz never saw is CAUGHT (the S4.0
                    gate asserts 0), not silently dropped.
  ``report``      — battle count, format mix, rating distribution, human
                    switch-rate, legality rate, unknown-tag breakdown.

Design (D58/D60/D61): the action is the compact per-state legal-action label
(D60); the observation is ``get_context()`` — the D61 observation overhaul
happens in S4.1; S4.0 only verifies the pipeline yields valid contexts for both
sides.
"""

__all__ = ["pull", "trajectories", "report"]
