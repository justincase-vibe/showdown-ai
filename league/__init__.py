"""League infrastructure — multi-bot self-play with ELO tracking.

The league is the next step after the self-play capture (D98) and the
native GUI (Phase 2). It adds:

  * **Bot registry** — track bots (checkpoint, name, ELO, W/L) in a JSON file
  * **Pairing** — decide which bots fight (round-robin, ELO-weighted)
  * **League engine** — pair → battle → update ELO → repeat (reuses the
    pool + runstate protocol from ``control/``)

The league is framework-agnostic (like ``control/`` and ``pool/``): it
imports no torch at import time, and the GUI (native or web) drives it
through the same ``gui.launcher`` control plane.
"""
