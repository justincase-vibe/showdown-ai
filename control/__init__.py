"""control/ — the Mission Control BACKEND RUN ENGINES (framework-agnostic).

The GUI (web or native — open decision #1) is a thin front over these. Each
engine:
  * is config-driven (a dataclass with ``to_dict``/``from_dict``);
  * speaks the ``gui.runstate`` on-disk protocol (status / heartbeat /
    stop.flag / log) — GUI restarts never lose a run;
  * can be driven in-process (library) or as a detached subprocess
    (``python -m control.<engine>_main --config ... --run-dir ...``),
    mirroring the established ``gui.trainer_main`` contract;
  * never touches the GPU (the user owns it; agents stay CPU here).

Currently: ``selfplay`` (feature 1's "adjustable games" + feature 2's save
side). League mechanics (multi-bot / ELO) are held per the user's note and
will slot in via the ``battle_driver`` hook.
"""
from .selfplay import SelfPlayConfig, SelfPlayRun

__all__ = ["SelfPlayConfig", "SelfPlayRun"]
