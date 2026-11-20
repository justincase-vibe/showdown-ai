"""The self-play REPLAY LIBRARY (Phase 2 core of the Mission Control GUI).

Save + index + fetch for self-play battle replays, in the project's established
BC fixture shape (``{id, formatid, format, rating, players, log}``) so the BC
walker (``corpus.trajectories.replay_battle``) consumes self-play records with
zero changes. Framework-agnostic: the GUI (web or native) calls the same API.

    from replays import ReplayLibrary, make_record
    lib = ReplayLibrary()                    # default root: <project>/runs/replays
    lib.save(make_record(battle, "gen9ou", players=[a1, a2]))
    last  = lib.fetch_last(50)
    top   = lib.fetch_top(200, key="elo")    # "elo" now / "rating" for the P4 era
    meta  = lib.summary()                    # small dict for the GUI

Storage: ``{root}/{formatid}.jsonl`` (append-only, the truth) + ``index.json``
(byte offsets — fetch reads exactly N records, O(N) in N not file size).
"""
from .library import ReplayLibrary, DEFAULT_ROOT
from .record import make_record
from .view import (  # D101 — replay viewer (browser / PS-client drag&drop)
    build_replay_html,
    open_replay_in_browser,
    write_replay_file,
)

__all__ = ["ReplayLibrary", "make_record", "DEFAULT_ROOT",
           "build_replay_html", "write_replay_file", "open_replay_in_browser"]
