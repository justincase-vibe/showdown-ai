# `_legacy/gui_browser/` — RETIRED browser (FastAPI) GUI (DO NOT USE)

The browser-based training console has been **superseded** by the native desktop
GUI and quarantined here (moved, not deleted).

- `server.py` — the FastAPI control plane (served the dashboard + `/api/*`).
- `static/index.html` — the browser dashboard.
- `__main__.py` — the `python -m gui` entry (ran `uvicorn gui.server:app`).
- `tests/` — the 4 tests that exercised `gui.server`
  (`test_p43_2/3/4.py`, `test_selfplay_collect.py`).

## What to use instead
The **native Mission Control GUI** (tkinter) is the single GUI:

    python -m gui.desktop          # or double-click start_gui.bat / Launcher.bat

It drives the same `gui/launcher.py` (detached trainer subprocess) +
`gui/runstate.py` (on-disk run protocol) + `gui/{trainer,bc,eval}_main.py`
subprocess entries, so **no training capability was lost** — only the browser
front-end. The shared pieces (`launcher.py`, `runstate.py`, `desktop/`, the
`*_main.py` entries, and the non-server tests) all remain in `gui/`.

## Reverting
`mv _legacy/gui_browser/server.py _legacy/gui_browser/__main__.py gui/ && \
 mv _legacy/gui_browser/static gui/static && \
 mv _legacy/gui_browser/tests/*.py gui/tests/`
</new_content>
            </invoke>
