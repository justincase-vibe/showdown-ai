"""D101 — GUI replay-viewer wiring (headless).

Tests the tkinter button + background-thread flow WITHOUT opening a browser
or touching the real library (``replays.open_replay_in_browser`` and
``replays.ReplayLibrary`` are monkeypatched). Uses the module-shared Tk root
(see gui/tests/conftest.py).

Run:  python -m pytest gui/tests/test_replay_viewer.py -v
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

FAKE_RECORD = {
    "id": "49960-battle-gen9ou-3196",
    "formatid": "gen9ou",
    "format": "gen9ou",
    "players": ["A", "B"],
    "log": "|init|battle\n|turn|1\n",
}


def _pump(app, predicate, timeout_s: float = 5.0) -> bool:
    """Pump the tkinter event loop until ``predicate()`` is true (or timeout)."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        app.root.update()
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_open_btn_exists_headless(shared_root):
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    assert app._replay_open_btn is not None
    assert app._replay_status_label is not None
    app.close()


def test_open_without_selection_shows_hint(shared_root):
    from gui.desktop.app import DesktopApp
    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    app._open_replay_in_browser()   # no row selected
    assert "Select a replay row first" in app._replay_status_label.cget("text")
    app.close()


def test_open_selected_row_writes_file_and_reports_path(monkeypatch, tmp_path, shared_root):
    """Click flow: row id -> get_by_id -> open_replay_in_browser -> status label
    shows the written path. (Monkeypatched: no browser, no real library.)"""
    import replays
    from gui.desktop.app import DesktopApp

    captured = {}

    class _FakeLib:
        def __init__(self, root=None):
            self.root = root

        def get_by_id(self, rid):
            captured["rid"] = rid
            return dict(FAKE_RECORD)

    def _fake_open(record, out_dir=None):
        captured["record"] = record
        p = tmp_path / f"{record['id']}.html"
        p.write_text("fake html", encoding="utf-8")
        return p

    monkeypatch.setattr(replays, "ReplayLibrary", _FakeLib)
    monkeypatch.setattr(replays, "open_replay_in_browser", _fake_open)

    app = DesktopApp(headless=True, root=shared_root)
    app.build()
    app._replay_tree.insert("", "end", values=("49960-battle-gen9ou-3196",
                                               "gen9ou", "", "2026-01-01T00:00:00Z"))
    app._replay_tree.selection_set(app._replay_tree.get_children()[0])
    app._open_replay_in_browser()

    ok = _pump(app, lambda: "Opened:" in app._replay_status_label.cget("text")
               or "Error" in app._replay_status_label.cget("text"))
    text = app._replay_status_label.cget("text")   # read BEFORE close (unsafe after)
    if app._replay_worker is not None:
        app._replay_worker.join(timeout=5.0)      # no live thread at teardown
    app.close()

    assert ok, "status label never reported completion (thread dead?)"
    assert captured["rid"] == "49960-battle-gen9ou-3196"
    assert captured["record"]["log"] == FAKE_RECORD["log"]
    assert "Opened:" in text
