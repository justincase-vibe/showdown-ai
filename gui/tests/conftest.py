"""Shared fixtures for the GUI tests (gui/tests/).

``shared_root`` — ONE ``tk.Tk`` (Tcl interpreter) per test module.

Why: creating a fresh ``tk.Tk()`` per test on this Windows machine can flake
("Can't find a usable init.tcl" — the new interpreter's Tcl init fails to read
init.tcl when it happens right after file IO; a ~3s delay makes it pass).
Sharing one root per module removes the repeated interpreter init entirely and
also exercises the app's embedding path (``DesktopApp(root=...)``).
"""
from __future__ import annotations

import pytest
import tkinter as tk


@pytest.fixture(scope="module")
def shared_root():
    root = tk.Tk()
    yield root
    root.destroy()
