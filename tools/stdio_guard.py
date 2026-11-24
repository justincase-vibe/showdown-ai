"""Crash-safe stdio for redirected runs (the Windows cp1252 trap).

Why this exists (2026-11-24 incident, ``runs/learnability_smoke.log``):
on Windows, a run whose stdout is redirected to a log file
(``python -m ... > run.log 2>&1``) gets the LOCALE encoding (cp1252) on its
streams. Printing any character cp1252 cannot encode -- the box-drawing
``U+2500`` is the one that killed the learnability smoke run right after a
perfectly good baseline eval -- raises ``UnicodeEncodeError`` and takes the
whole multi-hour run down with it. (Interactive consoles are UTF-8, which is
why the same prints never crashed in a terminal.)

Rule: call :func:`safe_stdio` ONCE at the very top of every entry point that
prints, before the first print. It keeps the stream's encoding and only
switches the error handler to ``"replace"``, so an unencodable character
becomes ``?`` instead of crashing the run.

The module is dependency-free and idempotent; it is safe under pytest's
captured (non-TextIOWrapper) streams.
"""
from __future__ import annotations

import sys


def safe_stdio() -> None:
    """Make ``sys.stdout`` / ``sys.stderr`` unable to raise UnicodeEncodeError.

    Idempotent: calling it multiple times (or re-calling it after the
    streams were swapped) just re-applies ``errors="replace"``. Streams
    without ``reconfigure`` (e.g. pytest capture objects) are skipped.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(errors="replace")
        except (ValueError, OSError):
            pass  # detached / non-reconfigurable stream -- nothing to do
