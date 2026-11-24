"""Hermetic tests for the cp1252-redirect trap and the stdio guard.

These pin BOTH the bug (a cp1252 stream raises on U+2500) and the fix
(safe_stdio makes the same print survive). No servers, no GPU, no network.
"""
import io
import sys

from tools.stdio_guard import safe_stdio


def test_cp1252_stream_raises_without_guard():
    """Pin the trap itself: cp1252 cannot encode the box-drawing U+2500 that
    killed runs/learnability_smoke.log (the round-header print)."""
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    try:
        stream.write("[learnability] " + "\u2500\u2500 round 1/15")
        stream.flush()
        raised = False
    except UnicodeEncodeError:
        raised = True
    finally:
        stream.detach()  # do not close the underlying BytesIO twice
    assert raised


def test_safe_stdio_makes_cp1252_stream_survive(monkeypatch):
    buf = io.BytesIO()
    stream = io.TextIOWrapper(buf, encoding="cp1252")
    monkeypatch.setattr(sys, "stdout", stream)
    monkeypatch.setattr(sys, "stderr", stream)
    safe_stdio()
    # Before the guard, the first print raised UnicodeEncodeError:
    print("[learnability] \u2500\u2500 round 1/15 (50 battles vs random) \u2500\u2500")
    # The em dash IS in cp1252 and always worked -- must still work:
    print("em dash fine: \u2014")
    stream.flush()
    stream.detach()
    out = buf.getvalue()
    assert b"round 1/15" in out
    assert b"em dash fine" in out


def test_safe_stdio_is_idempotent_and_tolerates_odd_streams(monkeypatch):
    class NoReconfigure:
        """Mimics pytest-style captured streams without reconfigure()."""
        def write(self, s):
            return len(s)
        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", NoReconfigure())
    monkeypatch.setattr(sys, "stderr", NoReconfigure())
    safe_stdio()
    safe_stdio()  # a second call must not raise either
