"""A3 — a lingering PLAY session must not hold the single active-run slot forever.

The gate is ``gui.launcher._assert_startable(runs_dir, mode)``:
  * refuse if a run is already active;
  * EXCEPT a NON-play start gracefully pre-empts an active PLAY session (play is
    the low-priority interactive mode, ``n_battles=0`` = "play until stopped");
  * a play start never pre-empts a training/eval/league/bc run;
  * two of the same workload still refuse;
  * a pre-empt only proceeds if the stop actually cleared the run (else refuse).

We monkeypatch ``find_active_run`` + ``stop_trainer`` so the test exercises the
DECISION without spawning a real child or killing a real process (the real
graceful-stop path is covered by the launcher's existing stop tests).

Run:  python -m pytest gui/tests/test_a3_play_preempt.py -v
"""
from __future__ import annotations

import pytest

from gui import launcher


def _active(rd):
    return (rd, {"state": "running", "pid": 999999})


def test_no_active_run_proceeds(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "find_active_run", lambda runs_dir: None)
    monkeypatch.setattr(launcher, "stop_trainer", lambda **k: pytest.fail("must not stop"))
    launcher._assert_startable(tmp_path / "runs", mode="eval")   # no raise


def test_nonplay_start_preempts_active_play(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    play_rd = runs / "playrun"
    play_rd.mkdir(parents=True)
    (play_rd / "mode").write_text("play", encoding="utf-8")
    state = {"active": _active(play_rd)}
    monkeypatch.setattr(launcher, "find_active_run", lambda runs_dir: state["active"])

    def fake_stop(runs_dir=None, timeout=180.0, force=False):
        state["active"] = None   # the play session is now stopped
        return {"run_id": "playrun", "state": "stopped"}
    monkeypatch.setattr(launcher, "stop_trainer", fake_stop)

    launcher._assert_startable(runs, mode="eval")   # no raise: pre-empted play
    assert state["active"] is None


def test_play_start_does_not_preempt_training(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    tr_rd = runs / "trainrun"
    tr_rd.mkdir(parents=True)
    (tr_rd / "mode").write_text("selfplay", encoding="utf-8")
    monkeypatch.setattr(launcher, "find_active_run", lambda runs_dir: _active(tr_rd))
    monkeypatch.setattr(launcher, "stop_trainer",
                        lambda **k: pytest.fail("must not kill a training run"))
    with pytest.raises(RuntimeError, match="already active"):
        launcher._assert_startable(runs, mode="play")


def test_nonplay_does_not_preempt_nonplay(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    tr_rd = runs / "trainrun"
    tr_rd.mkdir(parents=True)
    (tr_rd / "mode").write_text("selfplay", encoding="utf-8")
    monkeypatch.setattr(launcher, "find_active_run", lambda runs_dir: _active(tr_rd))
    monkeypatch.setattr(launcher, "stop_trainer",
                        lambda **k: pytest.fail("must not stop a second training run"))
    with pytest.raises(RuntimeError, match="already active"):
        launcher._assert_startable(runs, mode="eval")


def test_play_does_not_preempt_play(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    play_rd = runs / "playrun"
    play_rd.mkdir(parents=True)
    (play_rd / "mode").write_text("play", encoding="utf-8")
    monkeypatch.setattr(launcher, "find_active_run", lambda runs_dir: _active(play_rd))
    monkeypatch.setattr(launcher, "stop_trainer",
                        lambda **k: pytest.fail("must not kill a play session to start another"))
    with pytest.raises(RuntimeError, match="already active"):
        launcher._assert_startable(runs, mode="play")


def test_preempt_only_if_stop_actually_cleared(tmp_path, monkeypatch):
    """If stop_trainer does NOT clear the play session, start must REFUSE (safe)."""
    runs = tmp_path / "runs"
    play_rd = runs / "playrun"
    play_rd.mkdir(parents=True)
    (play_rd / "mode").write_text("play", encoding="utf-8")
    # find_active_run keeps returning the (still-active) play session
    monkeypatch.setattr(launcher, "find_active_run", lambda runs_dir: _active(play_rd))

    def fake_stop(runs_dir=None, timeout=180.0, force=False):
        return {"run_id": "playrun", "state": "stopped"}   # but it's still active
    monkeypatch.setattr(launcher, "stop_trainer", fake_stop)

    with pytest.raises(RuntimeError, match="could not stop"):
        launcher._assert_startable(runs, mode="eval")


def test_legacy_run_without_mode_is_not_preempted(tmp_path, monkeypatch):
    """A run started before ``mode`` was recorded (no 'mode' file) is treated as
    non-pre-emptible -> refuse (the safe, backward-compatible default)."""
    runs = tmp_path / "runs"
    legacy_rd = runs / "legacy"
    legacy_rd.mkdir(parents=True)   # no 'mode' file
    monkeypatch.setattr(launcher, "find_active_run", lambda runs_dir: _active(legacy_rd))
    monkeypatch.setattr(launcher, "stop_trainer",
                        lambda **k: pytest.fail("must not pre-empt a legacy run"))
    with pytest.raises(RuntimeError, match="already active"):
        launcher._assert_startable(runs, mode="eval")
