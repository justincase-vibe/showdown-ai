"""P4.3.1 gate — the training subprocess protocol (launcher + run-state).

Run from the project root:

    python -m pytest gui/tests/test_p43_1.py -v

The contract being pinned:

  * the run-state protocol (status / heartbeat / metrics / stop flag) is
    atomic, readable from ANY process, and reconciles runs whose process
    died without a final status (backstop: dead PID + no stop flag -> failed;
    dead PID + stop flag -> stopped);
  * ``validate_train_config`` rejects bad configs LOUDLY (every problem
    listed) before anything is spawned;
  * a live 1-battle CPU run through the REAL detached subprocess: start ->
    running -> completed, one metrics line with finite loss, a checkpoint in
    the run dir, a log file, and the second start REFUSED while active;
  * a graceful stop: the flag is honored BETWEEN battles (D80 — the in-flight
    battle finishes, the round completes atomically, the run exits 'stopped');
  * GUI-restart survival: a SEPARATE python process (a fresh GUI) sees the
    active run from disk alone and can stop it — the whole point of the
    detached design.

Live tests run ~2-3 real CPU battles total (the established 1-battle smoke
pattern from p41_gate) — they prove the loop works, not that the net is good.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]  # gui/tests/x.py -> project root
sys.path.insert(0, str(ROOT))

from gui import launcher, runstate  # noqa: E402

# ── helpers ──────────────────────────────────────────────────────────────────

def wait_for(predicate, timeout: float = 300.0, interval: float = 2.0,
             what: str = "condition"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if predicate():
            return
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what}")


def small_cfg(**over) -> dict:
    cfg = {"battle_format": "gen9ou", "battles_per_round": 1, "n_rounds": 1,
           "max_steps": 200, "seed": 42, "device": "cpu", "save_every": 1,
           "epochs": 1, "lr": 3e-4}
    cfg.update(over)
    return cfg


@pytest.fixture
def runs(tmp_path):
    d = tmp_path / "runs"
    d.mkdir()
    return d


# ── protocol (fast, no battles) ──────────────────────────────────────────────

def test_runstate_protocol_roundtrip(runs):
    rd = runs / "r1"
    rd.mkdir()
    runstate.write_status(rd, state="starting", run_id="r1", pid=1234)
    runstate.write_heartbeat(rd, 0, 3, "test")
    runstate.append_metric(rd, {"round": 1, "loss": 0.5})
    runstate.append_metric(rd, {"round": 2, "loss": 0.4})

    st = runstate.read_status(rd)
    assert st["state"] == "starting" and st["pid"] == 1234
    assert st["updated"] and st["run_id"] == "r1"
    assert runstate.read_heartbeat(rd)["battle"] == 3
    assert [r["round"] for r in runstate.read_metrics(rd)] == [1, 2]
    assert runstate.read_metrics(rd, 1)[0]["round"] == 2  # tail

    runstate.request_stop(rd)
    assert runstate.stop_requested(rd)
    # merge semantics: a later write keeps earlier fields
    st2 = runstate.write_status(rd, state="running")
    assert st2["run_id"] == "r1" and st2["state"] == "running"


def test_runstate_torn_read_is_not_a_crash(runs):
    rd = runs / "r2"
    rd.mkdir()
    (rd / "status.json").write_text('{"state": "run', encoding="utf-8")  # torn
    assert runstate.read_status(rd) == {}
    (rd / "metrics.jsonl").write_text('{"round": 1, "loss": 0.5}\n{"round": 2,', encoding="utf-8")
    rows = runstate.read_metrics(rd)
    assert len(rows) == 1 and rows[0]["round"] == 1  # the torn line is skipped


def test_reconcile_pid_none_is_never_finalized(runs):
    """D80: a 'starting' status WITHOUT a pid is the launcher mid-spawn —
    reconcile must leave it alone (finalizing it would mark a live
    just-started run 'failed')."""
    rd = runs / "r6"
    rd.mkdir()
    runstate.write_status(rd, state="starting", run_id="r6", pid=None)
    st = runstate.reconcile(rd)
    assert st["state"] == "starting" and not st.get("error")


def test_reconcile_dead_pid_finalizes(runs):
    # a real process that is guaranteed dead by the time we reconcile
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    time.sleep(1.5)
    assert runstate.pid_alive(p.pid)
    p.kill(); p.wait()
    time.sleep(0.5)

    # no stop flag -> failed
    rd = runs / "r3"
    rd.mkdir()
    runstate.write_status(rd, state="running", pid=p.pid)
    st = runstate.reconcile(rd)
    assert st["state"] == "failed" and st.get("error")

    # stop flag -> stopped
    rd2 = runs / "r4"
    rd2.mkdir()
    runstate.write_status(rd2, state="running", pid=p.pid)
    runstate.request_stop(rd2)
    st2 = runstate.reconcile(rd2)
    assert st2["state"] == "stopped"

    # a live PID is left alone
    p2 = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        rd3 = runs / "r5"
        rd3.mkdir()
        runstate.write_status(rd3, state="running", pid=p2.pid)
        assert runstate.reconcile(rd3)["state"] == "running"
    finally:
        p2.kill(); p2.wait()


def test_validate_train_config_loud():
    good = small_cfg()
    tc = launcher.validate_train_config(good)
    assert tc.battles_per_round == 1 and tc.device == "cpu" and tc.n_rounds == 1

    bad = small_cfg()
    bad.update({"battles_per_round": 0, "device": "gpu", "gamma": 2.0,
                "epochs": "x", "resume_from": "/nonexistent/ckpt.pt"})
    with pytest.raises(ValueError) as ei:
        launcher.validate_train_config(bad)
    msg = str(ei.value)
    # EVERY problem is reported, not just the first
    for needle in ("battles_per_round", "device", "gamma", "epochs", "resume_from"):
        assert needle in msg, f"expected {needle!r} in: {msg}"

    # unknown keys are tolerated (forward-compat), known-but-wrong types caught
    tc2 = launcher.validate_train_config({**good, "future_field": 123})
    assert tc2.battles_per_round == 1


def test_find_active_none_with_no_runs(runs):
    assert launcher.find_active_run(runs) is None
    st = launcher.status(runs)
    assert st["active"] is False and st["state"] is None and st["run_id"] is None


# ── live: real detached subprocess (CPU, 1 battle each) ─────────────────────

def test_live_start_refuse_second_and_completed(runs):
    """One real 1-battle CPU run through the REAL detached subprocess:
    start -> running -> completed, with metrics + checkpoint + log, and the
    second start REFUSED while the first is active."""
    info = launcher.start_trainer(small_cfg(), runs_dir=runs)
    rd = Path(info["run_dir"])
    assert (rd / "config.json").exists()
    assert (rd / "trainer.log").exists()

    # while active, a second start must be refused
    wait_for(lambda: launcher.find_active_run(runs) is not None, 60, what="active run")
    with pytest.raises(RuntimeError, match="already active"):
        launcher.start_trainer(small_cfg(), runs_dir=runs)

    # wait for completion (1 CPU battle + init; generous bound)
    def final():
        st = launcher.status(runs)
        return st["state"] in runstate.FINAL_STATES
    wait_for(final, 420, what="run completion")

    st = launcher.status(runs)
    assert st["state"] == "completed", f"state={st['state']} error={st.get('error')}"
    assert st["round"] == 1 and st["rounds_planned"] == 1
    assert st["active"] is False
    assert runstate.read_status(rd).get("exit_code") == 0

    rows = runstate.read_metrics(rd)
    assert len(rows) == 1, f"expected 1 metrics line, got {len(rows)}: {rows}"
    r1 = rows[0]
    assert r1["round"] == 1 and r1["battles"] == 1
    assert r1["transitions"] >= 2
    for k in ("loss", "value", "entropy"):
        assert r1.get(k) is not None and __import__("math").isfinite(r1[k]), \
            f"metric {k} missing/non-finite: {r1.get(k)}"

    ckpts = runstate.checkpoints(rd)
    assert ckpts == ["selfplay_round1.pt"], f"checkpoints: {ckpts}"
    assert (rd / "checkpoints" / "selfplay_round1.pt").stat().st_size > 0

    log_text = (rd / "trainer.log").read_text(encoding="utf-8", errors="replace")
    assert "completed:" in log_text  # the trainer's final loud line


def test_graceful_stop_between_rounds(runs):
    """A stop requested MID-ROUND is honored at the NEXT round boundary (the
    documented contract: the in-flight battle finishes — no partial update):
    round 1 completes, round 2 is skipped, clean 'stopped', exit 0, exactly
    ONE metrics line. Waiting for the round-1 heartbeat makes the ordering
    deterministic (no start/stop race)."""
    info = launcher.start_trainer(small_cfg(n_rounds=2), runs_dir=runs)
    rd = Path(info["run_dir"])
    # wait until the trainer is provably INSIDE round 1 (its per-battle
    # heartbeat has fired) — then a stop can only land at the round 2 boundary
    def in_round1():
        hb = runstate.read_heartbeat(rd)
        return bool(hb.get("note")) and hb.get("round") == 0
    wait_for(in_round1, 90, what="round-1 heartbeat")

    launcher.stop_trainer(runs_dir=runs, timeout=120)
    wait_for(lambda: runstate.read_status(rd)["state"] in runstate.FINAL_STATES,
             180, what="final state after stop")

    st = runstate.read_status(rd)
    assert st["state"] == "stopped", f"state={st['state']} error={st.get('error')}"
    assert st.get("exit_code") == 0
    rows = runstate.read_metrics(rd)
    assert len(rows) == 1 and rows[0]["round"] == 1, \
        f"round 1 should complete, round 2 be skipped: {rows}"
    assert not runstate.pid_alive(st.get("pid"))


def _fresh_process(code: str, runs_s: str) -> str:
    """Run ``code`` in a SEPARATE python process (a fresh GUI) with the project
    root on sys.path and the runs dir as argv[2] — returns stdout (stripped)."""
    r = subprocess.run([sys.executable, "-c", code, str(ROOT), runs_s],
                       capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0, f"fresh process failed: {r.stderr}"
    return r.stdout.strip()


def test_live_graceful_stop_mid_round(runs):
    """D80 live: with battles_per_round=3, a graceful stop requested right
    after battle 0 has started must land BETWEEN battles — the in-flight
    battle finishes, the remaining battles of the round are skipped, and the
    round still completes ATOMICALLY (update + checkpoint on what was
    collected). Pre-D80 this config would run ALL 3 battles and report
    'completed' (the flag was only checked at round boundaries — with
    n_rounds=1 it would be ignored entirely)."""
    info = launcher.start_trainer(small_cfg(battles_per_round=3, n_rounds=1),
                                  runs_dir=runs)
    rd = Path(info["run_dir"])

    # wait until battle 0 is provably IN FLIGHT (its pre-battle heartbeat
    # fired) — a stop now can only land between battles 0 and 1 (or, if
    # battle 0 is fast and already done, between 1 and 2)
    def battle0_started():
        hb = runstate.read_heartbeat(rd)
        return bool(hb.get("note")) and hb.get("battle") == 0
    wait_for(battle0_started, 90, what="battle-0 heartbeat")

    launcher.stop_trainer(runs_dir=runs, timeout=170)
    wait_for(lambda: runstate.read_status(rd)["state"] in runstate.FINAL_STATES,
             180, what="final state after mid-round stop")

    st = runstate.read_status(rd)
    # THE discriminator: pre-D80 this would be 'completed' with battles=3
    assert st["state"] == "stopped", f"state={st['state']} error={st.get('error')}"
    assert st.get("exit_code") == 0
    rows = runstate.read_metrics(rd)
    assert len(rows) == 1, f"exactly one (atomic) round: {rows}"
    r1 = rows[0]
    assert r1["battles"] in (1, 2), f"in-flight battle finishes, rest skipped: {r1}"
    assert r1["battles"] < 3, f"the stop must have been honored inside the round: {r1}"
    assert r1.get("stopped_early") is True
    # the round completed atomically: the checkpoint for it EXISTS
    assert (rd / "checkpoints" / "selfplay_round1.pt").exists()
    assert not runstate.pid_alive(st.get("pid"))


def test_gui_restart_survival(runs):
    """The detached guarantee, tested for real: a SEPARATE python process
    (a fresh GUI) sees the active run from disk alone and can stop it."""
    launcher.start_trainer(small_cfg(n_rounds=2), runs_dir=runs)
    rd = Path(launcher.status(runs)["run_id"] and (runs / launcher.status(runs)["run_id"]))
    runs_s = str(runs)

    status_code = (
        "import json, sys, pathlib\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from gui import launcher\n"
        "s = launcher.status(runs_dir=pathlib.Path(sys.argv[2]))\n"
        "print(json.dumps({'active': s['active'], 'state': s['state'], 'pid': s['pid']}))\n"
    )

    def active_from_disk() -> bool:
        out = json.loads(_fresh_process(status_code, runs_s).splitlines()[-1])
        return bool(out["active"]) and out["pid"] is not None

    wait_for(active_from_disk, 120, what="fresh process sees the active run")

    # stop from yet another fresh process (cross-process control via disk + PID)
    stop_code = (
        "import sys, pathlib\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from gui import launcher\n"
        "print(launcher.stop_trainer(runs_dir=pathlib.Path(sys.argv[2]))['state'])\n"
    )
    assert _fresh_process(stop_code, runs_s).splitlines()[-1] == "stopped"

    st = launcher.status(runs)
    assert st["state"] == "stopped"
    assert st["active"] is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
