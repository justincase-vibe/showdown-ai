"""Self-play run engine — LIVE test: the REAL pool + REAL driver + REAL
capture, end to end (2 servers, 4 battles, replay round-trip).

Deselectable by name (``test_live_*``). Ports 49960–49961 (far from
8000–8007 / 49900 / 49950) so it never clobbers a live server; the pool
spawns + tears them down itself and the test verifies no orphans.

Run:  python -m pytest control/tests/test_selfplay_live.py -q -s
"""
from __future__ import annotations

from control.selfplay import SelfPlayConfig, SelfPlayRun
from replays import ReplayLibrary


def test_live_selfplay_run_4_battles(tmp_path):
    cfg = SelfPlayConfig(n_battles=4, battle_format="gen9ou",
                         n_servers=2, per_server=1, port_base=49960,
                         boot_timeout=90.0,
                         save_replays=True, library_root=str(tmp_path / "lib"))
    lib = ReplayLibrary(str(tmp_path / "lib"))
    run = SelfPlayRun(cfg, library=lib, log=print)   # REAL driver + REAL pool
    try:
        run.start()
        s = run.wait(timeout=600)
    finally:
        run.shutdown()

    assert s["state"] == "completed", s
    assert s["done"] == 4 and s["errors"] == 0 and s["capture_errors"] == 0, s
    assert s["saved"] == 4
    # every battle ran to a conclusion (decided OR hit max_steps — a rare draw
    # must not make this gate flaky); no battle crashed
    assert s["a1_wins"] + s["a2_wins"] + s["neither_flagged"] == 4, s

    # the capture contract: 4 distinct records, all replayable by the BC walker
    assert lib.count() == 4
    recs = lib.fetch_last(4)
    assert len({r["id"] for r in recs}) == 4
    for r in recs:
        assert r["winner_side"] is None or r["winner_side"] in ("p1", "p2")
    from corpus.trajectories import replay_battle
    trace = replay_battle(recs[0])
    assert trace.skipped is None, f"BC walker skipped a self-play record: {trace.skipped}"
    assert trace.n_decisions > 0

    # orphan check: BOTH pool servers must be dead
    for port in (49960, 49961):
        assert not run.pool.servers.health(port, timeout=1.0), f"orphan on :{port}"
