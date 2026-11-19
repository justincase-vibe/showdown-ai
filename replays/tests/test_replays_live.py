"""Phase 2 LIVE test: capture a REAL self-play battle into the replay library
and replay it through the REAL BC walker (``corpus.trajectories.replay_battle``).

This is the end-to-end contract the Phase 2 core promises: a battle played on
the pool becomes a library record that the BC pipeline can consume.

Deselectable by name (``test_live_*``). Needs node + the fork + poke_env.
One server on :49950 (far from 8000-8007 / 49900) so it never clobbers a live
server; shut down again afterwards.

Run:  python -m pytest replays/tests/test_replays_live.py -q -s
"""
from pool import PoolConfig, ResourcePool
from pool.battle import play_random_battle
from replays import ReplayLibrary, make_record


def test_live_capture_roundtrip(tmp_path):
    from net.vocab_db import build_vocabs
    from corpus.trajectories import replay_battle

    fmt = "gen9ou"
    cfg = PoolConfig(n_servers=1, port_base=49950, per_server=1, boot_timeout=90)
    rp = ResourcePool(cfg, log=print)
    vocabs = build_vocabs()
    lib = ReplayLibrary(str(tmp_path))
    captured = {}

    def on_finish(battle):
        # capture BEFORE teardown (the hook contract); a capture bug must not
        # kill the battle result. players=None -> the battle's REAL usernames
        # (poke-env generates unique names per env)
        try:
            rec = make_record(battle, fmt)
            captured["rec"] = rec
        except Exception as e:  # noqa: BLE001
            captured["error"] = f"{type(e).__name__}: {e}"

    try:
        status = rp.start(wait_healthy=True)
        assert all(status.values()), status
        a1_won, a2_won, n_steps = play_random_battle(
            rp, fmt, seed=777, vocabs=vocabs, on_finish=on_finish)
        assert ("rec" in captured), f"capture failed: {captured.get('error')}"
        assert n_steps > 0
        assert len(captured["rec"]["players"]) == 2
        assert len(set(captured["rec"]["players"])) == 2, captured["rec"]["players"]
        # the battle itself ended decisively (random vs random: no stalemates)
        assert a1_won != a2_won, (a1_won, a2_won)
    finally:
        rp.shutdown()

    rec = captured["rec"]
    # library round trip
    lib.save(rec)
    out = lib.fetch_last(1)[0]
    assert out["log"] == rec["log"]
    assert out["winner_side"] in ("p1", "p2")
    assert rec["winner_side"] == ("p1" if a1_won else "p2")

    # THE contract: the BC walker replays a self-play battle
    trace = replay_battle(out)
    assert trace.skipped is None, f"BC walker skipped the self-play battle: {trace.skipped}"
    assert trace.n_decisions > 0, "self-play battle yielded no decisions"

    # orphan check (the pool spawned :49950 itself)
    up = rp.servers.health(49950, timeout=1.0)
    assert not up, "orphaned server still listening on :49950"
