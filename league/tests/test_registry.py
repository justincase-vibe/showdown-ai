"""League bot registry tests — add, remove, ELO update, standings, persistence."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from league.registry import BotRegistry, Bot, ELO_INITIAL, ELO_K


@pytest.fixture
def reg(tmp_path):
    return BotRegistry(path=tmp_path / "bots.json")


def test_add_and_get(reg):
    reg.add("alice", "scratch/50b_bc_a3f2.pt")
    bot = reg.get("alice")
    assert bot.name == "alice"
    assert bot.ckpt_path == "scratch/50b_bc_a3f2.pt"
    assert bot.elo == ELO_INITIAL
    assert bot.wins == 0


def test_remove(reg):
    reg.add("alice", "a.pt")
    reg.remove("alice")
    with pytest.raises(KeyError):
        reg.get("alice")
    assert len(reg) == 0


def test_remove_missing_raises(reg):
    with pytest.raises(KeyError, match="not in roster"):
        reg.remove("ghost")


def test_update_result_win(reg):
    reg.add("alice", "a.pt")
    reg.add("bob", "b.pt")
    reg.update_result("alice", "bob", winner="alice")
    assert reg.get("alice").wins == 1
    assert reg.get("bob").losses == 1
    # alice's ELO went up, bob's went down
    assert reg.get("alice").elo > ELO_INITIAL
    assert reg.get("bob").elo < ELO_INITIAL
    # total ELO is conserved (zero-sum)
    assert reg.get("alice").elo + reg.get("bob").elo == 2 * ELO_INITIAL


def test_update_result_loss(reg):
    reg.add("alice", "a.pt")
    reg.add("bob", "b.pt")
    reg.update_result("alice", "bob", winner="bob")
    assert reg.get("bob").wins == 1
    assert reg.get("alice").losses == 1
    assert reg.get("bob").elo > ELO_INITIAL
    assert reg.get("alice").elo < ELO_INITIAL


def test_update_result_draw(reg):
    reg.add("alice", "a.pt")
    reg.add("bob", "b.pt")
    reg.update_result("alice", "bob", winner=None)
    assert reg.get("alice").draws == 1
    assert reg.get("bob").draws == 1
    # ELO unchanged for a draw (both start at the same rating)
    assert reg.get("alice").elo == ELO_INITIAL
    assert reg.get("bob").elo == ELO_INITIAL


def test_elo_magnitudes_standard(reg):
    """Exact standard ELO magnitudes (K=32), pinned against the formula.

    A=1500 vs B=1700: E_A = 1/(1+10^(200/400)) = 0.24025...
      A wins -> A +round(32*(1-E_A)) = +24,  B -24
      B wins -> A +round(32*(0-E_A)) = -8,   B +8   (upset reward)
      draw   -> A +round(32*(0.5-E_A)) = +8, B -8
    The OLD loss branch used ``-K*(1-E_A)`` (giving -24/+24 instead of
    -8/+8) — the direction tests above cannot catch that; this one can.
    """
    reg.add("a", "a.pt", elo=1500)
    reg.add("b", "b.pt", elo=1700)
    reg.update_result("a", "b", winner="a")
    assert reg.get("a").elo == 1524
    assert reg.get("b").elo == 1676


def test_elo_magnitudes_underdog_win(reg):
    """An underdog win must pay the upset premium (K*(1-E_underdog))."""
    reg.add("a", "a.pt", elo=1500)
    reg.add("b", "b.pt", elo=1700)
    reg.update_result("a", "b", winner="b")          # b (favorite) wins
    assert reg.get("a").elo == 1492                    # -8, not -24
    assert reg.get("b").elo == 1708

    reg.add("c", "c.pt", elo=1700)
    reg.update_result("a", "c", winner="a")          # a (underdog) wins
    # a is now 1492 vs c's 1700 -> E_a=0.2319 -> upset premium round(32*0.768)=+25
    assert reg.get("a").elo == 1492 + 25
    assert reg.get("c").elo == 1700 - 25


def test_elo_draw_unequal_ratings(reg):
    """A draw between UNEQUAL ratings shifts ELO toward the favorite's
    expectation (standard ELO: score 0.5 for both)."""
    reg.add("a", "a.pt", elo=1500)
    reg.add("b", "b.pt", elo=1700)
    reg.update_result("a", "b", winner=None)
    assert reg.get("a").elo == 1508                    # +8
    assert reg.get("b").elo == 1692                    # -8


def test_elo_convergence(reg):
    """A bot that wins repeatedly should climb in ELO; one that loses
    should drop. After 10 wins/losses, the gap should be significant."""
    reg.add("strong", "s.pt")
    reg.add("weak", "w.pt")
    for _ in range(10):
        reg.update_result("strong", "weak", winner="strong")
    assert reg.get("strong").elo > ELO_INITIAL + 50
    assert reg.get("weak").elo < ELO_INITIAL - 50


def test_standings(reg):
    reg.add("alice", "a.pt")
    reg.add("bob", "b.pt")
    reg.add("carol", "c.pt")
    reg.update_result("alice", "bob", winner="alice")
    reg.update_result("alice", "carol", winner="carol")
    s = reg.standings()
    assert len(s) == 3
    # alice won 1, lost 1 → ELO near initial
    # bob lost 1 → ELO below initial
    # carol won 1 → ELO above initial
    assert s[0]["name"] == "carol"  # highest ELO
    assert s[2]["name"] == "bob"    # lowest ELO


def test_persistence(tmp_path):
    """The roster survives a save + reload."""
    path = tmp_path / "bots.json"
    reg1 = BotRegistry(path=path)
    reg1.add("alice", "a.pt")
    reg1.add("bob", "b.pt")
    reg1.update_result("alice", "bob", winner="alice")

    reg2 = BotRegistry(path=path)
    assert len(reg2) == 2
    assert reg2.get("alice").wins == 1
    assert reg2.get("bob").losses == 1
    assert reg2.get("alice").elo == reg1.get("alice").elo


def test_len_and_contains(reg):
    reg.add("alice", "a.pt")
    assert len(reg) == 1
    assert "alice" in reg
    assert "bob" not in reg


def test_names(reg):
    reg.add("alice", "a.pt")
    reg.add("bob", "b.pt")
    assert set(reg.names()) == {"alice", "bob"}


def test_concurrent_updates_consistent(tmp_path):
    """Concurrent update_result must not lose updates or corrupt the file.

    The league runs battles on multiple worker threads (see
    ``control/selfplay.py::SelfPlayRun.start``), all of which call
    ``update_result -> save`` concurrently on the shared registry. This hammers
    the registry from several threads — with OVERLAPPING bot pairs, so the
    read-moduate-persist must be atomic — and asserts the invariants a race
    would break:
      * ELO is zero-sum conserved (sum == N * initial) — a lost update breaks it
      * every battle is counted exactly once (W/L balanced, draws in pairs)
      * the persisted file is valid JSON and matches the in-memory state
    """
    path = tmp_path / "bots.json"
    reg = BotRegistry(path=path)

    n_bots = 6
    names = [f"bot{i}" for i in range(n_bots)]
    for n in names:
        reg.add(n, f"{n}.pt")
    total_initial_elo = n_bots * ELO_INITIAL

    n_threads = 8
    per_thread = 30
    total_battles = n_threads * per_thread
    barrier = threading.Barrier(n_threads)
    errors: list = []

    def worker(tid: int) -> None:
        try:
            barrier.wait()  # start all threads together to maximize contention
        except Exception as e:  # pragma: no cover - barrier only fails on teardown
            errors.append(e)
            return
        try:
            for i in range(per_thread):
                a = names[(tid + i) % n_bots]
                b = names[(tid + i + 1) % n_bots]  # always distinct (n_bots > 1)
                r = (tid * per_thread + i) % 3      # exercise win / loss / draw
                winner = a if r == 0 else (b if r == 1 else None)
                reg.update_result(a, b, winner=winner)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"worker errors: {errors}"

    # Zero-sum ELO conserved (a lost-update race breaks this).
    total_elo = sum(reg.get(n).elo for n in names)
    assert total_elo == total_initial_elo, (
        f"ELO not conserved: {total_elo} != {total_initial_elo} (a lost update)"
    )

    # Every battle counted exactly once (a dropped battle breaks this).
    total_wins = sum(reg.get(n).wins for n in names)
    total_losses = sum(reg.get(n).losses for n in names)
    total_draws = sum(reg.get(n).draws for n in names)
    assert total_wins == total_losses, f"win/loss imbalance: {total_wins} vs {total_losses}"
    assert total_draws % 2 == 0, f"draws not in pairs: {total_draws}"
    n_won = total_wins
    n_draw = total_draws // 2
    assert n_won + n_draw == total_battles, (
        f"battle count mismatch: {n_won} W/L + {n_draw} draws != {total_battles} total"
    )

    # The persisted file is valid JSON and matches in-memory state (a torn
    # concurrent write breaks this).
    data = json.loads(path.read_text(encoding="utf-8"))
    persisted = {b["name"]: b for b in data["bots"]}
    assert set(persisted) == set(names), "persisted roster names mismatch"
    for n in names:
        m, p = reg.get(n), persisted[n]
        assert p["elo"] == m.elo, f"{n}: file elo {p['elo']} != memory {m.elo}"
        assert p["wins"] == m.wins and p["losses"] == m.losses and p["draws"] == m.draws

    # A fresh registry reloads the exact same state.
    reg2 = BotRegistry(path=path)
    for n in names:
        assert reg2.get(n).elo == reg.get(n).elo
        assert reg2.get(n).wins == reg.get(n).wins
