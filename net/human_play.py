"""D92 Step 2 — PLAY against a checkpoint (a poke-env Player driven by PokeNet).

This is the interactive half of "fight a checkpoint": the user's browser is ONE
side of a battle on the LOCAL fork (``http://localhost:8000``), and this process
is the OTHER side — a ``poke_env`` :class:`Player` whose ``choose_move`` runs the
loaded :class:`net.model.PokeNet`. It is the new-architecture equivalent of the
legacy ``bot/play.py`` (which ran the old MaskedPPO), but it reuses the EXACT
training decision path so what you play is the checkpoint you trained:

    your move  ->  fork battle log
                ->  net.env.build_obs  (belief -> encoder -> registry, POV-correct)
                ->  net.env.build_action_list  (the current legal D60 actions)
                ->  net.agent.PokeAgent.act  (greedy argmax or sample)
                ->  net.env.action_to_pe  (D60 action -> poke-env int)
                ->  SinglesEnv.action_to_order  (poke-env int -> BattleOrder)

It speaks the SAME ``gui.runstate`` protocol as the other GUI entries, so the
dashboard's status / heartbeat / stop / history all work unchanged. The GUI
(D92 step 3) spawns it as::

    python -m net.human_play --config <run-dir>/config.json --run-dir <run-dir>

Config (a FLAT dict): ``ckpt_path``, ``battle_format``, ``bot_user``, ``you``,
``team_seed``, ``device`` (cpu/cuda), ``greedy`` (default True), ``n_battles``
(0 = play until you stop it).

CPU-first: the bot runs on CPU by default so your GPU stays free for the LLM.
Nothing here touches the training code — this only READS a checkpoint and REUSES
``net.env`` / ``net.agent`` (pure, additive).
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any, Dict, Optional

from poke_env.player.player import Player


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _fork_reachable(host: str = "127.0.0.1", port: int = 8000, timeout: float = 1.5) -> bool:
    """A quick TCP probe — fail fast (with a clear message) if the local fork
    isn't up, instead of hanging on the websocket connect."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _load_checkpoint(ckpt_path: str, vocabs: Any, device: str):
    """Load a PokeNet checkpoint (BOTH formats) + a ready PokeAgent.

    * BC checkpoints:        ``payload = {state_dict, vocab_sizes, layout_checksum}``
    * Self-play checkpoints: ``payload = {"net": {state_dict, ...}, "round", ...}``

    Mirrors ``gui/eval_main.py`` so the load path is one, not two.
    """
    import torch
    from net.model import PokeNet
    from net.agent import PokeAgent

    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "net" in payload and isinstance(payload["net"], dict) and "state_dict" in payload["net"]:
        net = PokeNet.load(payload["net"], vocabs)
    else:
        net = PokeNet.load(payload, vocabs)
    net.eval()
    agent = PokeAgent(net, seed=None, device=device)
    return net, agent


#: poke-env “pass / no decision this turn” sentinel (the same value the trainer uses).
PASS = -2


def decide_order(agent: Any, battle: Any, fmt: str, vocabs: Any,
                 greedy: bool = True) -> int:
    """The pure decision path (POV-correct): build_obs -> act -> action_to_pe.

    Returns a poke-env int (0..5 switch, 6..9 move, 22..25 tera) or :data:`PASS`
    when there is no legal action this turn (a wait). It does NOT call poke-env's
    ``action_to_order`` (that needs a real battle) — ``CheckpointPlayer.choose_move``
    bridges the int -> ``BattleOrder``. Extracted so the net->legal-order path is
    unit-testable on a fake battle WITHOUT a running fork.
    """
    from net.env import action_to_pe, build_action_list, build_obs, my_side

    lines = [ln for ln in battle._build_replay_log().split("\n") if ln.strip()]
    side = my_side(battle, lines)
    obs = build_obs(battle, fmt, vocabs, side)
    acts = build_action_list(battle)
    d = agent.act(obs, acts, deterministic=greedy)
    if d.action is None:
        return PASS
    pe = action_to_pe(d.action, battle)
    return pe if pe >= 0 else PASS


class CheckpointPlayer(Player):
    """A poke-env :class:`Player` that plays with a loaded :class:`PokeNet`.

    ``choose_move`` is the ONLY policy hook poke-env calls each turn — everything
    else is poke-env's (connect, receive the battle, send the order). POV is
    handled by ``net.env.my_side`` (the belief pipeline is p1-POV, so a bot
    sitting on p2 must swap). ``Player`` is imported at module level — a pure
    library import that needs no running fork (only *connecting* does).
    """

    def __init__(self, *, net: Any, agent: Any, vocabs: Any, fmt: str,
                 bot_user: str, team: Any, greedy: bool = True,
                 log_level: Optional[int] = None):
        import logging
        from poke_env.ps_client import (
            AccountConfiguration,
            LocalhostServerConfiguration,
        )
        from poke_env.player.battle_order import DefaultBattleOrder

        self._net = net
        self._agent = agent
        self._vocabs = vocabs
        self._fmt = fmt
        self._greedy = bool(greedy)
        self._DefaultBattleOrder = DefaultBattleOrder

        super().__init__(
            AccountConfiguration(bot_user, ""),
            battle_format=fmt,
            team=team,
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=1,
            start_listening=True,          # auto-connect on the shared POKE_LOOP
            log_level=log_level or logging.WARNING,
        )

    # ── the one policy hook poke-env calls each of our turns ───────────────
    def choose_move(self, battle: Any) -> Any:
        from poke_env.environment.singles_env import SinglesEnv
        import numpy as np

        try:
            pe = decide_order(self._agent, battle, self._fmt, self._vocabs, self._greedy)
            if pe < 0:
                return self._pass_order(battle)
            # strict=False: a momentarily-stale pick falls back to a valid move
            # instead of aborting the live battle (the same D60-env discipline).
            return SinglesEnv.action_to_order(np.int64(pe), battle, strict=False)
        except Exception as exc:  # noqa: BLE001 — a live battle must never crash on obs
            _log(f"choose_move error ({type(exc).__name__}: {exc}) - passing the turn")
            return self._pass_order(battle)

    def _pass_order(self, battle: Any) -> Any:
        orders = list(getattr(battle, "valid_orders", None) or [])
        return orders[0] if orders else self._DefaultBattleOrder()


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="net.human_play")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args(argv)

    rd = Path(args.run_dir).resolve()
    cfg = json.loads(Path(args.config).resolve().read_text(encoding="utf-8"))
    from gui import runstate
    runstate.harden_stdio()   # a non-ASCII log char must never kill the bot on cp1252

    ckpt_path = str(cfg.get("ckpt_path", ""))
    fmt = str(cfg.get("battle_format", "gen9ou") or "gen9ou")
    bot_user = str(cfg.get("bot_user", "checkpointbot") or "checkpointbot")
    you = str(cfg.get("you", "") or "").strip()
    team_seed = cfg.get("team_seed")
    device = str(cfg.get("device", "cpu") or "cpu")
    greedy = bool(cfg.get("greedy", True))
    n_battles = int(cfg.get("n_battles", 0) or 0)   # 0 = play until you stop it

    # ── guards (fail fast, on disk, before any connection) ────────────────
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            runstate.write_status(rd, state="failed",
                                  error="device='cuda' requested but torch.cuda.is_available() is False",
                                  exit_code=1)
            _log("FATAL: cuda requested but unavailable")
            return 1
    if not _fork_reachable():
        runstate.write_status(rd, state="failed",
                              error="local Showdown fork not reachable on localhost:8000 "
                                    "(start it first:  start_play_server.bat)",
                              exit_code=1)
        _log("FATAL: fork not reachable on localhost:8000")
        return 1

    runstate.write_status(rd, state="running", started=runstate.now_iso())
    runstate.write_heartbeat(rd, 0, 0, "loading checkpoint")
    _log(f"PLAY start run={rd.name} ckpt={ckpt_path} fmt={fmt} bot='{bot_user}' "
         f"you='{you}' greedy={greedy} n={n_battles if n_battles else 'unlimited'} device={device}")

    # ── load the net + build the team ─────────────────────────────────────
    from net.vocab_db import build_vocabs
    from net.env import StrongTeamBuilder
    try:
        vocabs = build_vocabs()
        net, agent = _load_checkpoint(ckpt_path, vocabs, device)
        team = StrongTeamBuilder(fmt, seed=(int(team_seed) if team_seed is not None else None))
        player = CheckpointPlayer(net=net, agent=agent, vocabs=vocabs, fmt=fmt,
                                  bot_user=bot_user, team=team, greedy=greedy)
    except Exception as exc:  # noqa: BLE001
        runstate.write_status(rd, state="failed",
                              error=f"setup failed: {type(exc).__name__}: {exc}", exit_code=1)
        _log(f"FATAL: {type(exc).__name__}: {exc}")
        return 1

    _log(f"connected as '{bot_user}'")
    if you:
        _log(f"in your browser, log in as '{you}' and type:   /challenge {bot_user}")
    else:
        _log(f"in your browser, log in and type:              /challenge {bot_user}")
    runstate.write_heartbeat(rd, 0, 0, f"connected as {bot_user} — waiting for a challenge")

    # ── poll loop: log each finished battle, honor stop.flag, exit on n ───
    done_before = int(player.n_finished_battles)
    results: Dict[str, int] = {"wins": 0, "losses": 0, "draws": 0}
    n_done = 0
    try:
        while True:
            time.sleep(1.0)
            if player.n_finished_battles > done_before:
                b = list(player.battles.values())[-1]
                won, lost = bool(b.won), bool(b.lost)
                res = {"won": won, "lost": lost, "draw": (not won and not lost),
                       "turns": int(getattr(b, "turn", 0) or 0), "ts": time.time()}
                n_done += 1
                if won: results["wins"] += 1
                elif lost: results["losses"] += 1
                else: results["draws"] += 1
                _log(f"battle {n_done}: bot {'WON' if won else 'LOST' if lost else 'DREW'} "
                     f"in {res['turns']} turns")
                runstate.append_metric(rd, res)
                runstate.write_heartbeat(rd, 0, n_done, f"battle {n_done} done")
                done_before = player.n_finished_battles
            if n_battles and n_done >= n_battles:
                break
            if runstate.stop_requested(rd):
                _log("stop requested - exiting (the fork connection tears down with the process)")
                break
    except Exception as exc:  # noqa: BLE001
        runstate.write_status(rd, state="failed",
                              error=f"play loop crashed: {type(exc).__name__}: {exc}", exit_code=1)
        _log(f"FATAL: {type(exc).__name__}: {exc}")
        return 1

    state = "stopped" if runstate.stop_requested(rd) else "completed"
    runstate.write_status(rd, state=state, exit_code=0)
    runstate.write_heartbeat(rd, 0, 0, state)
    _log(f"{state}: bot {results['wins']}W / {results['losses']}L / {results['draws']}D "
         f"over {n_done} battle(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
