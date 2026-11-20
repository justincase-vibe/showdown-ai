"""Play HUMAN vs the trained NN bot on play.pokemonshowdown.com.

The bot (ttvJustinCaseBOT) challenges you for each scheduled match:
    1x gen9ou  (strong team)
    1x gen8ou  (strong team)
    2x gen9randombattle
    1x gen8randombattle
You accept each challenge in your browser. The bot plays with the trained
policy (greedy/argmax -- deterministic exhibition play).

Key design: ONE Player instance across all matches. Teams are swapped with
update_team(); the format is passed explicitly to ps_client.challenge().
This avoids the same-username kick that happens when you create a new
Player per match.

Run:
    python bot/play.py --you ttvJustinCase --bot-user ttvJustinCaseBOT --bot-pass 123456
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from obs import OBS_DIM, embed_battle as _embed_raw          # noqa: E402
from action_space import get_action_space                     # noqa: E402
from policy import MaskedPPO                                  # noqa: E402
from state_from_log import reconstruct_current_state          # noqa: E402
from env import ShowdownEnv, _to_pe_action, _pe_valid         # noqa: E402
from teams import StrongTeamBuilder                           # noqa: E402

from poke_env.player.player import Player                     # noqa: E402
from poke_env.ps_client import AccountConfiguration, ServerConfiguration  # noqa: E402
from poke_env.concurrency import POKE_LOOP                    # noqa: E402
from poke_env.player.battle_order import (                    # noqa: E402
    DefaultBattleOrder,
    ForfeitBattleOrder,
)
from poke_env.environment.singles_env import SinglesEnv       # noqa: E402

# (format, strong-team?)
MATCHES = [
    ("gen9ou", True),
    ("gen8ou", True),
    ("gen9randombattle", False),
    ("gen9randombattle", False),
    ("gen8randombattle", False),
]

MAX_REASK = 8
BATTLE_TIMEOUT = 30 * 60      # 30 min per match
CHALLENGE_TIMEOUT = 120       # 2 min to get the challenge through


# --------------------------------------------------------------------- bot
class NNPlayer(Player):
    """Single persistent Player that plays with the trained policy.

    Starts in _forfeit_mode so any stale battles from previous sessions
    get forfeited on re-announcement. After cleanup, _forfeit_mode is
    cleared and the bot plays normally.
    """

    def __init__(self, battle_format: str, team, username: str, policy: MaskedPPO,
                 ws_url: str, password: str = ""):
        super().__init__(
            AccountConfiguration(username, password),
            battle_format=battle_format,
            team=team,
            server_configuration=ServerConfiguration(
                ws_url,
                "https://play.pokemonshowdown.com/action.php?",
            ),
            max_concurrent_battles=1,
            log_level=logging.WARNING,
            save_replays=os.path.join(HERE, "replays"),
        )
        self.policy = policy
        self.policy.actor.eval()
        self._forfeit_mode = True  # clean up stale battles first

    def choose_move(self, battle):
        # --- cleanup phase: forfeit any stale battle the server re-announces ---
        if self._forfeit_mode:
            print("  NNbot: [cleanup] forfeiting stale battle")
            return ForfeitBattleOrder()

        policy = self.policy
        device = policy.device

        # --- observation: same reconstruction path as training (P1 perspective) ---
        try:
            log = battle._build_replay_log()
            gen = 8
            for ln in log.splitlines():
                if ln.startswith("|gen|"):
                    try:
                        gen = int(ln.split("|")[2])
                    except (ValueError, IndexError):
                        pass
                    break
            obs = reconstruct_current_state(log, gen, "p1")
        except Exception:
            obs = _embed_raw(battle)

        # --- legal-action mask (963 move-identity space) ---
        m_np = np.asarray(ShowdownEnv.get_action_mask(battle), dtype=bool)
        if int(m_np.sum()) == 0:
            print("  NNbot: waiting (no decision this turn)")
            return DefaultBattleOrder()

        # --- greedy argmax under the mask, re-ask if the server rejects a pick ---
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        m_t = torch.as_tensor(m_np, dtype=torch.bool, device=device)
        chosen = None
        for _ in range(MAX_REASK):
            with torch.no_grad():
                logits = policy.actor(o)[0].masked_fill(~m_t, float("-inf"))
                a = int(torch.argmax(logits).item())
            pe = _to_pe_action(a, battle)
            if _pe_valid(pe, battle):
                chosen = a
                break
            m_np[a] = False
            m_t[a] = False
            if int(m_np.sum()) == 0:
                break

        if chosen is None:
            orders = list(battle.valid_orders)
            order = orders[0] if orders else DefaultBattleOrder()
            print(f"  NNbot: [fallback] {type(order).__name__}")
            return order

        as_ = get_action_space()
        if as_.SWITCH_BASE <= chosen < as_.SWITCH_BASE + 6:
            label = f"switch (slot {chosen - as_.SWITCH_BASE})"
        elif chosen >= as_.TRANSFORM_BASE:
            label = f"transform #{chosen - as_.TRANSFORM_BASE}"
        else:
            label = as_.id_to_move[chosen]
        print(f"  NNbot: {label}")
        return SinglesEnv.action_to_order(np.int64(_to_pe_action(chosen, battle)), battle)


# --------------------------------------------------------------------- helpers
def wait_for_battle_finish(player: NNPlayer, done_before: int,
                           timeout: int = BATTLE_TIMEOUT) -> bool:
    """Block until one more battle finishes. Returns True on success."""
    t0 = time.time()
    last_status = 0.0
    while player.n_finished_battles == done_before:
        if time.time() - t0 > timeout:
            print("  ! timed out waiting for the battle to finish")
            return False
        if time.time() - last_status > 30:
            last_status = time.time()
            try:
                active = [b for b in player.battles.values() if not b.finished]
                if active:
                    b = active[-1]
                    try:
                        t = b.turn
                    except (AttributeError, TypeError):
                        t = 0
                    state = f"turn {t}" if t > 0 else "battle started, waiting..."
                else:
                    state = "waiting for challenge to be accepted"
            except Exception:
                state = "..."
            print(f"  ... {state} ({int(time.time() - t0)}s)")
        time.sleep(1)
    return True


def cleanup_stale_battles(player: NNPlayer) -> None:
    """Forfeit any battles the server re-announces after login.

    Two-pronged: (1) choose_move returns ForfeitBattleOrder while
    _forfeit_mode is on, (2) directly send /forfeit to any room that
    is still active after the grace period.
    """
    print("  checking for stale battles...")
    time.sleep(6)  # give the server time to re-announce old battles

    stale = [(rid, b) for rid, b in player.battles.items() if not b.finished]
    if stale:
        print(f"  found {len(stale)} stale battle(s):")
        for rid, b in stale:
            print(f"    - {rid}")
            try:
                asyncio.run_coroutine_threadsafe(
                    player.ps_client.send_message("/forfeit", rid),
                    POKE_LOOP,
                ).result(timeout=5)
                print(f"    -> forfeited")
            except Exception as e:
                print(f"    -> failed: {type(e).__name__}: {e}")
        time.sleep(3)

    player._forfeit_mode = False
    remaining = [b for b in player.battles.values() if not b.finished]
    if remaining:
        print(f"  WARNING: {len(remaining)} battle(s) still active -- you may need "
              f"to forfeit from your browser")
    else:
        print("  clean -- ready to play")
    print()


# --------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Play human vs NN bot on play.pokemonshowdown.com")
    ap.add_argument("--you", required=True,
                    help="your Showdown username (the human player)")
    ap.add_argument("--bot-user", default="ttvJustinCaseBOT",
                    help="bot account username")
    ap.add_argument("--bot-pass", default="",
                    help="bot account password")
    ap.add_argument("--model", default=os.path.join(HERE, "checkpoints", "ppo.pt"))
    ap.add_argument("--hidden", default="1024,2048,4096,4096,2048,1024")
    ap.add_argument("--device", default=None, help="cuda|cpu")
    ap.add_argument("--seed", type=int, default=None, help="team draw seed")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        sys.exit(f"no model at {args.model} -- run training first or pass --model")

    hidden = tuple(int(h) for h in args.hidden.split(","))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading policy from {args.model}  (hidden={hidden}, device={device})")
    global policy
    policy = MaskedPPO(OBS_DIM, get_action_space().N_ACTIONS, hidden=hidden, device=device)
    policy.load(args.model)
    policy.actor.eval()

    ws_url = "wss://sim3.psim.us/showdown/websocket"
    site = "https://play.pokemonshowdown.com"

    print(f"\nSchedule: {len(MATCHES)} matches")
    for i, (f, s) in enumerate(MATCHES, 1):
        tag = " [strong team]" if s else " [random battle]"
        print(f"  {i}. {f}{tag}")
    print(f"\nServer: {site}")
    print(f"Bot:    '{args.bot_user}'")
    print(f"You:    '{args.you}'")
    print(f"Make sure you are logged in at {site}\n")

    # ---- create ONE player for all matches ----
    first_fmt = MATCHES[0][0]
    first_team = (StrongTeamBuilder(first_fmt, seed=args.seed)
                  if MATCHES[0][1] else None)
    player = NNPlayer(first_fmt, first_team, args.bot_user,
                      policy, ws_url=ws_url, password=args.bot_pass)
    print(f"  bot logged in as '{args.bot_user}'")

    # ---- clean up any stale battles from previous sessions ----
    cleanup_stale_battles(player)

    # ---- play the schedule ----
    results = []
    for i, (fmt, strong) in enumerate(MATCHES, 1):
        tag = " [strong team]" if strong else " [random battle]"
        print(f"\n=== Match {i}/{len(MATCHES)}: {fmt}{tag} ===")

        # swap team for this match
        if strong:
            team = StrongTeamBuilder(fmt,
                                     seed=(args.seed + i) if args.seed is not None else None)
            player.update_team(team)
        else:
            player.update_team(None)

        done_before = player.n_finished_battles

        # send challenge with the correct format (bypasses self._format)
        try:
            asyncio.run_coroutine_threadsafe(
                player.ps_client.challenge(args.you, fmt, player.next_team),
                POKE_LOOP,
            ).result(timeout=CHALLENGE_TIMEOUT)
            print("  challenge sent -- accept it in your browser!")
        except Exception as e:
            print(f"  ! could not send challenge: {type(e).__name__}: {e}")
            results.append((fmt, False))
            continue

        # wait for the battle to finish
        finished = wait_for_battle_finish(player, done_before)
        if not finished:
            results.append((fmt, False))
            continue

        # report result
        battles = list(player.battles.values())
        b = battles[-1]
        result = "WON" if b.won else ("LOST" if b.lost else "DREW")
        print(f"  === Match {i} over: NNbot {result} ===")
        results.append((fmt, b.won))

    wins = sum(1 for _, w in results if w)
    print(f"\n{'='*40}")
    print(f"  All matches done.  NNbot: {wins}W - {len(results) - wins}L")
    for fmt, w in results:
        print(f"    {'W' if w else 'L'}  {fmt}")
    print(f"{'='*40}")


if __name__ == "__main__":
    main()
