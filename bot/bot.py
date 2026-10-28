"""Starter Pokemon Showdown bot built on poke-env.

Run against the local Showdown server (see README):

    python bot/bot.py [battle_format]

Defaults to the `gen9randombattle` format and plays one full battle
against a second random agent (self-play) so you can verify the
whole pipeline works.
"""

import asyncio
import sys

from poke_env.battle import AbstractBattle
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.player import Player

class RandomAgent(Player):
    """A poke-env player that picks a random legal move every turn."""

    def choose_move(self, battle: AbstractBattle):
        # This is the single method you implement. Replace the random
        # choice with your own logic, e.g. based on:
        #   battle.me, battle.opponent, battle.valid_orders,
        #   battle.log (recent events), battle.field, battle.weather
        return Player.choose_random_move(battle)


def build_agent(username: str, fmt: str) -> Player:
    # Random-battle formats (*random*) reject custom teams, so the server
    # generates them. For other formats, pass a showdown team string instead,
    # e.g. team="Pikachu\n\nCharmander\n\n..."
    return RandomAgent(
        account_configuration=AccountConfiguration(username, None),
        battle_format=fmt,
        team=None,
        max_concurrent_battles=1,
        save_replays=True,
        server_configuration=LocalhostServerConfiguration,
    )


def main() -> None:
    fmt = sys.argv[1] if len(sys.argv) > 1 else "gen9randombattle"

    agent_a = build_agent("BotA", fmt)
    agent_b = build_agent("BotB", fmt)

    print(f"Playing one {fmt} battle: BotA vs BotB (local server :8000)")
    asyncio.run(agent_a.battle_against(agent_b, n_battles=1))
    print("Battle finished.")


if __name__ == "__main__":
    main()
