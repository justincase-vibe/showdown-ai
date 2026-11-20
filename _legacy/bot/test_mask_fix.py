"""Unit test for the get_action_mask switch fix: with an EMPTY available_switches
(whole bench fainted), NO switch actions may be un-masked. Previously this state
un-masked all 6 slots -> the policy picked a switch the server rejected -> re-pick.
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env import ShowdownEnv  # noqa: E402
from action_space import get_action_space  # noqa: E402

as_ = get_action_space()


class Mon:
    def __init__(self, bs, fainted=False):
        self.base_species = bs
        self.fainted = fainted


class Mv:
    def __init__(self, mid):
        self.id = mid


class Active:
    def __init__(self):
        self.moves = {"thunderbolt": Mv("thunderbolt"), "hurricane": Mv("hurricane")}


class Battle:
    def __init__(self, avail_moves, avail_switches, active=True, trapped=False, force_switch=False):
        self.trapped = trapped
        self.force_switch = force_switch
        self._wait = False
        self.can_mega_evolve = False
        self.can_dynamax = False
        self.can_tera = False
        self.available_moves = [Mv(m) for m in avail_moves]
        self.available_switches = avail_switches
        if active:
            self.active_pokemon = Active()
        else:
            self.active_pokemon = None
        # 6 distinct mons; the active one is slot 0, bench is 1..5
        self.team = {i: Mon(f"mon{i}") for i in range(6)}


def switch_band(mask):
    return [int(i) for i in np.flatnonzero(mask) if as_.SWITCH_BASE <= i < as_.SWITCH_BASE + 6]


def move_band(mask):
    return [int(i) for i in np.flatnonzero(mask) if i < as_.SWITCH_BASE]


def main():
    failures = 0

    # CASE 1 (the bug): active mon with 1 move, EMPTY bench (no available_switches).
    b = Battle(avail_moves=["thunderbolt"], avail_switches=[], active=True)
    m = ShowdownEnv.get_action_mask(b)
    sw, mv = switch_band(m), move_band(m)
    print(f"CASE1 empty-bench: switches={sw} moves={mv}")
    if sw:
        print("  FAIL: switches un-masked with an empty bench (the bug)")
        failures += 1
    if "thunderbolt" not in [as_.id_to_move[i] for i in mv]:
        print("  FAIL: available move not un-masked")
        failures += 1

    # CASE 2: active mon with 2 moves, 2 available bench mons -> exactly those 2 switches.
    bench = [Mon("mon2"), Mon("mon3")]
    b = Battle(avail_moves=["thunderbolt", "hurricane"], avail_switches=bench, active=True)
    m = ShowdownEnv.get_action_mask(b)
    sw, mv = switch_band(m), move_band(m)
    # switches map to slots 2 and 3
    expected_sw = [as_.switch_id(2), as_.switch_id(3)]
    print(f"CASE2 2-bench: switches={sorted(sw)} expected={sorted(expected_sw)} moves={len(mv)}")
    if sorted(sw) != sorted(expected_sw):
        print("  FAIL: switch set != the two available bench mons")
        failures += 1
    if len(mv) != 2:
        print("  FAIL: expected exactly 2 moves un-masked")
        failures += 1

    # CASE 3: force_switch -> NO moves, only the available switches.
    bench = [Mon("mon1"), Mon("mon4")]
    b = Battle(avail_moves=["thunderbolt"], avail_switches=bench, active=True, force_switch=True)
    m = ShowdownEnv.get_action_mask(b)
    sw, mv = switch_band(m), move_band(m)
    print(f"CASE3 force_switch: switches={sorted(sw)} moves={mv}")
    if mv:
        print("  FAIL: moves un-masked during a forced switch")
        failures += 1
    if sorted(sw) != sorted([as_.switch_id(1), as_.switch_id(4)]):
        print("  FAIL: forced-switch set != available bench")
        failures += 1

    print("\n=== " + ("ALL MASK TESTS PASSED" if failures == 0 else f"{failures} FAILURES") + " ===")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
