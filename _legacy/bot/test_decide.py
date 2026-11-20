"""Unit test for TraineeEnv._decide: the 'ask until legal + slight penalty' mechanism.

Monkey-patches the legality check so we control exactly which action is legal, then
verifies:
  1. an illegal pick is re-asked (agent gets another chance) and counted as 1 penalty;
  2. an immediately-legal pick is used with 0 penalty;
  3. if re-asks are exhausted it falls back to a server-valid order (never stalls).
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import env as E  # noqa: E402
from env import TraineeEnv, UNIFIED_ACTION_SIZE  # noqa: E402

# --- controlled legality: only action 500 is "server-legal" ---
E._to_pe_action = lambda a, battle: a                      # identity mapping for the test
E._pe_valid = lambda pe, battle: pe in battle.valid_set    # legality = membership

te = TraineeEnv.__new__(TraineeEnv)  # bypass __init__ (we only exercise _decide)
battle = type("B", (), {"valid_set": {500}})()

# mask allows 499/500/501 but the server only accepts 500
mask = np.zeros(UNIFIED_ACTION_SIZE, dtype=bool)
mask[[499, 500, 501]] = True


class Actor:
    def __init__(self, seq):
        self.seq = list(seq)
        self.calls = 0

    def act(self, obs, m):
        a = self.seq[min(self.calls, len(self.seq) - 1)]
        self.calls += 1
        return a, -1.0, 0.5


# 1) illegal first, legal second -> used, 1 penalty
a, lp, v, ill = te._decide(None, mask, Actor([499, 500]), battle)
assert a == 500 and ill == 1, (a, ill)
print("1) illegal -> re-asked -> legal (500), penalty=1  PASS")

# 2) legal immediately -> 0 penalty
a, lp, v, ill = te._decide(None, mask, Actor([500]), battle)
assert a == 500 and ill == 0, (a, ill)
print("2) legal immediately, penalty=0                 PASS")

# 3) always-illegal actor -> exhausts re-asks, falls back to a legal order (no stall)
E.TraineeEnv._first_valid_from_orders = staticmethod(lambda battle: 500)
a, lp, v, ill = te._decide(None, mask, Actor([499]), battle)
assert a == 500 and ill >= 1, (a, ill)
print(f"3) exhausted -> safe fallback to legal (500), ill={ill}  PASS")

# 4) empty mask -> 'wait' (no decision), no crash
a, lp, v, ill = te._decide(None, np.zeros(UNIFIED_ACTION_SIZE, dtype=bool), Actor([500]), battle)
assert a is None and ill == 0
print("4) empty mask -> 'wait' (None)                   PASS")

print("\n=== ALL _decide TESTS PASSED ===")
