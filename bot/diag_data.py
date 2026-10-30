"""Check the imitation training data's action-label distribution (move vs switch vs transform).

If the experts' data is move-heavy (expected) but the actor outputs ~100% switches in live play,
the bug is a train/live mismatch (observation or action-mapping). If the data is ALSO switch-heavy,
the labels themselves are mis-mapped. Either way this localizes the root cause.
"""
from __future__ import annotations
import sys
import numpy as np

sys.path.insert(0, "bot")
from action_space import get_action_space  # noqa: E402

AS = get_action_space()
d = np.load("bot/data/imitation_dataset.npz")
print("KEYS:")
for k in d.files:
    a = d[k]
    print(f"  {k}: shape={a.shape} dtype={a.dtype}")

act_key = next((k for k in d.files if k.lower() in ("actions", "action", "acts", "labels", "y", "target")), None)
print(f"\naction key: {act_key}")
if act_key is None:
    sys.exit("no action array found -- inspect keys above")
acts = d[act_key].astype(np.int64).ravel()
n = len(acts)
nm = int(np.sum(acts < AS.SWITCH_BASE))
ns = int(np.sum((acts >= AS.SWITCH_BASE) & (acts < AS.TRANSFORM_BASE)))
nt = int(np.sum(acts >= AS.TRANSFORM_BASE))
print(f"n={n}")
print(f"  move     : {nm}  ({nm / n:.1%})")
print(f"  switch   : {ns}  ({ns / n:.1%})")
print(f"  transform: {nt}  ({nt / n:.1%})")
print(f"  (SWITCH_BASE={AS.SWITCH_BASE}, TRANSFORM_BASE={AS.TRANSFORM_BASE}, N_ACTIONS={AS.N_ACTIONS})")
print(f"  min={acts.min()}  max={acts.max()}")
