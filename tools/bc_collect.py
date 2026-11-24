"""(c) BC warm start — the data pipeline (Arm B, 2026-11-24 DECISION).

Collects the STRATEGY TEACHER's live decisions (its own legal action at each
decision point, with the EXACT observation the net sees), filters the
ping-pong / no-progress sequences (D6 ruling, 2026-11-23), and stores a
compact dataset for the BC trainer (``net/bc.py``).

Design (no code written twice):
  * battle loop      : tools.train_step4_vs_pool.play_vs_checkpoint (shared;
                       the teacher sits on p1 via the ``decides_from_battle``
                       contract — the SAME path the p2 yardsticks use)
  * observation      : the env's encoded obs (build_obs pipeline) — captured
                       by the ``decision_hook`` at the decision moment
  * teacher          : fitness.strategy_agent.strategy_reference (the yardstick)
  * progress signal  : net.env.build_ctx (the ONE belief pipeline) -> a
                       deterministic progress signature (pure function)
  * ping-pong filter : pure keep-mask over the signature sequence (D6 ruling:
                       "re-collect or filter its data so ping-pong sequences
                       (turns with no progress) are excluded")

OWN SIDE FULLY KNOWN (the Arm B spec) holds by construction: this is a LIVE
battle, so the teacher's own team/moves are ground truth in the env — there is
no replay move-set guessing problem here (that was the imitation path).

Storage (npz, no pickle): per-field stacked obs arrays in registry LAYOUT
order + action_index + battle_id + the keep-mask bookkeeping in meta.json.
The trainer (net/bc.py) loads it into the SAME batch shape the PPO core uses.

NOT RUN automatically: collection needs free servers (the DECISION: "run the
collection only when the servers are free"). CPU-light by default.

Run (when servers are free):
    python -m tools.bc_collect --battles 100 --opps random,strategy
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import numpy as np

from net.env import build_ctx
from net.registry import LAYOUT
from pool import PoolConfig, ResourcePool
from fitness.m1_pool import build_m1_pool
from fitness.strategy_agent import strategy_reference
from tools.train_step4_vs_pool import play_vs_checkpoint, play_with_watchdog

FMT = "gen9ou"
COLLECT_PORT_BASE = 49990     # smoke band (the learnability run uses the same
                              # band — run ONE big job at a time, not both)
PING_PONG_MIN_RUN = 3         # 3+ consecutive no-progress decisions = a loop
                              # (keep the FIRST of the run — usually a
                              # legitimate reactive switch — drop the rest)


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure logic (hermetic — unit-tested without any server)
# ═══════════════════════════════════════════════════════════════════════════════
def _mon_fields(m: Dict) -> tuple:
    """The deterministic, turn-independent state of one slot (the progress
    signal). EXCLUDED by design: turn, active (a switch with nothing else
    changing is exactly the no-progress we must detect), and the per-turn
    flags (hit_this_turn / last_hit / last_move_failed / used_protect)."""
    return (
        m.get("hp_frac"), m.get("maxhp"),
        tuple(m.get("statuses") or ()),
        tuple(sorted((m.get("boosts") or {}).items())),
        tuple(m.get("volatiles") or ()),
        m.get("stockpile"), m.get("tera"),
    )


def progress_signature(ctx: Dict) -> str:
    """A deterministic 'did the battle actually progress' fingerprint from
    the belief context: every slot's HP/maxHP/statuses/boosts/volatiles/
    stockpile/tera on BOTH sides + the field (weather/terrain/screens/
    hazards/tailwind) + the fainted counters.

    Pure over the ctx dict (JSON-safe primitives only) — unit-testable with
    fake contexts. A signature change == real progress (damage, status,
    boost, hazard/screen/weather, faint). A signature repeat == a no-progress
    decision (a switch, a pass, a protect — nothing changed).
    """
    own = ctx.get("own") or {}
    opp = ctx.get("opponent") or {}
    own_slots = tuple(sorted((_mon_fields(own[k]) for k in own), key=repr))
    opp_slots = tuple(sorted((_mon_fields(opp[k]) for k in opp), key=repr))
    field = ctx.get("field") or {}
    fainted = tuple(sorted((ctx.get("fainted") or {}).items()))
    sig = (own_slots, opp_slots,
           field.get("weather"), field.get("terrain"),
           field.get("screens"), tuple(sorted((field.get("hazards") or {})
                                             .items())),
           field.get("tailwind"),
           fainted)
    return repr(sig)


def ping_pong_keep_mask(signatures: Sequence[str],
                        min_run: int = PING_PONG_MIN_RUN) -> List[bool]:
    """D6 filter: True = keep the decision, False = drop it.

    A decision is NO-PROGRESS when its signature equals the previous
    decision's (the battle state did not change since the teacher last acted).
    Within a run of consecutive no-progress decisions (the strategy bot's
    ping-pong: switch, re-switch, re-switch...), the FIRST no-progress
    decision is kept — it is usually the legitimate reactive switch — and the
    run is dropped from the (min_run - 1)-th consecutive no-progress onward.
    A signature change (real progress) resets the run.

    With the default ``min_run = 3``: a lone reactive switch (1 no-progress
    decision) is kept; the 2nd and every further consecutive no-progress
    decision is dropped — i.e. a 2-switch ping-pong loop contributes exactly
    one (the first) decision to the BC data.

    Pure over the signature sequence — unit-testable without a battle.
    (The exact threshold semantics are pinned by tools/tests/test_bc_collect.py.)
    """
    n = len(signatures)
    keep = [True] * n
    if n < 2 or min_run < 2:
        return keep
    run = 0  # consecutive no-progress decisions ending at i
    for i in range(1, n):
        if signatures[i] == signatures[i - 1]:
            run += 1
            if run >= min_run - 1:      # the min_run-th repeat in a row
                keep[i] = False
        else:
            run = 0
    return keep


def action_index_of(acts: Sequence[Dict], act: Optional[Dict]) -> int:
    """The position of ``act`` in the legal list, by IDENTITY (the same
    identity the pointer head scores by — kind + move/species). -1 if absent
    (a teacher anomaly — recorded, never silently re-mapped)."""
    if act is None:
        return -1
    key = (act.get("kind"), act.get("move"), act.get("species"))
    for i, a in enumerate(acts):
        if (a.get("kind"), a.get("move"), a.get("species")) == key:
            return i
    return -1


# ═══════════════════════════════════════════════════════════════════════════════
#  The per-battle recorder (the decision_hook)
# ═══════════════════════════════════════════════════════════════════════════════
class BCTrajectoryRecorder:
    """Records the teacher's (obs, legal list, chosen action, progress
    signature) at every p1 decision. One instance PER battle (the signature
    sequence is battle-scoped)."""

    def __init__(self, fmt: str = FMT) -> None:
        self.fmt = fmt
        self.records: List[Dict[str, Any]] = []

    def __call__(self, obs_dict: Dict, acts: List[Dict], chosen: Dict,
                 battle: Any) -> None:
        idx = action_index_of(acts, chosen)
        try:
            ctx = build_ctx(battle, self.fmt, "p1")
            if ctx is None:
                raise ValueError("build_ctx returned None")
            sig = progress_signature(ctx)
        except Exception:  # noqa: BLE001 — signature failure must never kill a battle
            # UNIQUE per record on purpose: an UNKNOWN progress state must not
            # masquerade as a no-progress repeat (identical CTX_ERR strings
            # would fabricate a ping-pong run and make the D6 filter drop
            # decisions we have no evidence are loops).
            sig = f"CTX_UNKNOWN_{len(self.records)}"
        self.records.append({
            "obs": obs_dict,
            "action_index": idx,
            "action": chosen,
            "sig": sig,
            "n_legal": len(acts),
        })

    def finish(self) -> Dict[str, Any]:
        """Apply the D6 filter; return the kept records + bookkeeping."""
        sigs = [r["sig"] for r in self.records]
        keep = ping_pong_keep_mask(sigs)
        kept = [r for r, k in zip(self.records, keep) if k]
        return {
            "n_decisions": len(self.records),
            "n_kept": len(kept),
            "n_dropped": len(self.records) - len(kept),
            "kept": kept,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset I/O (npz — no pickle, field order = registry LAYOUT)
# ═══════════════════════════════════════════════════════════════════════════════
def save_bc_dataset(path_dir: str, per_battle: List[Dict[str, Any]],
                    meta: Dict[str, Any]) -> Dict[str, Any]:
    """Stack all kept decisions into ``path_dir``:

      * ``data.npz``  — each LAYOUT field as (N, *field.shape) + action_index
      * ``meta.json`` — per-battle bookkeeping + filter config + totals

    Returns a small summary dict (N, kept/dropped totals)."""
    kept = [r for b in per_battle for r in b["result"]["kept"]]
    n = len(kept)
    if n == 0:
        # np.stack would raise an opaque error here; say what actually happened.
        raise ValueError(
            "save_bc_dataset: no decisions survived the D6 filter "
            f"({len(per_battle)} battles) — check the battle logs before "
            "re-collecting")
    arrays: Dict[str, np.ndarray] = {}
    for fld in LAYOUT:
        stacked = np.stack([np.ascontiguousarray(r["obs"][fld.name])
                            for r in kept])
        arrays[fld.name] = stacked
    arrays["action_index"] = np.array(
        [r["action_index"] for r in kept], dtype=np.int64)
    arrays["battle_id"] = np.array(
        [b["battle_id"] for b in per_battle
         for r in b["result"]["kept"]], dtype=np.int32)
    os.makedirs(path_dir, exist_ok=True)
    np.savez_compressed(os.path.join(path_dir, "data.npz"), **arrays)
    meta_out = dict(meta)
    meta_out["n_decisions"] = sum(b["result"]["n_decisions"]
                                  for b in per_battle)
    meta_out["n_kept"] = n
    meta_out["n_dropped"] = meta_out["n_decisions"] - n
    with open(os.path.join(path_dir, "meta.json"), "w", encoding="utf8") as fh:
        json.dump(meta_out, fh, indent=2, default=str)
    return {"n_kept": n, "n_decisions": meta_out["n_decisions"],
            "n_dropped": meta_out["n_dropped"]}


# ═══════════════════════════════════════════════════════════════════════════════
#  The collection driver
# ═══════════════════════════════════════════════════════════════════════════════
def collect_bc(n_battles: int, opp_names: Sequence[str], fmt: str,
               vocabs, out_dir: str, seed: int,
               n_servers: int = 2, per_server: int = 2,
               port_base: int = COLLECT_PORT_BASE,
               max_steps: int = 300, battle_timeout: int = 240,
               log=None) -> Dict[str, Any]:
    """Play n_battles with the strategy teacher on p1 against the named pool
    opponents (cycled), record + filter + save. The ONE collection path —
    the Arm B data for net/bc.py."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # CPU-light (DECISION); never
                                               # at import time (library-safe)
    lg = log or (lambda m: print(m, flush=True))
    # build_m1_pool is the pool's real API (a Ladder with .get); pass vocabs
    # so the net references do not rebuild the DB. A typo'd opponent name
    # raises a loud KeyError up front instead of failing mid-collection.
    pool_refs = build_m1_pool(fmt=fmt, vocabs=vocabs)
    refs = [pool_refs.get(n) for n in opp_names]
    teacher_ref = strategy_reference(fmt=fmt)

    pool = ResourcePool(
        PoolConfig(n_servers=n_servers, per_server=per_server,
                   port_base=port_base, boot_timeout=120),
        log=lambda m: lg(f"  [pool] {m}"))
    lg(f"[bc_collect] starting pool ({n_servers} x {per_server}) ...")
    if not all(pool.start(wait_healthy=True).values()):
        pool.shutdown()
        raise SystemExit("[bc_collect] pool not healthy — aborting")

    per_battle: List[Dict[str, Any]] = []
    n_won = n_lost = n_crash = 0
    try:
        for i in range(n_battles):
            bseed = seed + i
            opp = refs[i % len(refs)]
            rec = BCTrajectoryRecorder(fmt=fmt)
            teacher = teacher_ref.make_agent(seed=bseed)
            opp_agent = opp.make_agent(seed=bseed + 9000)
            with pool.battle_slot() as server_cfg:
                r = play_with_watchdog(
                    play_vs_checkpoint, teacher, opp_agent, fmt, bseed,
                    max_steps=max_steps, server_configuration=server_cfg,
                    vocabs=vocabs, decision_hook=rec,
                    timeout_s=battle_timeout)
            if r.get("timed_out") or r.get("crashed"):
                n_crash += 1
                lg(f"  [b{i+1}] {'TIMEOUT' if r.get('timed_out') else 'CRASH'}: "
                   f"{r.get('error', '')[:70]}")
                continue
            res = rec.finish()
            if r.get("a1_won"):
                n_won += 1
            elif r.get("a2_won"):
                n_lost += 1
            per_battle.append({
                "battle_id": len(per_battle),
                "seed": bseed, "opponent": opp.name,
                "teacher_won": bool(r.get("a1_won")),
                "steps": r.get("steps", 0),
                **{k: res[k] for k in ("n_decisions", "n_kept", "n_dropped")},
                "result": res,
            })
            lg(f"  [b{i+1}] vs {opp.name}: teacher "
               f"{'W' if r.get('a1_won') else 'L'} "
               f"decisions={res['n_decisions']} kept={res['n_kept']} "
               f"dropped={res['n_dropped']}")
    finally:
        pool.shutdown()

    meta = {
        "fmt": fmt,
        "teacher": "strategy (rules 1-4, the Step-6 yardstick)",
        "opponents": list(opp_names),
        "seed": seed,
        "battles_requested": n_battles,
        "battles_completed": len(per_battle),
        "teacher_wins": n_won, "teacher_losses": n_lost,
        "crashes": n_crash,
        "filter": {"rule": "D6 ping-pong", "min_run": PING_PONG_MIN_RUN,
                   "kept": "first decision of each no-progress run, dropped: rest"},
        "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "battles": [{k: b[k] for k in ("battle_id", "seed", "opponent",
                                       "teacher_won", "steps", "n_decisions",
                                       "n_kept", "n_dropped")}
                    for b in per_battle],
    }
    summary = save_bc_dataset(out_dir, per_battle, meta)
    lg(f"[bc_collect] DONE: {summary['n_kept']} kept decisions "
       f"({summary['n_dropped']} dropped by the D6 filter) -> {out_dir}")
    return {"summary": summary, "meta": meta}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--battles", type=int, default=100)
    p.add_argument("--opps", type=str, default="random,strategy",
                   help="comma list from the m1 pool (random,heuristic,"
                        "strategy,strategy_posterior)")
    p.add_argument("--fmt", type=str, default=FMT)
    p.add_argument("--out", type=str, default=None,
                   help="default: runs/bc_data_<ts>")
    p.add_argument("--seed", type=int, default=20261127)
    p.add_argument("--n-servers", type=int, default=2)
    p.add_argument("--per-server", type=int, default=2)
    p.add_argument("--port-base", type=int, default=COLLECT_PORT_BASE)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--battle-timeout", type=int, default=240)
    args = p.parse_args()

    from net.vocab_db import build_vocabs
    vocabs = build_vocabs()
    out = args.out or os.path.join(
        "runs", f"bc_data_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    collect_bc(n_battles=args.battles, opp_names=args.opps.split(","),
               fmt=args.fmt, vocabs=vocabs, out_dir=out, seed=args.seed,
               n_servers=args.n_servers, per_server=args.per_server,
               port_base=args.port_base, max_steps=args.max_steps,
               battle_timeout=args.battle_timeout)


if __name__ == "__main__":
    main()
