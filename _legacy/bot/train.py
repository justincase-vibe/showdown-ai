"""Self-play training harness for the Pokemon Showdown RL bot.

Runs N concurrent battles across a *mix of formats* (e.g. gen7ou + gen7randombattle),
collects transitions into a PPO buffer, updates the shared policy on the chosen device,
and tracks rolling win-rate per format. The opponent is pluggable:

  * ``random`` -- a random-legal-move baseline (sanity check / curriculum floor). Only
                  the trainee (P1) learns; P2 is a noise floor.
  * ``frozen`` -- a periodically-updated copy of the trainee (P1 learns; P2 is a lagging
                  target, not stored).
  * ``live``   -- the trainee itself plays BOTH sides (default). A loss on one side is a
                  win on the other; BOTH sides' transitions update the SAME shared policy
                  (zero-sum reward). This is the "both sides train the model" mode.

Everything here is device-agnostic. Run the CPU smoke test first (no GPU needed):

    python bot/train.py --smoke

Real GPU training (start only when the GPU is free):

    python bot/train.py --device cuda --num-envs 6 --buffer-size 4096 --max-battles 2000
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import random
import shutil
import sys
import time
import traceback
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Make sibling modules importable when run as a script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import ShowdownEnv, TraineeEnv, UNIFIED_ACTION_SIZE  # noqa: E402
from action_space import get_action_space  # noqa: E402
from obs import OBS_DIM  # noqa: E402
from policy import MaskedPPO  # noqa: E402
from teams import RandomTeamBuilder, StrongTeamBuilder, MixedTeamBuilder  # noqa: E402
from poke_env.ps_client import ServerConfiguration  # noqa: E402


# --------------------------------------------------------------------------- #
# Env pool construction
# --------------------------------------------------------------------------- #
def apportion(total: int, weights: List[float]) -> List[int]:
    """Distribute ``total`` items across buckets by (proportional) weight."""
    s = sum(weights) or 1.0
    raw = [total * w / s for w in weights]
    counts = [int(r) for r in raw]
    rem = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(rem):
        counts[order[i % len(order)]] += 1
    return counts


def _server_config(port: int, ip: str = "localhost") -> ServerConfiguration:
    """Build a poke_env ServerConfiguration pointing at a local server on ``ip:port``."""
    return ServerConfiguration(
        websocket_url=f"ws://{ip}:{port}/showdown/websocket",
        authentication_url=f"http://{ip}:{port}/action.php?",
    )


def _replay_says_forfeit(tag: str, replays_dir: Optional[str]) -> Optional[bool]:
    """Ground truth for a forfeit: the saved replay stream must contain an explicit
    `|forfeit|` line (or a 'forfeited' message). A legit fast sweep (one team wiped
    while the other stays intact) does NOT contain one -- the alive/turns heuristic
    alone cannot tell the two apart, so the warning is verified against the replay.

    Returns True/False when the replay file is found, None when it is not (yet).
    """
    def _check() -> Optional[bool]:
        try:
            if not tag or not replays_dir or not os.path.isdir(replays_dir):
                return None
            cands = glob.glob(os.path.join(replays_dir, f"*{tag}*.html"))
            if not cands:
                return None
            newest = max(cands, key=os.path.getmtime)
            with open(newest, encoding="utf-8", errors="replace") as fh:
                txt = fh.read().lower()
            return ("|forfeit|" in txt) or ("forfeited" in txt)
        except Exception:
            return None

    res = _check()
    if res is None:
        # Replay may be written a beat after done: give it one retry.
        time.sleep(1.0)
        res = _check()
    return res


def build_envs(
    formats: List[str],
    weights: List[float],
    num_envs: int,
    seed: int,
    save_replays: Optional[str],
    team_mode: str = "strong",
    p_strong: float = 0.5,
    server_ports: Optional[List[int]] = None,
    server_ip: str = "localhost",
) -> List[TraineeEnv]:
    counts = apportion(num_envs, weights)
    envs: List[TraineeEnv] = []
    idx = 0
    if server_ports is None:
        server_ports = [8000]
    for fmt, c in zip(formats, counts):
        for _ in range(c):
            if "randombattle" in fmt:
                team = None
            elif team_mode == "random":
                team = RandomTeamBuilder(battle_format=fmt, seed=seed + idx)
            elif team_mode == "mixed":
                team = MixedTeamBuilder(battle_format=fmt, seed=seed + idx, p_strong=p_strong)
            else:
                team = StrongTeamBuilder(battle_format=fmt, seed=seed + idx)
            port = server_ports[idx % len(server_ports)]
            env = ShowdownEnv(
                battle_format=fmt,
                team=team,
                server_configuration=_server_config(port, server_ip),
                strict=False,
                save_replays=save_replays,
                log_level=logging.WARNING,
            )
            envs.append(TraineeEnv(env, env_id=idx, fmt=fmt))
            idx += 1
    return envs


# --------------------------------------------------------------------------- #
# Main training loop
# --------------------------------------------------------------------------- #
def _store_opponent_side(e: "TraineeEnv", policy: "MaskedPPO", opponent, done: bool) -> None:
    """Both-sides self-play: store the opponent's (P2) transition when the opponent IS
    the shared training policy (``--opponent live``). P2's (obs, action, logprob, value)
    come from the state P2 decided from (captured in TraineeEnv.step); the reward is
    P2's zero-sum reward (``rew[a2]`` = -P1's at a terminal win/loss). Stored under a
    distinct env id so P1 and P2 are separate trajectories for per-episode GAE. No-op
    when the opponent is not the shared policy (random / frozen) or P2 had nothing to
    decide (a Showdown 'wait' request).
    """
    if opponent is not policy:
        return
    lo = e.last_opp
    if not lo or not lo.get("made"):
        return
    o2, a2, m2 = lo.get("obs"), lo.get("a"), lo.get("mask")
    if o2 is None or a2 is None or m2 is None or int(np.sum(m2)) == 0:
        return
    policy.store(
        e.id + 1_000_000, o2, a2, lo.get("lp", 0.0), e.last_opp_rew, done, m2, lo.get("v", 0.0)
    )


def _best_effort_save(policy, ckpt_dir) -> None:
    """Save the checkpoint, but NEVER let a save failure mask the real error.
    Used from the exception handlers so that even a crash leaves the newest possible
    checkpoint on disk (best-effort)."""
    log = logging.getLogger("train")
    try:
        ck = os.path.join(ckpt_dir, "ppo.pt")
        policy.save(ck)
        log.info("checkpoint saved (best-effort) -> %s", ck)
    except Exception as exc:
        log.error("!!! best-effort save FAILED: %s: %s (relying on the last successful save)",
                  type(exc).__name__, exc)


def run(args) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("train")

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    weights = [float(w) for w in args.weights.split(",")]
    if len(weights) != len(formats):
        weights = [1.0 / len(formats)] * len(formats)

    log.info(
        "formats=%s weights=%s num_envs=%d opponent=%s buffer=%d max_battles=%d device=%s",
        formats, weights, args.num_envs, args.opponent, args.buffer_size, args.max_battles, args.device,
    )

    # Parse server targets (e.g. "8000,8001" or "192.168.1.5:8000,192.168.1.5:8001")
    if getattr(args, "server_ports", None):
        server_ip = "localhost"
        server_ports = []
        for p in args.server_ports.split(","):
            p = p.strip()
            if ":" in p:
                h, port = p.split(":")
                server_ip = h.strip()
                server_ports.append(int(port))
            else:
                server_ports.append(int(p))
    else:
        server_ip = "localhost"
        server_ports = [8000]
    log.info("server_ip=%s server_ports=%s (%d server(s))", server_ip, server_ports, len(server_ports))

    envs = build_envs(formats, weights, args.num_envs, args.seed, args.replays,
                      team_mode=args.team_mode, p_strong=args.p_strong,
                      server_ports=server_ports, server_ip=server_ip)
    fmt_of = {e.id: e.fmt for e in envs}

    policy = MaskedPPO(
        obs_dim=OBS_DIM,
        n_actions=UNIFIED_ACTION_SIZE,
        hidden=tuple(int(h) for h in args.hidden.split(",")),
        lr=args.lr,
        epochs=args.ppo_epochs,
        minibatch=args.minibatch,
        device=args.device if args.device != "auto" else None,
        seed=args.seed,
    )
    log.info("policy device=%s | params=%d", policy.device,
             sum(p.numel() for p in policy.actor.parameters()))

    opponent: object = "random"
    if args.opponent == "frozen":
        opponent = MaskedPPO(
            obs_dim=OBS_DIM, n_actions=UNIFIED_ACTION_SIZE,
            hidden=tuple(int(h) for h in args.hidden.split(",")),
            device=policy.device, seed=args.seed + 1,
        )
        # start matched: both begin from the same weights (random-init, or the
        # imitation warm-start loaded just below). A trainee vs a *random* opponent
        # would get an easy early win-rate spike that misleads the value baseline.
        policy.sync_opponent(opponent)
    elif args.opponent == "live":
        opponent = policy

    # ---- optional warm-start: load a pre-trained actor (e.g. the imitation BC) ----
    if getattr(args, "init_actor", None) and os.path.isfile(args.init_actor):
        try:
            sd = torch.load(args.init_actor, map_location=policy.device, weights_only=True)
            policy.actor.load_state_dict(sd)
            log.info("warm-started policy actor from %s", args.init_actor)
            if args.opponent == "frozen" and opponent is not policy:
                opponent.actor.load_state_dict(sd)  # keep the frozen target matched
        except Exception as e:
            log.warning("--init-actor load failed (%s); continuing with random init", e)
    elif getattr(args, "init_actor", None):
        log.warning("--init-actor path not found: %s", args.init_actor)

    # ---- optional RESUME: continue from a FULL checkpoint (actor+critic+opt+n_updates) ----
    # Unlike --init-actor (a bare actor state_dict), this restores the whole net so the
    # run truly continues from where a previous one left off.
    if getattr(args, "resume", None):
        if os.path.isfile(args.resume):
            policy.load(args.resume)
            log.info("RESUMED policy from full checkpoint %s (n_updates=%d)", args.resume, policy.n_updates)
            if args.opponent == "frozen" and opponent is not policy:
                opponent.load(args.resume)  # keep the frozen target matched
        else:
            log.warning("--resume path not found: %s (starting fresh)", args.resume)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    stats_path = os.path.join(args.ckpt_dir, "stats.jsonl")

    wins: Dict[str, int] = {}
    losses: Dict[str, int] = {}
    battles_done = 0
    transitions = 0
    start = time.time()

    # --- quality tracking (switch ratio, forfeits) ---
    as_ = get_action_space()
    battle_moves = 0
    battle_switches = 0
    battle_switch_slots = [0] * 6   # which bench slot each switch targeted (degeneracy check)
    battle_forfeits = 0
    battle_turns = 0
    illegal_window = 0              # illegal picks since the last "battles" log (window, not cumulative)
    recent_battles: List[dict] = []  # last N battles for quality stats

    # --- degenerate-switching (PER-BATTLE): a switch-spam loop (across one or more slots)
    # shows up as a high switches-per-turn rate. The threshold is CLI-configurable
    # (--degen-sw-per-turn, default 0.33) and also editable in the launcher GUI. ---
    DEGEN_MIN_SW = 4           # need at least this many switches to call it spam (anti-noise)
    warned_degen: set = set()  # battle numbers already flagged (avoid duplicate warns)

    # --- regression / instability tracking (updated on each policy.update) ---
    best_calib = -1.0          # best single-update calib_sign_hit seen so far
    regress_streak = 0         # consecutive updates below (best - margin)
    clip_streak = 0            # consecutive updates with high clip_frac
    CALIB_REGRESS_MARGIN = 0.10  # sign_hit must drop >this below best-so-far
    REGRESS_STREAK_N = 3       # ...for this many consecutive updates to warn
    CLIP_FRAC_HIGH = 0.50      # clip_frac above this = policy shifting too hard
    CLIP_STREAK_N = 4          # ...for this many consecutive updates to warn
    sign_hit_window: deque = deque(maxlen=3)  # rolling window for the best.pt score
    best_score = -1.0          # best rolling calib_sign_hit (drives snapshots/best.pt)

    cur_obs: Dict[int, np.ndarray] = {}
    cur_mask: Dict[int, np.ndarray] = {}
    # NOTE: the initial reset is done by the guarded startup loop below (safe_reset).
    # A bare ``e.reset()`` here would create a live battle that the guard then sees and
    # either forfeits (old behavior) or aborts (current) -- the root of the turn-1
    # 'concede'. So we must NOT reset here; the guarded loop is the single initial reset.

    def buffer_size() -> int:
        # Count ONLY what update() can actually drain: the *merged* buffer built
        # from FINISHED episodes (policy._buffer). In-progress live episodes
        # (policy._episodes) are NOT trainable yet -- update() only consumes the
        # merged buffer. Counting live episodes made the update gate fire
        # prematurely, which (a) broke out of the collect loop after the FIRST env
        # (starving the rest) and (b) spun update() on an empty merged buffer
        # (empty upstats) while battles never got to finish.
        if policy._buffer is not None:
            return len(policy._buffer["act"])
        return 0

    def log_stats(tag: str, extra: dict = None):
        # Quality metrics from recent battles
        n_recent = min(20, len(recent_battles))
        if n_recent > 0:
            recent = recent_battles[-n_recent:]
            avg_sw = round(np.mean([b["sw%"] for b in recent]))
            avg_turns = round(np.mean([b["turns"] for b in recent]), 1)
            n_forfeit = sum(1 for b in recent if b["forfeit"])
            avg_sw_conc = round(float(np.mean([b.get("sw_conc", 0.0) for b in recent])), 3)
            avg_sw_per_turn = round(float(np.mean([b.get("sw_per_turn", 0.0) for b in recent])), 3)
            total_sw_recent = sum(b.get("sw", 0) for b in recent)
        else:
            avg_sw = 0
            avg_turns = 0
            n_forfeit = 0
            avg_sw_conc = 0
            avg_sw_per_turn = 0
            total_sw_recent = 0
        rec = {
            "t": int(time.time() - start),
            "battles": battles_done,
            "wins": dict(wins),
            "losses": dict(losses),
            "winrate": {
                k: (wins.get(k, 0) / (wins.get(k, 0) + losses.get(k, 0))
                    if (wins.get(k, 0) + losses.get(k, 0)) else None)
                for k in (set(wins) | set(losses))
            },
            "quality": {"avg_sw%": avg_sw, "avg_turns": avg_turns, "forfeits": n_forfeit,
                        "sw_conc": avg_sw_conc, "avg_sw_per_turn": avg_sw_per_turn,
                        "illegal_picks": sum(getattr(e, "illegal_total", 0) for e in envs),
                        "illegal_window": illegal_window},
            "tag": tag,
        }
        if extra:
            rec.update(extra)
        line = json.dumps(rec)
        with open(stats_path, "a") as fh:
            fh.write(line + "\n")
        log.info("%s", line)
        # NOTE: degenerate switching is now flagged PER-BATTLE at record time (see the
        # sw_per_turn / sw_conc check in the step loop) so a single bad game is not diluted
        # by the 20-battle window -- that dilution is why an obvious degenerate replay used
        # to slip through unflagged. Only the illegal-pick spike stays in this window check.
        if illegal_window >= 5:
            log.warning(
                "!!! ILLEGAL-PICK SPIKE: %d server-rejected picks in this window -- the action "
                "mask likely disagrees with the server. Check env.get_action_mask.",
                illegal_window,
            )

    def safe_reset(e, max_retries: int = 3, grace_s: float = 5.0, cooldown: float = 11.0):
        """Reset an env (start the next battle) with guard + cooldown + narrow retry.

        GUARD (the forfeit fix): poke-env's reset()/close() auto-forfeit any battle that
        is not finished yet (poke_env/environment/env.py) -- the sole source of the
        spurious turn-1 'forfeit' replays. Root cause: step() computes ``done`` from the
        battle it pulled via battle_queue.race_get(), but never writes that battle back
        to env.battle1, so env.battle1 (what reset() inspects) can be a DIFFERENT/stale
        battle. Before every reset we make sure the current battle has actually finished
        (a timing race is waited out via grace_s); if it is still LIVE we ABORT the
        re-arm (return failure) instead of calling reset() -- we never forfeit. The
        caller drops this env slot.
        COOLDOWN: Showdown cancels a challenge issued <10s after the previous one; a
        cancelled challenge leaves a limbo battle that the next reset then forfeits. So
        we space re-challenges per env (>= cooldown since the last).
        NARROW RETRY: only the expected failure (team-rejection / challenge-timeout,
        surfaced as RuntimeError by TraineeEnv.reset) is retried -- a bare ``except
        Exception`` was re-calling reset() on ANY error (forfeits a live battle AND
        hammers the cooldown).

        Returns (obs, success); success is False when the re-arm is aborted (live battle)
        or a reset keeps failing.
        """
        def finished() -> bool:
            try:
                for b in (e.env.battle1, e.env.battle2):
                    if b is not None and not bool(getattr(b, "finished", False)):
                        return False
                return True
            except Exception:
                return True  # unknown state: let reset()'s own logic + trace handle it

        def sig(b) -> str:
            try:
                alive = sum(1 for p in (b.team or {}).values() if not p.fainted)
                return (f"turn={b.turn} alive={alive} finished={bool(b.finished)} "
                        f"tag={getattr(b, 'battle_tag', '?')} id={id(b)}")
            except Exception:
                return "n/a"

        log.info(
            "safe_reset ENTER [env %d]: b1=%s | b2=%s | e.done=%s | "
            "a1_to_move=%s a2_to_move=%s",
            e.id,
            sig(e.env.battle1), sig(e.env.battle2),
            getattr(e, "done", "?"),
            getattr(e.env, "agent1_to_move", "?"),
            getattr(e.env, "agent2_to_move", "?"),
        )

        for attempt in range(max_retries):
            # --- guard: a LIVE battle at re-arm time would be FORFEITED by reset() ---
            # (the turn-1 'concede' bug). Root cause: poke-env's step() computes ``done``
            # from the battle it pulled via battle_queue.race_get(), but it never writes
            # that battle back to env.battle1 -- so env.battle1 (what reset() inspects)
            # can be a DIFFERENT/stale battle. When that stale battle is not finished,
            # reset() forfeits the live one. Fix: if the battle is still live after the
            # grace wait, ABORT the re-arm (return failure) instead of calling reset().
            # The caller drops this env slot; we deliberately do NOT forfeit.
            if not finished():
                deadline = time.time() + grace_s
                while not finished() and time.time() < deadline:
                    time.sleep(0.25)
                if not finished():
                    log.error(
                        "LIVE-BATTLE AT RE-ARM [env %d, attempt %d/%d]: e.done=%s | "
                        "b1=%s | b2=%s | a1_to_move=%s a2_to_move=%s -- ABORTING "
                        "re-arm (NOT forfeiting); dropping this env slot.",
                        e.id, attempt + 1, max_retries,
                        getattr(e, "done", "?"),
                        sig(e.env.battle1), sig(e.env.battle2),
                        getattr(e.env, "agent1_to_move", "?"),
                        getattr(e.env, "agent2_to_move", "?"),
                    )
                    return None, False
            # --- FIX #4 cooldown: never re-challenge inside the server's 10s window ---
            wait = cooldown - (time.time() - getattr(e, "_last_challenge", 0.0))
            if wait > 0:
                time.sleep(wait)
            e._last_challenge = time.time()
            # --- reset (FIX #2: retry only the expected failure) ---
            try:
                obs = e.reset()
                return obs, True
            except (RuntimeError, TimeoutError) as exc:
                log.warning("env %d reset failed (attempt %d/%d): %s",
                            e.id, attempt + 1, max_retries, str(exc)[:160])
                if attempt == max_retries - 1:
                    log.error("env %d: giving up after %d resets -- skipping this slot", e.id, max_retries)
                    return None, False
        return None, False

    def _drain(max_steps_per_env: int = 2500, per_env_timeout_s: float = 1200.0) -> int:
        """Let every battle still in progress run to a NATURAL win/loss before teardown.

        This is the root-cause fix for the spurious 'X forfeited' replays: poke-env's
        close() forfeits any battle not finished yet, so instead we step each live env
        to completion and store the transitions. A long battle therefore keeps all of
        its info and is trained on, rather than being cut off at the run boundary.

        Pokemon battles always terminate (Showdown enforces a turn limit -- there is no
        open-ended stall), so each one simply runs to `done`. The caps are only hard
        hang-safety: a per-env step cap set above the turn limit, and a generous
        per-env wall-clock timeout (NOT shared across envs, so each battle gets its own
        allowance). A normal battle finishes long before either; if one is ever hit,
        ShowdownEnv.close() abandons just that battle cleanly (no forfeit, no hang) and
        the rest still finish.
        """
        active = [e for e in envs if e.id not in skipped_envs and not e.done]
        if not active:
            return 0
        log.info("letting %d in-progress battle(s) run to a natural result before teardown...", len(active))
        run_start = time.time()
        finished = 0
        for e in active:
            steps = 0
            env_start = time.time()
            try:
                while (not e.done and steps < max_steps_per_env
                       and (time.time() - env_start) < per_env_timeout_s):
                    obs_before = e.last_obs
                    new_obs, r, done = e.step(policy, opponent)
                    ls = e.last_self
                    if ls.get("made"):
                        policy.store(e.id, ls["obs"], int(ls["a"]), ls["lp"], r, done, ls["mask"], ls.get("v", 0.0))
                    else:
                        policy.store_pass(e.id, obs_before, r, policy.value(obs_before), done)
                    _store_opponent_side(e, policy, opponent, done)
                    steps += 1
                    if done:
                        break
            except Exception as exc:
                log.warning("drain: env %d stopped early (%s); close() will abandon it cleanly",
                            e.id, str(exc)[:140])
            finished += int(e.done)
            log.info("drain: env %d %s after %d step(s) (%.1fs)",
                     e.id, "finished" if e.done else "still live (abandoning)", steps, time.time() - env_start)
        log.info("drain complete in %.1fs (%d battle(s) finished)", time.time() - run_start, finished)
        return finished

    # --- initial reset with safety net ---
    skipped_envs: set = set()
    for e in envs:
        obs, ok = safe_reset(e)
        if ok:
            cur_obs[e.id] = obs
            cur_mask[e.id] = e.last_mask
        else:
            skipped_envs.add(e.id)
            log.warning("env %d skipped at startup", e.id)

    try:
        while battles_done < args.max_battles:
            # --- collect a batch of transitions across envs ---
            for e in envs:
                if e.id in skipped_envs:
                    continue
                if e.done:
                    # re-arm finished envs as they come up
                    obs, ok = safe_reset(e)
                    if not ok:
                        skipped_envs.add(e.id)
                        continue
                    cur_obs[e.id] = obs
                    cur_mask[e.id] = e.last_mask
                obs = cur_obs[e.id]
                mask = cur_mask[e.id]
                try:
                    # Both sides make their OWN server-validated decision inside step():
                    # P1 = the trainee policy, P2 = the opponent. If a pick is illegal the
                    # env re-asks the agent (and applies a small penalty) until it is legal,
                    # so the (action, logprob) pairs exposed on e.last_self / e.last_opp are
                    # always consistent for PPO.
                    new_obs, r, done = e.step(policy, opponent)
                    ls = e.last_self
                    if ls.get("made"):
                        a = int(ls["a"])
                        # Track action type for quality stats
                        if as_.SWITCH_BASE <= a < as_.SWITCH_BASE + 6:
                            battle_switches += 1
                            battle_switch_slots[a - as_.SWITCH_BASE] += 1
                        else:
                            battle_moves += 1
                        illegal_window += int(ls.get("ill") or 0)
                        policy.store(e.id, ls["obs"], a, ls["lp"], r, done, ls["mask"], ls.get("v", 0.0))
                    else:
                        # Showdown 'wait' request: P1 made no decision; record the step pass.
                        policy.store_pass(e.id, obs, r, policy.value(obs), done)
                except Exception as exc:
                    log.error("env %d: step crashed: %s -- skipping this env slot", e.id, str(exc)[:200])
                    skipped_envs.add(e.id)
                    continue
                # Both-sides self-play: also store the opponent's transition when the
                # opponent is the shared policy (--opponent live); no-op otherwise.
                _store_opponent_side(e, policy, opponent, done)
                transitions += 1
                battle_turns += 1
                if done:
                    battles_done += 1
                    key = fmt_of[e.id]
                    if r > 0:
                        wins[key] = wins.get(key, 0) + 1
                    elif r < 0:
                        losses[key] = losses.get(key, 0) + 1
                    else:
                        pass  # draw / double-kO
                    
                    # --- quality check: detect forfeit (one side >=5 mons alive, other 0, <15 turns) ---
                    # NOTE: use the battle object's own turn count -- `battle_turns` is a
                    # GLOBAL counter shared by all envs (reset on ANY battle end), so it is
                    # near-meaningless per-battle under parallel envs and made this check
                    # fire on normal sweep losses. Likewise the per-battle flag below must
                    # NOT depend on the cumulative `battle_forfeits` counter: that stickiness
                    # turned every later "winner kept 5 mons" battle into a false
                    # 'FORFEIT DETECTED -- bot conceded' warning.
                    b1 = e.env.battle1
                    b2 = e.env.battle2
                    b1_alive = sum(1 for p in (b1.team or {}).values() if not p.fainted)
                    b2_alive = sum(1 for p in (b2.team or {}).values() if not p.fainted)
                    battle_turns_obj = max(getattr(b1, "turn", 0) or 0, getattr(b2, "turn", 0) or 0)
                    battle_is_forfeit = (
                        battle_turns_obj < 15
                        and ((b1_alive >= 5 and b2_alive == 0) or (b2_alive >= 5 and b1_alive == 0))
                    )
                    if battle_is_forfeit:
                        # Verify against the replay (ground truth): a REAL forfeit has a
                        # `|forfeit|` line; a fast legit sweep (e.g. 14 turns, winner's team
                        # intact) does not. Suppress the false positive; keep the flag when
                        # the replay is unavailable so a genuine forfeit is never silenced.
                        if _replay_says_forfeit(
                            getattr(b1, "battle_tag", "") or "", args.replays) is False:
                            battle_is_forfeit = False
                    if battle_is_forfeit:
                        battle_forfeits += 1
                    
                    # Log per-battle quality
                    total_actions = battle_moves + battle_switches
                    sw_pct = (battle_switches / total_actions * 100) if total_actions > 0 else 0
                    sw_per_turn = (battle_switches / battle_turns) if battle_turns > 0 else 0.0
                    # Switch-slot concentration: "always switch to the same mon" (a mask/
                    # forced-switch bug) shows up as ~100% of switches hitting ONE slot. A
                    # healthy policy spreads across the bench, so a dominant slot is a flag.
                    sw_conc, sw_ent = 0.0, 0.0
                    tot_sw = sum(battle_switch_slots)
                    if tot_sw > 0:
                        ps = [x / tot_sw for x in battle_switch_slots if x > 0]
                        sw_conc = max(ps)
                        sw_ent = -sum(p * math.log2(p) for p in ps)
                    recent_battles.append({
                        "b": battles_done, "fmt": key, "turns": battle_turns,
                        "mv": battle_moves, "sw": battle_switches, "sw%": round(sw_pct),
                        "sw_per_turn": round(sw_per_turn, 3),
                        "sw_conc": round(sw_conc, 3), "sw_ent": round(sw_ent, 3),
                        "forfeit": battle_is_forfeit,
                        "result": "W" if r > 0 else ("L" if r < 0 else "D")
                    })
                    # Reset per-battle counters
                    battle_moves = 0
                    battle_switches = 0
                    battle_switch_slots = [0] * 6
                    battle_turns = 0

                    # --- LOUD FORFEIT FLAG: if this happened, it's a BUG ---
                    is_forfeit = recent_battles[-1]["forfeit"]
                    if is_forfeit:
                        log.warning("!!! FORFEIT DETECTED [battle %d, %s] -- a side forfeited early. THIS SHOULD NOT HAPPEN. Check the replay for WHICH side forfeited, and env timeout/switch logic.", battles_done, key)

                    # --- DEGENERATE SWITCHING (PER-BATTLE, so one bad game is not diluted
                    # by the 20-battle window): frequent spam of ONE bench slot. The per-turn
                    # rate keeps a legit long stall (many switches over many turns) unflagged.
                    # ---
                    last = recent_battles[-1]
                    if (last["b"] not in warned_degen
                            and last["sw_per_turn"] >= args.degen_sw_per_turn
                            and last["sw"] >= DEGEN_MIN_SW):
                        warned_degen.add(last["b"])
                        log.warning(
                            "!!! DEGENERATE SWITCHING [battle %d, %s]: %.2f switches/turn over %d turns "
                            "(%d switches total). Looks like a switch-spam loop / forced-switch-mask bug, not a "
                            "real choice -- open this replay and check env.get_action_mask / _decide.",
                            last["b"], key, last["sw_per_turn"], last["turns"], last["sw"])
                    
                    # Re-arm ONLY if we still need battles. Once we hit max_battles we
                    # must NOT start a new battle here -- that would leave it in progress
                    # at teardown (the old source of spurious 'forfeit' replays). The
                    # drain step below finishes the envs that are still mid-battle.
                    if battles_done < args.max_battles:
                        obs, ok = safe_reset(e)
                        if not ok:
                            skipped_envs.add(e.id)
                            continue
                        cur_obs[e.id] = obs
                        cur_mask[e.id] = e.last_mask
                    else:
                        e.done = True  # target reached; leave this env finished
                    if battles_done % args.log_every == 0:
                        log_stats("battles")
                        illegal_window = 0
                    if buffer_size() >= args.buffer_size:
                        break

            # --- safety: every env slot dropped -> stop (avoids an infinite no-op
            #     loop now that a live-battle re-arm ABORTS instead of forfeiting) ---
            if len(skipped_envs) >= len(envs):
                log.error("all %d env slots were dropped -- aborting the run "
                          "(see LIVE-BATTLE AT RE-ARM warnings above)", len(envs))
                break

            # --- update policy when buffer is full ---
            if buffer_size() >= args.buffer_size:
                t0 = time.time()
                upstats = policy.update()
                if args.opponent == "frozen" and policy.n_updates % args.sync_every == 0:
                    policy.sync_opponent(opponent)
                ck = os.path.join(args.ckpt_dir, "ppo.pt")
                policy.save(ck)

                sh = (upstats or {}).get("calib_sign_hit")
                sh_ok = sh is not None and not (isinstance(sh, float) and math.isnan(sh))

                # --- auto-save snapshots/best.pt on a rolling-best of the critic's game-read
                #     (so there is ALWAYS a known-good net to restore, without remembering)
                if sh_ok:
                    sign_hit_window.append(float(sh))
                if len(sign_hit_window) >= 2:
                    score = sum(sign_hit_window) / len(sign_hit_window)
                    if score > best_score + 1e-4:
                        best_score = score
                        best_path = os.path.join(args.ckpt_dir, "snapshots", "best.pt")
                        try:
                            os.makedirs(os.path.dirname(best_path), exist_ok=True)
                            shutil.copy2(ck, best_path)
                            log.info("saved snapshots/best.pt (rolling calib_sign_hit=%.3f @ update %d)", score, policy.n_updates)
                        except Exception as exc:
                            log.warning("best.pt save failed: %s", exc)

                # --- REGRESSION: critic's game-read (sign-hit) falling off its best ---
                if sh_ok:
                    if sh > best_calib:
                        best_calib = float(sh)
                        regress_streak = 0
                    elif sh < best_calib - CALIB_REGRESS_MARGIN:
                        regress_streak += 1
                        if regress_streak >= REGRESS_STREAK_N:
                            log.warning(
                                "!!! REGRESSION (critic): calib_sign_hit=%.3f for %d straight updates below best %.3f (margin %.2f). "
                                "Net may be degrading -- restore snapshots/best.pt or lower --lr / --ppo-epochs.",
                                sh, regress_streak, best_calib, CALIB_REGRESS_MARGIN)
                            regress_streak = 0  # cooldown; re-warns if it keeps being bad
                    else:
                        regress_streak = 0

                # --- INSTABILITY: policy shifting too hard between updates ---
                cf = (upstats or {}).get("clip_frac")
                if cf is not None and not (isinstance(cf, float) and math.isnan(cf)):
                    if cf > CLIP_FRAC_HIGH:
                        clip_streak += 1
                        if clip_streak >= CLIP_STREAK_N:
                            log.warning(
                                "!!! UNSTABLE UPDATES: clip_frac=%.3f for %d straight updates (threshold %.2f). "
                                "Lower --lr or --ppo-epochs to stop the policy overshooting.",
                                cf, clip_streak, CLIP_FRAC_HIGH)
                            clip_streak = 0  # cooldown
                    else:
                        clip_streak = 0

                log_stats("update", {
                    "update": policy.n_updates,
                    "dt": round(time.time() - t0, 3),
                    "upstats": {k: round(v, 5) for k, v in (upstats or {}).items()},
                })

        # --- Graceful teardown (root-cause fix for spurious 'forfeit' replays): ---
        # Play out any battle still in progress so close() has nothing live to abandon.
        # Without this, the old close() would forfeit the last in-progress battle on each
        # env -- exactly the 'X forfeited / win Y' replays we saw one-per-env at run end.
        drained = _drain(args.drain_max_steps, args.drain_timeout)

        # Capture the drained battles' transitions (they flushed to the buffer on done).
        if buffer_size() > 0:
            t0 = time.time()
            upstats = policy.update()
            policy.save(os.path.join(args.ckpt_dir, "ppo.pt"))
            log_stats("final-update", {
                "update": policy.n_updates,
                "dt": round(time.time() - t0, 3),
                "upstats": {k: round(v, 5) for k, v in (upstats or {}).items()},
            })
        battles_done += drained
        log_stats("run-end", {"battles": battles_done, "drained": drained, "elapsed_s": round(time.time() - start, 1)})
    except KeyboardInterrupt:
        log.info("interrupted (Ctrl+C) -- saving checkpoint")
        _best_effort_save(policy, args.ckpt_dir)
        log_stats("interrupt", {"battles": battles_done})
    except Exception as exc:
        # Safety net for an UNATTENDED run: any unexpected error (a save failure, a
        # torch/CUDA error, a MemoryError, a bad server frame, ...) is logged LOUDLY
        # (grep for "UNEXPECTED ERROR"), the best checkpoint we can is saved, and we
        # exit cleanly -- instead of dying on a raw traceback a monitoring agent
        # wouldn't know what to do with. Frequent in-loop saves already limit how much
        # work is lost (at most back to the last successful save).
        log.error("!!! UNEXPECTED ERROR -- run aborted at battle %d: %s: %s",
                  battles_done, type(exc).__name__, exc)
        traceback.print_exc()
        _best_effort_save(policy, args.ckpt_dir)
        log_stats("aborted", {"battles": battles_done, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        for e in envs:
            e.close()
        log.info("done. battles=%d elapsed=%.1fs", battles_done, time.time() - start)


# --------------------------------------------------------------------------- #
# Smoke test: prove the whole pipeline works on CPU, no heavy training
# --------------------------------------------------------------------------- #
def smoke(args) -> None:
    """Prove the full pipeline works with no heavy training: env + teams +
    masked policy + one PPO update, across an OU format and a random-battle format.

    Device-aware: default is CPU (safe -- never contends with a GPU in use). Pass
    ``--smoke --device cuda`` for a quick GPU preflight (a few CUDA act() + one CUDA
    PPO update + GPU memory report) before launching the long run.
    """
    logging.basicConfig(level=logging.WARNING)
    device = args.device if args.device in ("cpu", "cuda") else "cpu"
    print(f"=== SMOKE TEST ({device.upper()}) ===")
    print(f"OBS_DIM={OBS_DIM}  ACTIONS={UNIFIED_ACTION_SIZE}")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False -- "
                             "check the driver / that the GPU is not fully held by another process.")
        print(f"  GPU: {torch.cuda.get_device_name(0)} "
              f"| free/total = {torch.cuda.mem_get_info(0)[0]/1e9:.1f}/{torch.cuda.mem_get_info(0)[1]/1e9:.1f} GB")

    policy = MaskedPPO(obs_dim=OBS_DIM, n_actions=UNIFIED_ACTION_SIZE,
                       hidden=(64, 64), device=device, seed=0)

    # --- OU format: teams come from our pool ---
    env = ShowdownEnv(
        battle_format="gen7ou",
        team=RandomTeamBuilder(battle_format="gen7ou", seed=0),
        strict=False,
        save_replays=args.replays,
        log_level=logging.WARNING,
    )
    te = TraineeEnv(env, 0, "gen7ou")
    obs = te.reset()
    assert obs.shape == (OBS_DIM,), f"obs shape {obs.shape} != ({OBS_DIM},)"
    assert te.last_mask.shape == (UNIFIED_ACTION_SIZE,), "mask shape wrong"
    assert np.any(te.last_mask), "action mask is all-zero!"
    print(f"[gen7ou reset] obs={obs.shape} | legal_actions={int(te.last_mask.sum())} "
          f"| finite={bool(np.all(np.isfinite(obs)))}")

    cur_obs, cur_mask = obs, te.last_mask
    steps = 0
    total_reward = 0.0
    done = False
    wait_steps = 0
    while not done and steps < 200:
        # Both sides decide inside step() (P1 = policy, P2 = random). Re-ask + small
        # penalty for an illegal pick happen there; the decision is on te.last_self.
        new_obs, r, done = te.step(policy, "random")
        ls = te.last_self
        if ls.get("made"):
            a = int(ls["a"])
            policy.store(0, ls["obs"], a, ls["lp"], r, done, ls["mask"], ls.get("v", 0.0))
        else:
            # Showdown 'wait' request: P1 made no decision; record the step pass.
            policy.store_pass(0, cur_obs, r, policy.value(cur_obs), done)
            wait_steps += 1
        total_reward += r
        cur_obs, cur_mask = new_obs, te.last_mask
        steps += 1
    print(f"[gen7ou battle] steps={steps} wait_steps={wait_steps} re_picks={te.re_picks} "
          f"terminal_reward={total_reward:+.0f} (+1 win / -1 loss) | won={env.battle1.won} lost={env.battle1.lost}")

    # --- random-battle format: teams are server-generated (team=None) ---
    env2 = ShowdownEnv(battle_format="gen7randombattle", team=None, strict=False,
                       save_replays=args.replays, log_level=logging.WARNING)
    te2 = TraineeEnv(env2, 1, "gen7randombattle")
    o2 = te2.reset()
    print(f"[gen7randombattle reset] ok | obs={o2.shape} legal={int(te2.last_mask.sum())}")

    # --- one PPO update on the collected OU transitions ---
    up = policy.update()
    print(f"[ppo] update ran | {up}")

    te.close(); te2.close()
    print("=== SMOKE TEST PASSED ===")


def main() -> None:
    p = argparse.ArgumentParser(description="Pokemon Showdown self-play RL training")
    p.add_argument("--formats", default="gen7ou,gen7randombattle")
    p.add_argument("--weights", default="0.5,0.5")
    p.add_argument("--num-envs", type=int, default=4)
    p.add_argument("--team-mode", choices=["strong", "random", "mixed"], default="strong",
                   help="strong=coherent curated teams; random=competitive-set bags; "
                        "mixed=draw BOTH (the launcher's default)")
    p.add_argument("--p-strong", type=float, default=0.5,
                   help="mixed mode: probability a drawn team is a coherent 'strong' "
                        "team (default 0.5)")
    p.add_argument("--opponent", choices=["random", "frozen", "live"], default="live",
                   help="random/frozen = P1-only learning; live = SHARED policy plays "
                        "both sides and both transitions train the model (default)")
    p.add_argument("--buffer-size", type=int, default=2048)
    p.add_argument("--max-battles", type=int, default=200)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", default="128,128")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sync-every", type=int, default=5)
    p.add_argument("--degen-sw-per-turn", type=float, default=0.33,
                   help="flag a battle as degenerate switching when switches/turn >= this (default 0.33)")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--drain-max-steps", type=int, default=2500,
                   help="per-env step cap for finishing in-progress battles at teardown (safety net, set above the turn limit)")
    p.add_argument("--drain-timeout", type=float, default=1200.0,
                   help="per-env wall-clock seconds allowed to finish one in-progress battle at teardown (safety net)")
    p.add_argument("--ckpt-dir", default=os.path.join(os.path.dirname(__file__), "checkpoints"))
    p.add_argument("--replays", default=os.path.join(os.path.dirname(__file__), "replays"))
    p.add_argument("--init-actor", default=None,
                   help="path to a pre-trained actor state_dict to warm-start from (optional)")
    p.add_argument("--resume", default=None,
                   help="path to a FULL checkpoint (actor+critic+opt) to continue training from (e.g. checkpoints/ppo.pt)")
    p.add_argument("--server-ports", default=None,
                   help="comma-separated list of Showdown server ports (e.g. 8000,8001,8002)")
    p.add_argument("--smoke", action="store_true", help="run a quick CPU pipeline check")
    args = p.parse_args()

    if args.smoke:
        smoke(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
