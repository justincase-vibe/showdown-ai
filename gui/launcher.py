"""P4.3 — start / stop / inspect the trainer as a DETACHED child process.

The GUI (``gui.server``) is a thin control plane: it writes intent to disk
(config / stop flag) and reads state from disk (``gui.runstate``). The
training itself runs in its OWN python process (``gui.trainer_main``) that:

  * is spawned with a NEW process group and NO console window (Windows), so
    Ctrl+C / kill / restart of the GUI cannot take the training down with it;
  * writes its log, heartbeat, metrics and checkpoints into the run dir;
  * polls ``stop.flag`` between battles for a graceful stop.

All of this is CPU-verifiable (a 1-battle CPU run proves the loop end to
end); the user's GPU training is the same code with ``device="cuda"``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from . import runstate

#: project root (gui/ -> root); the child runs with this as its cwd so
#: ``python -m gui.trainer_main`` + the ``net``/``belief``/``damage`` packages
#: resolve no matter where the GUI server was started from.
ROOT = Path(__file__).resolve().parents[1]


def python_exe() -> str:
    return sys.executable or "python"


def default_runs_dir() -> Path:
    return ROOT / "runs"


# ── config validation (loud + specific, before anything is spawned) ─────────

def _pos_int(v: Any, name: str, problems: list, minimum: int = 1) -> None:
    if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
        problems.append(f"{name} must be an int >= {minimum} (got {v!r})")


def _finite_float(v: Any, name: str, problems: list, lo: float = 0.0,
                  hi: float = None) -> None:
    import math
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(float(v)):
        problems.append(f"{name} must be a finite number (got {v!r})")
        return
    x = float(v)
    if x <= lo or (hi is not None and x > hi):
        problems.append(f"{name} must be in ({lo}, {hi if hi is not None else 'inf'}) (got {x})")


def _resolve_ckpt_path(ckpt: str) -> Path:
    """Resolve a checkpoint path against the PROJECT ROOT when it is relative.

    ``browse_checkpoints()`` returns root-RELATIVE paths (``scratch/...``), and
    the GUI may be started from ANY working directory — so a bare
    ``Path(ckpt).is_file()`` check resolved against the GUI's CWD and wrongly
    reported a valid checkpoint as missing (the "typing the path by hand
    doesn't work" bug). Absolute paths are used as-is (``~`` expanded)."""
    p = Path(ckpt).expanduser()
    return p if p.is_absolute() else (ROOT / p)


def validate_train_config(d: Optional[Dict[str, Any]]) -> "TrainConfig":
    """Validate a GUI-submitted config into a ``TrainConfig`` (mode
    ``selfplay``, the default). Raises ``ValueError`` with every problem
    listed (not just the first)."""
    from net.trainer import TrainConfig
    tc = TrainConfig.from_dict(d or {})
    problems = []
    if not isinstance(tc.battle_format, str) or not tc.battle_format.strip():
        problems.append("battle_format must be a non-empty string (e.g. 'gen9ou')")
    if tc.device not in ("cpu", "cuda"):
        problems.append("device must be 'cpu' or 'cuda' (got %r)" % (tc.device,))
    _pos_int(tc.battles_per_round, "battles_per_round", problems)
    _pos_int(tc.n_rounds, "n_rounds", problems)
    _pos_int(tc.max_steps, "max_steps", problems)
    _pos_int(tc.epochs, "epochs", problems)
    _pos_int(tc.save_every, "save_every", problems)
    _finite_float(tc.lr, "lr", problems, lo=0.0)
    _finite_float(tc.gamma, "gamma", problems, lo=0.0, hi=1.0)
    _finite_float(tc.lam, "lam", problems, lo=0.0, hi=1.0)
    _finite_float(tc.clip_eps, "clip_eps", problems, lo=0.0, hi=1.0)
    if tc.resume_from is not None:
        if not isinstance(tc.resume_from, str) or not tc.resume_from.strip():
            problems.append("resume_from must be a checkpoint path (or null)")
        else:
            # resolve a RELATIVE path against the project root (not the GUI's
            # CWD) and carry the ABSOLUTE path into the config, so the spawned
            # child loads it correctly regardless of its own CWD.
            resolved = str(_resolve_ckpt_path(tc.resume_from))
            tc.resume_from = resolved
            if not Path(resolved).exists():
                problems.append(f"resume_from does not exist: {tc.resume_from!r}")
    if problems:
        raise ValueError("; ".join(problems))
    return tc


# ── process control ──────────────────────────────────────────────────────────

def validate_bc_config(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a GUI-submitted BC/imitation config (mode ``bc``) into a
    normalized dict. Same loud contract as :func:`validate_train_config`
    (every problem listed). The fixture must EXIST — a missing replay file
    is a config error (400 before spawn), not a trainer crash (D86 contract).
    ``out_dir`` is intentionally NOT a field: the entry always checkpoints
    into the run dir (per-run separation, like the self-play entry)."""
    d = d or {}
    problems: list = []
    _pos_int(d.get("max_decisions"), "max_decisions", problems)
    _pos_int(d.get("epochs"), "epochs", problems)
    _pos_int(d.get("batch_size"), "batch_size", problems)
    if not isinstance(d.get("seed"), int) or isinstance(d.get("seed"), bool) or d.get("seed") < 0:
        problems.append(f"seed must be an int >= 0 (got {d.get('seed')!r})")
    _finite_float(d.get("lr"), "lr", problems, lo=0.0)
    _finite_float(d.get("heldout_frac"), "heldout_frac", problems, lo=0.0, hi=1.0)
    device = d.get("device", "cpu")
    if device not in ("cpu", "cuda"):
        problems.append(f"device must be 'cpu' or 'cuda' (got {device!r})")
    # ── the replay source: EITHER a raw fixture path OR the library ────────
    replay_source = d.get("replay_source")
    fixture = d.get("fixture")
    rs_n = None
    if replay_source is not None:
        if fixture is not None:
            problems.append("provide EITHER 'fixture' (a path) OR 'replay_source' "
                            "(the replay library) — not both")
        if not isinstance(replay_source, dict):
            problems.append(f"replay_source must be an object {{n, key}} "
                            f"(got {type(replay_source).__name__})")
        else:
            rs_n = replay_source.get("n")
            if not isinstance(rs_n, int) or isinstance(rs_n, bool) or rs_n < 1:
                problems.append(f"replay_source.n must be an int >= 1 (got {rs_n!r})")
            rs_key = replay_source.get("key", "elo")
            if rs_key not in ("elo", "rating", "ts", "last"):
                problems.append(f"replay_source.key must be one of "
                                f"'elo','rating','ts','last' (got {rs_key!r})")
    else:
        if not isinstance(fixture, str) or not fixture.strip():
            problems.append("provide a 'fixture' replay JSONL path "
                            "(e.g. 'bot/data/replays/gen9ou.jsonl') OR a "
                            "'replay_source' {n, key} from the replay library")
        elif not _resolve_ckpt_path(fixture).is_file():
            problems.append(f"fixture does not exist (relative to the project root): {fixture!r}")

    # n_battles: required for a raw fixture; for replay_source it defaults to
    # the selection size (a one-click "imitate the top-N" needs only ONE knob).
    nb = d.get("n_battles", rs_n if replay_source is not None else None)
    _pos_int(nb, "n_battles", problems)
    ckpt = d.get("ckpt_name")  # None = auto-generate ({n}b_bc_{id}.pt, D99)
    if ckpt is not None:
        if not isinstance(ckpt, str) or not ckpt.strip() or "/" in ckpt or "\\" in ckpt:
            problems.append(f"ckpt_name must be a plain file name or null (got {ckpt!r})")
    # initial_ckpt: the optional "continue training / fine-tune from" starting
    # checkpoint (D102.3). None/empty = random init. When set it must EXIST
    # (a 400, not a trainer crash) and is resolved to an ABSOLUTE path so the
    # spawned child loads it regardless of its CWD.
    init_ckpt = d.get("initial_ckpt")
    init_ckpt_out = None
    if init_ckpt is not None:
        if not isinstance(init_ckpt, str) or not init_ckpt.strip():
            problems.append("initial_ckpt must be a checkpoint path (.pt) or null")
        else:
            init_ckpt_out = str(_resolve_ckpt_path(init_ckpt))
            if not Path(init_ckpt_out).is_file():
                problems.append(f"initial_ckpt does not exist: {init_ckpt!r}")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "mode": "bc",
        "n_battles": nb,
        "fixture": fixture,
        "replay_source": replay_source,
        "max_decisions": d["max_decisions"],
        "heldout_frac": d["heldout_frac"],
        "epochs": d["epochs"],
        "batch_size": d["batch_size"],
        "lr": d["lr"],
        "device": device,
        "seed": d["seed"],
        "ckpt_name": ckpt,
        "initial_ckpt": init_ckpt_out,
    }


def validate_eval_config(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a GUI-submitted EVAL config (mode ``eval``) into a normalized
    dict. The checkpoint must EXIST (a missing .pt is a 400, not a crash)."""
    d = d or {}
    problems: list = []
    _pos_int(d.get("n_battles"), "n_battles", problems)
    device = d.get("device", "cpu")
    if device not in ("cpu", "cuda"):
        problems.append(f"device must be 'cpu' or 'cuda' (got {device!r})")
    if not isinstance(d.get("seed"), int) or isinstance(d.get("seed"), bool) or d.get("seed") < 0:
        problems.append(f"seed must be an int >= 0 (got {d.get('seed')!r})")
    fmt = d.get("battle_format", "gen9ou")
    if not isinstance(fmt, str) or not fmt.strip():
        problems.append("battle_format must be a non-empty string")
    ckpt = d.get("ckpt_path")
    ckpt_out = ckpt
    if not isinstance(ckpt, str) or not ckpt.strip():
        problems.append("ckpt_path must be a checkpoint file path (.pt)")
    else:
        ckpt_out = str(_resolve_ckpt_path(ckpt))
        if not Path(ckpt_out).is_file():
            problems.append(f"ckpt_path does not exist: {ckpt!r}")
    if problems:
        raise ValueError("; ".join(problems))
    return {
        "mode": "eval",
        "ckpt_path": ckpt_out,
        "n_battles": d["n_battles"],
        "device": device,
        "seed": d["seed"],
        "battle_format": fmt,
    }


def validate_play_config(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a GUI-submitted PLAY config (mode ``play``) into a normalized
    dict — the interactive "you vs the checkpoint" battle (D92.3).

    Same loud contract as the other validators (every problem listed, not just
    the first). Differences from ``eval`` (the bot plays YOU, not a random
    opponent, and you stop it when done):

      * ``n_battles`` may be ``0`` = "play until I stop it" (the default); the
        bot stays connected and keeps taking battles after each one finishes.
      * ``bot_user`` is REQUIRED (the username the bot connects as); ``you`` is
        OPTIONAL (the bot works without it — it's only used to render the
        ``/challenge`` hint) so an empty string is a valid value, not a 400.
      * ``greedy`` defaults to ``True`` (deterministic argmax = what you trained);
        unchecking it samples from the policy (stochastic).
      * ``ckpt_path`` must EXIST as a file (a missing .pt is a 400, not a crash).

    ``team_seed`` is optional (int >= 0 or null) — a seed for the bot's curated
    team draw, so you can replay the same bot team if you want.
    """
    d = d or {}
    problems: list = []

    n_battles = d.get("n_battles", 0)
    if not isinstance(n_battles, int) or isinstance(n_battles, bool) or n_battles < 0:
        problems.append(f"n_battles must be an int >= 0 (0 = play until stopped; "
                        f"got {n_battles!r})")

    device = d.get("device", "cpu")
    if device not in ("cpu", "cuda"):
        problems.append(f"device must be 'cpu' or 'cuda' (got {device!r})")

    greedy = d.get("greedy", True)
    if not isinstance(greedy, bool):
        problems.append(f"greedy must be a boolean (got {greedy!r})")

    fmt = d.get("battle_format", "gen9ou")
    if not isinstance(fmt, str) or not fmt.strip():
        problems.append("battle_format must be a non-empty string (e.g. 'gen9ou')")

    bot_user = d.get("bot_user", "checkpointbot")
    if not isinstance(bot_user, str) or not bot_user.strip():
        problems.append("bot_user must be a non-empty string (the bot's Showdown username)")

    you = d.get("you", "")
    if not isinstance(you, str):
        problems.append(f"you must be a string (your Showdown username; empty is allowed) "
                        f"(got {you!r})")

    ckpt = d.get("ckpt_path")
    ckpt_out = ckpt
    if not isinstance(ckpt, str) or not ckpt.strip():
        problems.append("ckpt_path must be a checkpoint file path (.pt)")
    else:
        # resolve a RELATIVE path against the project root (not the GUI's CWD)
        # and carry the ABSOLUTE path into the config, so the spawned child
        # loads it correctly no matter what CWD it runs from.
        ckpt_out = str(_resolve_ckpt_path(ckpt))
        if not Path(ckpt_out).is_file():
            problems.append(f"ckpt_path does not exist: {ckpt!r}")

    team_seed = d.get("team_seed")
    if team_seed is not None and (
            not isinstance(team_seed, int) or isinstance(team_seed, bool) or team_seed < 0):
        problems.append(f"team_seed must be an int >= 0 or null (got {team_seed!r})")

    if problems:
        raise ValueError("; ".join(problems))
    return {
        "mode": "play",
        "ckpt_path": ckpt_out,
        "n_battles": n_battles,
        "device": device,
        "greedy": greedy,
        "battle_format": fmt,
        "bot_user": bot_user,
        "you": you,
        "team_seed": team_seed,
    }


def validate_selfplay_config(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a GUI-submitted self-play-COLLECT config (mode
    ``selfplay-collect``) into a ``SelfPlayConfig`` dict. This mode runs the
    ``control/`` capture engine (pool + threaded battles + replay capture into
    the library) — the "self-play that produces fetchable replays" (the
    self-play -> fetch -> imitate loop, features 1 -> 2). Loud contract
    (every problem listed, not just the first). The RAM ceiling (n_servers x
    per_server) is enforced BY THE POOL, not here — but a nonsensical config is
    a 400 before spawn, not a crash. ``control`` is imported lazily so the
    launcher stays importable without pulling in the pool."""
    from control.selfplay import SelfPlayConfig
    d = d or {}
    problems: list = []
    _pos_int(d.get("n_battles", 8), "n_battles", problems)
    fmt = d.get("battle_format", "gen9ou")
    if not isinstance(fmt, str) or not fmt.strip():
        problems.append("battle_format must be a non-empty string (e.g. 'gen9ou')")
    _pos_int(d.get("n_servers", 2), "n_servers", problems)
    _pos_int(d.get("per_server", 2), "per_server", problems)
    _pos_int(d.get("port_base", 8000), "port_base", problems)
    _pos_int(d.get("seed", 20261119), "seed", problems)
    _pos_int(d.get("max_steps", 200), "max_steps", problems)
    _pos_int(d.get("max_consecutive_errors", 3), "max_consecutive_errors", problems)
    if problems:
        raise ValueError("; ".join(problems))
    out = SelfPlayConfig.from_dict(d).to_dict()
    out["mode"] = "selfplay-collect"
    return out


def validate_league_config(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate a GUI-submitted LEAGUE config (mode ``league``) into a
    ``LeagueConfig`` dict. Same loud contract as the other validators (every
    problem listed). The ROSTER is part of the contract: a league with fewer
    than 2 bots, or a bot whose checkpoint file is missing, is a config error
    (400 before spawn) — not a league_main crash after the pool booted.

    The roster file (``league/bots.json`` next to the project root by default)
    is the persistent league state; the config carries no bot list, so the same
    config re-fights the current roster (D100). The child runs with cwd=ROOT,
    so the default registry path is resolved HERE against ROOT — where the GUI
    may have been started from anywhere.

    The payload carries ``_rounds_planned`` (pairings x best_of) for the
    runstate protocol; :func:`start_trainer` pops it before writing config.json.
    ``league`` is imported lazily (no torch) to keep the launcher light."""
    from league.engine import LeagueConfig
    from league import pairing
    from league.registry import BotRegistry
    d = d or {}
    problems: list = []
    cfg = LeagueConfig.from_dict(d)
    if not isinstance(cfg.battle_format, str) or not cfg.battle_format.strip():
        problems.append("battle_format must be a non-empty string (e.g. 'gen9ou')")
    if cfg.pairing not in ("round_robin", "elo_weighted"):
        problems.append(f"pairing must be 'round_robin' or 'elo_weighted' (got {cfg.pairing!r})")
    _pos_int(cfg.best_of, "best_of", problems)
    _pos_int(cfg.max_steps, "max_steps", problems)
    _pos_int(cfg.n_servers, "n_servers", problems)
    _pos_int(cfg.per_server, "per_server", problems)
    _pos_int(cfg.port_base, "port_base", problems)
    if cfg.device not in ("cpu", "cuda"):
        problems.append(f"device must be 'cpu' or 'cuda' (got {cfg.device!r})")
    if not isinstance(cfg.seed, int) or isinstance(cfg.seed, bool) or cfg.seed < 0:
        problems.append(f"seed must be an int >= 0 (got {cfg.seed!r})")
    _pos_int(cfg.max_consecutive_errors, "max_consecutive_errors", problems)
    # ── the roster (the league's persistent state) ────────────────────────
    reg_path = (Path(cfg.registry_path) if cfg.registry_path
                else ROOT / "league" / "bots.json")
    reg = BotRegistry(reg_path)
    bots = reg.all()
    if len(bots) < 2:
        problems.append(
            f"league needs >= 2 bots in the roster (have {len(bots)}: "
            f"{[b.name for b in bots] or 'none'}; registry: {reg_path})")
    for b in bots:
        ck = Path(b.ckpt_path)
        if not ck.is_absolute():
            ck = ROOT / ck   # the child runs with cwd=ROOT
        if not ck.exists():
            problems.append(f"checkpoint for bot {b.name!r} does not exist: {b.ckpt_path}")
    if problems:
        raise ValueError("; ".join(problems))
    payload = cfg.to_dict()
    payload["registry_path"] = str(reg_path)
    base = (pairing.round_robin(bots) if cfg.pairing == "round_robin"
            else pairing.elo_weighted(bots))
    payload["_rounds_planned"] = max(1, int(cfg.best_of)) * len(base)
    return payload


# ── process control ──────────────────────────────────────────────────────────

def _popen_kwargs() -> Dict[str, Any]:
    kw: Dict[str, Any] = {}
    if os.name == "nt":
        # new process group (survives GUI Ctrl+C / console teardown) + no
        # console window popping up for the user.
        kw["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                               | subprocess.CREATE_NO_WINDOW)
    else:
        kw["start_new_session"] = True  # own session -> kill_tree via killpg
    return kw


def _read_mode(rd: Path) -> Optional[str]:
    """A run's mode ('selfplay','bc','eval','play','league',...), recorded at
    start. ``None`` for runs started before that (treated as non-pre-emptible)."""
    try:
        m = (Path(rd) / "mode").read_text(encoding="utf-8").strip()
        return m or None
    except (OSError, ValueError):
        return None


def _assert_startable(runs_dir: Path, mode: str) -> None:
    """A3 gate before spawning a new run: refuse if a run is already active,
    EXCEPT that a NON-play start may pre-empt a lingering PLAY session. Play is
    the low-priority interactive mode (``n_battles=0`` = "play until stopped"),
    so it must not hold the single slot forever and block real work — it is
    stopped gracefully first. Nothing else is ever pre-empted: a
    training/eval/league/bc run is never killed to start a play session, and
    two of the same workload still refuse. Raises RuntimeError on refusal."""
    active = find_active_run(runs_dir)
    if active is None:
        return
    rd, st = active
    active_mode = _read_mode(rd)
    if mode != "play" and active_mode == "play":
        stop_trainer(runs_dir=runs_dir, timeout=30)
        still = find_active_run(runs_dir)
        if still is not None:
            rd, st = still
            raise RuntimeError(
                f"refusing to start: could not stop the active PLAY session "
                f"{rd.name} (state={st.get('state')}, pid={st.get('pid')}) "
                f"— stop it manually first")
        return  # pre-empted the idle play session; proceed
    raise RuntimeError(
        f"refusing to start: run {rd.name} is already active "
        f"(state={st.get('state')}, pid={st.get('pid')}) — stop it first")


def start_trainer(cfg: Dict[str, Any], runs_dir: Optional[Path] = None
                  ) -> Dict[str, Any]:
    """Validate the config (by mode), refuse if a run is already active, and
    spawn the matching entry DETACHED:

      * ``mode == "selfplay"`` (the default) -> ``gui.trainer_main``
      * ``mode == "bc"``                     -> ``gui.bc_main`` (D87)
      * ``mode == "eval"``                   -> ``gui.eval_main`` (D92)
      * ``mode == "play"``                   -> ``net.human_play`` (D92.3)

    All speak the same ``gui.runstate`` protocol, so stop / dashboard /
    history work unchanged. Returns ``{run_id, pid, run_dir}``.

    ``play`` is the interactive "you vs the checkpoint" mode: the child connects
    to the local fork as ``bot_user`` and waits for YOU to challenge it in the
    browser. ``rounds_planned`` is ``n_battles`` (0 = unlimited — the run stays
    active until you stop it or it finishes N battles)."""
    cfg = dict(cfg or {})
    mode = cfg.get("mode", "selfplay")
    if mode == "bc":
        payload = validate_bc_config(cfg)
        entry = "gui.bc_main"
        rounds_planned = int(payload["epochs"])
    elif mode == "eval":
        payload = validate_eval_config(cfg)
        entry = "gui.eval_main"
        rounds_planned = int(payload["n_battles"])
    elif mode == "play":
        payload = validate_play_config(cfg)
        entry = "net.human_play"
        rounds_planned = int(payload["n_battles"])   # 0 = unlimited (play until stopped)
    elif mode == "selfplay":
        payload = validate_train_config(cfg).to_dict()
        entry = "gui.trainer_main"
        rounds_planned = int(payload["n_rounds"])
    elif mode == "selfplay-collect":
        # the capture engine: plays n_battles across the pool + saves every
        # replay into the library (feature 2's source). Spawns
        # control.selfplay_main, which owns the pool lifecycle.
        payload = validate_selfplay_config(cfg)
        entry = "control.selfplay_main"
        rounds_planned = int(payload["n_battles"])
    elif mode == "league":
        # D102: the league engine (checkpoint-vs-checkpoint over the roster).
        # league.league_main owns the pool lifecycle + the runstate protocol.
        payload = validate_league_config(cfg)
        entry = "league.league_main"
        rounds_planned = int(payload.pop("_rounds_planned", 1))
    else:
        raise ValueError(f"unknown mode {mode!r} (expected 'selfplay', "
                         f"'selfplay-collect', 'bc', 'eval', 'play', or 'league')")
    runs = Path(runs_dir) if runs_dir is not None else default_runs_dir()
    _assert_startable(runs, mode)   # A3: refuse if active — except a non-play
    # start gracefully pre-empts a lingering PLAY session (see _assert_startable).

    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rd = runs / run_id
    try:
        rd.mkdir(parents=True)
    except FileExistsError as exc:
        raise RuntimeError(f"run dir already exists: {rd}") from exc

    (rd / "config.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (rd / "mode").write_text(mode, encoding="utf-8")   # A3: lets a later start pre-empt an idle PLAY session

    log = open(rd / "trainer.log", "ab", buffering=0)
    try:
        cmd = [python_exe(), "-m", entry,
               "--config", str(rd / "config.json"), "--run-dir", str(rd)]
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, **_popen_kwargs())
    except BaseException:
        log.close()
        # no process was ever created: remove the half-created run dir so it
        # doesn't linger (list_runs would ignore it anyway — no status.json)
        import shutil
        shutil.rmtree(rd, ignore_errors=True)
        raise
    # the child now holds its own reference to trainer.log; our handle may stay
    # open harmlessly (same file, append mode)
    # D80: the status FIRST appears with the LIVE pid — a 'starting' status with
    # pid=None would let a concurrent reconcile() (e.g. the dashboard's poll)
    # finalize the run as 'failed' before the pid lands (narrow race, wrong verdict).
    runstate.write_status(rd, state="starting", run_id=run_id, pid=proc.pid,
                          started=runstate.now_iso(), round=0,
                          rounds_planned=rounds_planned, error=None, exit_code=None)
    return {"run_id": run_id, "pid": proc.pid, "run_dir": str(rd)}


def find_active_run(runs_dir: Path) -> Optional[Tuple[Path, Dict[str, Any]]]:
    """The newest run that is (still) active, or None. Reconciles stale
    final-state guesses for dead PIDs on the way (the GUI's backstop)."""
    runs_dir = Path(runs_dir)
    for run_id in runstate.list_runs(runs_dir):
        rd = runs_dir / run_id
        st = runstate.reconcile(rd)
        if st.get("state") in runstate.ACTIVE_STATES and runstate.pid_alive(st.get("pid")):
            return rd, st
    return None


def stop_trainer(runs_dir: Optional[Path] = None, timeout: float = 180.0,
                 force: bool = False) -> Dict[str, Any]:
    """Request a graceful stop (the trainer checks between battles and exits
    cleanly after the current one — D80: the flag is checked BEFORE EVERY
    battle, so a stop lands within one battle + one update). If ``force``
    (or the grace period expires) kill the whole process tree. Finalizes the
    run status either way. D80: grace default 180s — a CPU battle can take
    ~60s + the update, so the old 90s would have force-killed inside that
    window."""
    runs = Path(runs_dir) if runs_dir is not None else default_runs_dir()
    active = find_active_run(runs)
    if active is None:
        raise RuntimeError("no active training run to stop")
    rd, st = active
    pid = int(st.get("pid") or 0)
    runstate.request_stop(rd)
    runstate.write_status(rd, stop_requested=True)

    if not force and pid:
        deadline = time.time() + max(5.0, timeout)
        while runstate.pid_alive(pid) and time.time() < deadline:
            time.sleep(1.0)
    if runstate.pid_alive(pid):
        # graceful stop did not land (or force was asked for): kill the tree
        runstate.kill_tree(pid)
        deadline = time.time() + 15.0
        while runstate.pid_alive(pid) and time.time() < deadline:
            time.sleep(0.5)

    # the trainer usually writes its own final status; this is the backstop
    # (only if it hasn't already recorded a final state — e.g. 'failed' from
    # a crash, which is preserved above rather than overwritten to 'stopped')
    cur = runstate.read_status(rd)
    if cur.get("state") not in runstate.FINAL_STATES:
        runstate.write_status(rd, state="stopped",
                              error=cur.get("error") or "stopped by the GUI")
    return {"run_id": rd.name, "pid": pid, "state": "stopped"}


def status(runs_dir: Optional[Path] = None, metrics_tail: int = 5) -> Dict[str, Any]:
    """The GUI's single status view: the newest run (active or final) plus
    environment info. Everything from disk — safe to call from any process."""
    runs = Path(runs_dir) if runs_dir is not None else default_runs_dir()
    out: Dict[str, Any] = {
        "active": False, "state": None, "run_id": None, "pid": None,
        "round": None, "rounds_planned": None, "battles_done": None,
        "heartbeat_age_s": None,
        "error": None, "stop_requested": False,
        "metrics_tail": [], "checkpoints": [], "cuda_available": cuda_available(),
    }
    runs_list = runstate.list_runs(runs)
    if not runs_list:
        return out
    rd = runs / runs_list[0]
    st = runstate.reconcile(rd)
    active = (st.get("state") in runstate.ACTIVE_STATES
              and runstate.pid_alive(st.get("pid")))
    out.update({
        "active": active,
        "state": st.get("state"),
        "run_id": rd.name,
        "pid": st.get("pid"),
        "round": st.get("round"),
        "rounds_planned": st.get("rounds_planned"),
        # self-play / league runs track progress by battles_done (round stays 0)
        "battles_done": st.get("battles_done"),
        "heartbeat_age_s": runstate.heartbeat_age_s(rd),
        "error": st.get("error"),
        "stop_requested": bool(st.get("stop_requested")),
        "metrics_tail": runstate.read_metrics(rd, metrics_tail),
        "checkpoints": runstate.checkpoints(rd),
        "started": st.get("started"),
    })
    return out


# ── environment ──────────────────────────────────────────────────────────────

_CUDA: Optional[bool] = None


def browse_checkpoints(root: Optional[Path] = None) -> list:
    """List all .pt checkpoints in scratch/ and runs/*/checkpoints/.

    Shared between the web dashboard (``gui/server.py::api_browse``) and the
    native GUI (Phase 2) so both reuse the SAME D97-fixed meta-reading logic.
    Each entry carries ``type``, ``top1``, ``verdict``, ``n_battles`` from
    the sibling ``.meta.json`` (D97: the runs-branch was fixed to read meta
    just like the scratch-branch).

    Returns a list of dicts (possibly empty)."""
    import time as _time
    root = Path(root) if root is not None else ROOT
    results: list = []
    scratch = root / "scratch"
    if scratch.is_dir():
        for p in sorted(scratch.glob("*.pt")):
            meta_path = p.with_name(p.name + ".meta.json")
            meta = {}
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            st = p.stat()
            results.append({
                "path": str(p.relative_to(root)),
                "name": p.name,
                "size_mb": round(st.st_size / 1e6, 1),
                "modified": _time.strftime("%Y-%m-%d %H:%M", _time.gmtime(st.st_mtime)),
                "type": "bc" if "bc" in p.name else "unknown",
                "n_battles": meta.get("n_battles"),
                "verdict": meta.get("verdict_pass"),
                "top1": (meta.get("final") or {}).get("top1_acc"),
            })
    runs_dir = root / "runs"
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.is_dir():
                continue
            for p in sorted(ckpt_dir.glob("*.pt")):
                st = p.stat()
                meta_path = p.with_name(p.name + ".meta.json")
                meta = {}
                if meta_path.is_file():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        pass
                results.append({
                    "path": str(p.relative_to(root)),
                    "name": p.name,
                    "size_mb": round(st.st_size / 1e6, 1),
                    "modified": _time.strftime("%Y-%m-%d %H:%M", _time.gmtime(st.st_mtime)),
                    "type": "bc" if "bc" in p.name else (
                        "selfplay" if ("selfplay" in p.name or "round" in p.name) else "unknown"),
                    "run_id": run_dir.name,
                    "n_battles": meta.get("n_battles"),
                    "verdict": meta.get("verdict_pass"),
                    "top1": (meta.get("final") or {}).get("top1_acc"),
                })
    return results


def cuda_available() -> bool:
    """Cached torch CUDA probe (the server imports torch once; it does NO
    training — this only reports what the user's machine can do)."""
    global _CUDA
    if _CUDA is None:
        try:
            import torch
            _CUDA = bool(torch.cuda.is_available())
        except Exception:
            _CUDA = False
    return _CUDA
