"""Battle RECORD capture — the ONE place a live battle becomes a stored record.

The record format is the project's established BC fixture shape
(``bot/data/replays/*.jsonl``): ``{id, formatid, format, rating, players, log}``
— byte-compatible, so ``corpus.pull.load_local`` + ``corpus.trajectories.
replay_battle`` (the BC walker) consume a self-play record with ZERO changes.
Self-play adds a few metadata fields on top (``ts``, ``winner_side``, ``elo``,
``bots``) that the fixture pipeline simply ignores.

Capture uses the project's established log pattern — ``battle._build_replay_log()``
(same call ``net.env.build_obs`` and ``net.human_play`` use), which guarantees the
terminal ``|win|``/``|tie|`` line is present even when the protocol dropped it.
"""
from __future__ import annotations

import datetime
from typing import Any, List, Optional


def _canonicalize_win(log: str, battle: Any) -> str:
    """Normalize the terminal ``|win|`` line to the side-token form.

    poke-env's ``_build_replay_events`` has a fallback for battles where the
    server dropped the terminal event: it appends ``|win|<username>`` (the RAW
    username, POV-dependent — verified in
    ``poke_env/battle/abstract_battle.py::_build_replay_events``). We rewrite
    such lines to the canonical ``|win|p1a``/``|win|p2a`` (POV-independent,
    matching the fixture log format) using the battle object's own identity
    (``player_role`` = the local side, ``player_username``/``opponent_username``
    = the two real names). Returns the log unchanged if it is already
    canonical or cannot be resolved.
    """
    lines = log.split("\n")
    idx = None
    target = None
    for i in range(len(lines) - 1, -1, -1):
        parts = lines[i].split("|")
        if len(parts) >= 2 and parts[1] == "win":
            idx = i
            target = parts[2] if len(parts) >= 3 else None
            break
    if idx is None or target is None or target.startswith(("p1", "p2")):
        return log
    role = None
    if target == getattr(battle, "player_username", None):
        role = getattr(battle, "player_role", None)
    elif target == getattr(battle, "opponent_username", None):
        local = getattr(battle, "player_role", None)
        if local in ("p1", "p2"):
            role = "p2" if local == "p1" else "p1"
    if role is None:                      # last resort: the battle's own result
        local = getattr(battle, "player_role", None)
        if local in ("p1", "p2"):
            won = getattr(battle, "won", None)
            if won is True:
                role = local
            elif won is False:
                role = "p2" if local == "p1" else "p1"
    if role not in ("p1", "p2"):
        return log
    lines[idx] = f"|win|{role}a"
    return "\n".join(lines)


def _winner_side_from_log(log: str) -> Optional[str]:
    """Parse the terminal side from the log's LAST ``|win|``/``|tie|`` line.

    POV-independent (the log is always p1/p2-relative), so this is the source of
    truth even when the caller only knows their own side. ``None`` = tie/unknown.
    """
    side = None
    for line in reversed(log.split("\n")):
        parts = line.split("|")
        if len(parts) < 2 or parts[1] != "win":   # tag is parts[1]; parts[0] is always ''
            continue
        # ``|win|p2a`` splits to ['', 'win', 'p2a'] — the side token is parts[2]
        # (the D92.1 lesson: protocol lines carry an empty leading field)
        target = parts[2] if len(parts) > 2 else ""
        for token in ("p1a", "p1b", "p1c", "p1d", "p2a", "p2b", "p2c", "p2d"):
            if target.startswith(token):
                side = token[:2]
                break
        if side is None and target.startswith(("p1", "p2")):
            side = target[:2]
        break                      # only the LAST win line counts
    return side


def make_record(battle: Any,
                formatid: str,
                players: Optional[List[str]] = None,
                elo: Optional[float] = None,
                bots: Optional[List[str]] = None,
                format_name: Optional[str] = None,
                ts: Optional[str] = None) -> dict:
    """Build the fixture-compatible record for ONE finished ``battle``.

    :param battle: a poke-env ``Battle`` (duck-typed: ``battle_tag``, ``rating``,
                   ``_build_replay_log()``) — either side's battle object works.
    :param formatid: e.g. ``"gen9ou"``.
    :param players: ``[p1_username, p2_username]`` (log order); ``None`` = the
                   battle's real usernames (poke-env generates unique names).
    :param elo: the league ELO delta/rating for this battle (P5); ``None`` now.
    :param bots: bot identifiers (the league's multi-bot era); ``None`` for P4.
    :param format_name: human display name (defaults to the formatid).
    :param ts: ISO-8601 timestamp (defaults to now, UTC).
    """
    log = _canonicalize_win(battle._build_replay_log(), battle)
    return {
        # ── the BC fixture fields (byte-compatible) ─────────────────────
        "id": str(battle.battle_tag),
        "formatid": formatid,
        "format": format_name or formatid,
        "rating": float(battle.rating or 0.0),
        # default to the battle's REAL usernames (poke-env generates unique
        # names per env — the caller's label may be cosmetic)
        "players": list(players) if players is not None
        else list(getattr(battle, "players", None) or []),
        "log": log,
        # ── self-play metadata (ignored by the fixture pipeline) ─────────
        "ts": ts or datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "winner_side": _winner_side_from_log(log),
        "elo": elo,
        "bots": list(bots) if bots else None,
    }
