"""Replay VIEWER — export library records as client-accepted replay HTML (D101).

ONE file format, TWO viewing paths (format = the fork's ``BattleLog.createReplayFile``,
``_client_src/.../battle-log.ts:1811``):

  * **Browser (Chrome):** the file carries the ``replay-embed.js`` bootstrap,
    so it plays standalone from a ``file://`` URL (internet needed for CDN
    assets — sprites/sounds come from play.pokemonshowdown.com).
  * **PS client (offline):** drag & drop the .html onto the Showdown client
    window. The client only requires ``<title>`` + the ``battle-log-data``
    script tag (``panel-battle.tsx::handleDrop``) and replays fully client-side.

Verified contract details:
  * the log is ``/`` → ``\\/`` escaped (``handleDrop`` reverses it before
    parsing); it is NOT HTML-escaped (the client parses it as battle-log text);
  * title/attributes ARE HTML-escaped (``createReplayFile`` uses escapeHTML);
  * the ``replayid`` hidden input carries the record id (room name).

Shared by the GUI (``gui/desktop``) — the GUI CALLS these functions and never
reimplements the format (no code written twice).
"""
from __future__ import annotations

import html
import re
import webbrowser
from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: default export dir (gitignored, under ``runs/``)
DEFAULT_OUT_DIR = PROJECT_ROOT / "runs" / "replays_view"

#: inline stylesheet so replays are readable WITHOUT internet
#: (verbatim from createReplayFile, battle-log.ts)
_STYLE = """html,body {font-family:Verdana, sans-serif;font-size:10pt;margin:0;padding:0;}body{padding:12px 0;} .battle-log {font-family:Verdana, sans-serif;font-size:10pt;} .battle-log-inline {border:1px solid #AAAAAA;background:#EEF2F5;color:black;max-width:640px;margin:0 auto 80px;padding-bottom:5px;} .battle-log .inner {padding:4px 8px 0px 8px;} .battle-log .inner-preempt {padding:0 8px 4px 8px;} .battle-log .inner-after {margin-top:0.5em;} .battle-log h2 {margin:0.5em -8px;padding:4px 8px;border:1px solid #AAAAAA;background:#E0E7EA;border-left:0;border-right:0;font-family:Verdana, sans-serif;font-size:13pt;} .battle-log .chat {vertical-align:middle;padding:3px 0 3px 0;font-size:8pt;} .battle-log .chat strong {color:#40576A;} .battle-log .chat em {padding:1px 4px 1px 3px;color:#000000;font-style:normal;} .chat.mine {background:rgba(0,0,0,0.05);margin-left:-8px;margin-right:-8px;padding-left:8px;padding-right:8px;} .spoiler {color:#BBBBBB;background:#BBBBBB;padding:0px 3px;} .spoiler:hover, .spoiler:active, .spoiler-shown {color:#000000;background:#E2E2E2;padding:0px 3px;} .spoiler a {color:#BBBBBB;} .spoiler:hover a, .spoiler:active a, .spoiler-shown a {color:#2288CC;} .chat code, .chat .spoiler:hover code, .chat .spoiler:active code, .chat .spoiler-shown code {border:1px solid #C0C0C0;background:#EEEEEE;color:black;padding:0 2px;} .chat .spoiler code {border:1px solid #CCCCCC;background:#CCCCCC;color:#CCCCCC;} .battle-log .rated {padding:3px 4px;} .battle-log .rated strong {color:white;background:#89A;padding:1px 4px;border-radius:4px;} .spacer {margin-top:0.5em;} .message-announce {background:#6688AA;color:white;padding:1px 4px 2px;} .message-announce a, .broadcast-green a, .broadcast-blue a, .broadcast-red a {color:#DDEEFF;} .broadcast-green {background-color:#559955;color:white;padding:2px 4px;} .broadcast-blue {background-color:#6688AA;color:white;padding:2px 4px;} .infobox {border:1px solid #6688AA;padding:2px 4px;} .infobox-limited {max-height:200px;overflow:auto;overflow-x:hidden;} .broadcast-red {background-color:#AA5544;color:white;padding:2px 4px;} .message-learn-canlearn {font-weight:bold;color:#228822;text-decoration:underline;} .message-learn-cannotlearn {font-weight:bold;color:#CC2222;text-decoration:underline;} .message-effect-weak {font-weight:bold;color:#CC2222;} .message-effect-resist {font-weight:bold;color:#6688AA;} .message-effect-immune {font-weight:bold;color:#666666;} .message-learn-list {margin-top:0;margin-bottom:0;} .message-throttle-notice, .message-error {color:#992222;} .message-overflow, .chat small.message-overflow {font-size:0pt;} .message-overflow::before {font-size:9pt;content:'...';} .subtle {color:#3A4A66;}"""


def _esc(value: object) -> str:
    """HTML-escape a value for a title/attribute (createReplayFile uses escapeHTML)."""
    return html.escape(str(value), quote=True)


def build_replay_html(record: dict) -> str:
    """The full replay-file HTML for ONE library record.

    :param record: a :class:`replays.ReplayLibrary` record — needs ``id`` and
        ``log``; ``format``/``formatid`` and ``players`` are used for the title.
    :raises ValueError: if the record has no ``id``.
    """
    rid = str(record.get("id") or "")
    if not rid:
        raise ValueError("build_replay_html: record has no 'id'")
    tier = str(record.get("format") or record.get("formatid") or "unknown")
    players = record.get("players") or []
    p1 = str(players[0]) if len(players) > 0 else "Player 1"
    p2 = str(players[1]) if len(players) > 1 else "Player 2"
    data = str(record.get("log") or "").replace("/", "\\/")   # client contract

    parts = [
        "<!DOCTYPE html>\n",
        '<meta charset="utf-8" />\n',
        "<!-- version 1 -->\n",
        f"<title>{_esc(tier)} replay: {_esc(p1)} vs. {_esc(p2)}</title>\n",
        "<style>\n" + _STYLE + "\n",
        "</style>\n",
        '<div class="wrapper replay-wrapper" style="max-width:1180px;margin:0 auto">\n',
        f'<input type="hidden" name="replayid" value="{_esc(rid)}" />\n',
        '<div class="battle"></div><div class="battle-log"></div>'
        '<div class="replay-controls"></div><div class="replay-controls-2"></div>\n',
        f'<h1 style="font-weight:normal;text-align:center">'
        f"<strong>{_esc(tier)}</strong><br />"
        f"{_esc(p1)} vs. {_esc(p2)}</h1>\n",
        f'<script type="text/plain" class="battle-log-data">{data}</script>\n',
        "</div>\n",
        '<div class="battle-log battle-log-inline"><div class="inner"></div></div>\n',
        "</div>\n",
        "<script>\n",
        "let daily = Math.floor(Date.now()/1000/60/60/24);"
        "document.write('<script src=\"https://play.pokemonshowdown.com/js/"
        "replay-embed.js?version'+daily+'\"></'+'script>');\n",
        "</script>\n",
    ]
    return "".join(parts)


def write_replay_file(record: dict,
                      out_dir: Optional[Union[str, Path]] = None) -> Path:
    """Write ``{id}.html`` into ``out_dir`` (default :data:`DEFAULT_OUT_DIR`),
    idempotent for the same id (overwrite). Returns the file path."""
    rid = str(record.get("id") or "")
    if not rid:
        raise ValueError("write_replay_file: record has no 'id'")
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", rid)    # id becomes a FILENAME
    out = Path(out_dir) if out_dir is not None else DEFAULT_OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{safe}.html"
    path.write_text(build_replay_html(record), encoding="utf-8")
    return path


def open_replay_in_browser(record: dict,
                           out_dir: Optional[Union[str, Path]] = None) -> Path:
    """Write the replay file and open it in the default browser (Chrome here).

    Returns the written file path (the GUI shows it to the user)."""
    path = write_replay_file(record, out_dir)
    webbrowser.open(path.as_uri())
    return path


__all__ = ["build_replay_html", "write_replay_file", "open_replay_in_browser",
           "DEFAULT_OUT_DIR"]
