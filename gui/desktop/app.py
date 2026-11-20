"""Native tkinter GUI — Mission Control for the Showdown AI training pipeline.

Drives the SAME control plane as the web dashboard (``gui.launcher``,
``gui.runstate``, ``replays.ReplayLibrary``) — no duplication (D98).

Panels (tabs):
  * **Status**   — current run state, progress, heartbeat, CUDA info
  * **Start**    — mode selector + per-mode fields + Start button
  * **Replays**  — library summary + top-N / last-N preview
  * **Checkpoints** — all .pt files with meta (type, top1, verdict)
  * **Log**      — scrolling tail of the active run's trainer.log

The refresh loop uses ``root.after(~1500, poll)`` (the native equivalent of
the web dashboard's WebSocket poll).

Headless test: ``gui/tests/test_desktop.py`` constructs the app, builds all
widgets, and destroys — no display needed.
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional

# ── control-plane imports (shared with the web dashboard) ────────────────────
from gui import launcher, runstate

#: how often the GUI polls the run dir (same cadence as the web WS poll)
POLL_MS = 1500

#: collect-mode defaults (avoid :8000 — the user's personal Showdown server)
COLLECT_DEFAULTS = {
    "port_base": 49960,
    "n_servers": 2,
    "per_server": 2,
}


class DesktopApp:
    """The native Mission Control GUI.

    Usage:
        app = DesktopApp()
        app.run()   # starts the tkinter main loop

    Headless (test):
        app = DesktopApp(headless=True)
        app.build()  # creates root + all widgets
        app.root.destroy()
    """

    def __init__(self, headless: bool = False, root: Optional[tk.Tk] = None):
        """If ``root`` is given, build into that existing ``tk.Tk`` (embedding);
        otherwise :meth:`build` creates one. ``close()`` only destroys a root
        the app OWNS."""
        self.headless = headless
        self.root: Optional[tk.Tk] = root
        self._owns_root = root is None
        self._poll_id: Optional[str] = None
        self._runs_dir = launcher.default_runs_dir()

        # per-mode field frames (built lazily by _build_start_tab)
        self._mode_frames: Dict[str, ttk.Frame] = {}

        # replay library cache
        self._replay_summary: Dict[str, Any] = {}
        self._replay_battles: List[Dict[str, Any]] = []

        # D101 replay-export worker (main-thread queue drain — tkinter-safe)
        self._replay_msg_q: "queue.Queue" = queue.Queue()
        self._replay_worker: Optional[threading.Thread] = None

        # checkpoint cache
        self._checkpoints: List[Dict[str, Any]] = []
        # checkpoint dropdowns: a list of (var, {display_name: full_path}) so the
        # box can show a SHORT, distinguishing name (relative to the project root)
        # while the caller resolves it back to the full path via _resolve_ckpt().
        self._ckpt_maps: List[Any] = []

        # log tail
        self._log_text = ""

        # tkinter variables (created in build(), after the root exists)
        self._mode_var: Optional[tk.StringVar] = None
        self._replay_order: Optional[tk.StringVar] = None
        self._replay_n: Optional[tk.IntVar] = None

    # ── build (create all widgets) ───────────────────────────────────────────

    def build(self) -> None:
        """Create the root window (unless one was provided) and all widgets.
        Safe to call headlessly."""
        if self.root is None:
            self.root = tk.Tk()
            self._owns_root = True
        self.root.title("Showdown AI — Mission Control")
        self.root.geometry("900x640")
        self.root.minsize(720, 480)

        # tkinter variables (MUST be created after the root exists)
        self._mode_var = tk.StringVar(value="Imitation (BC)")
        self._replay_order = tk.StringVar(value="last")
        self._replay_n = tk.IntVar(value=10)
        # mode labels mapping (set in _build_start_tab)
        self._mode_labels: Dict[str, str] = {}

        # ── status bar (top) ─────────────────────────────────────────────
        self._build_status_bar()

        # ── tabbed main area ─────────────────────────────────────────────
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self._build_start_tab(notebook)
        self._build_replays_tab(notebook)
        self._build_checkpoints_tab(notebook)
        self._build_log_tab(notebook)

        # ── bottom bar (stop buttons) ────────────────────────────────────
        self._build_bottom_bar()

        # initial poll
        self._poll()

    def close(self) -> None:
        """Tear down: destroy the root if we own it, else only our widgets
        (an embedded app must not kill the host's Tk interpreter).
        Cancels the pending poll callback so it can't fire on dead widgets."""
        if self.root is None:
            return
        if self._poll_id is not None:
            try:
                self.root.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        if self._owns_root:
            self.root.destroy()
        else:
            for w in self.root.winfo_children():
                w.destroy()

    # ── status bar ─────────────────────────────────────────────────────────

    def _build_status_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(fill=tk.X)

        self._status_label = ttk.Label(bar, text="Status: idle", font=("", 10, "bold"))
        self._status_label.pack(side=tk.LEFT)

        self._progress_label = ttk.Label(bar, text="")
        self._progress_label.pack(side=tk.LEFT, padx=(16, 0))

        self._cuda_label = ttk.Label(bar, text="")
        self._cuda_label.pack(side=tk.RIGHT)

    # ── start tab ──────────────────────────────────────────────────────────

    def _build_start_tab(self, parent: ttk.Notebook) -> None:
        frame = ttk.Frame(parent, padding=8)
        parent.add(frame, text="Start a run")

        # mode selector
        ttk.Label(frame, text="Mode:").grid(row=0, column=0, sticky="w", pady=2)
        mode_var = self._mode_var  # already created in build()
        modes = [
            ("selfplay-collect", "Self-play (collect replays)"),
            ("selfplay", "Self-play (PPO training)"),
            ("bc", "Imitation (BC)"),
            ("league", "League (bot vs bot)"),
            ("play", "Play against a checkpoint"),
            ("eval", "Eval vs random"),
        ]
        cb = ttk.Combobox(
            frame, textvariable=mode_var,
            values=[label for _, label in modes],
            state="readonly", width=30,
        )
        cb.grid(row=0, column=1, columnspan=3, sticky="we", pady=2)
        # store the mode->key mapping
        self._mode_labels = {label: key for key, label in modes}
        cb.bind("<<ComboboxSelected>>", self._on_mode_change)

        # per-mode field frames
        self._fields_frame = ttk.Frame(frame)
        self._fields_frame.grid(row=1, column=0, columnspan=4, sticky="we", pady=4)

        # start button
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=2, column=0, columnspan=4, pady=6)
        self._start_btn = ttk.Button(btn_frame, text="▶  Start", command=self._on_start)
        self._start_btn.pack(side=tk.LEFT)
        self._start_msg = ttk.Label(btn_frame, text="")
        self._start_msg.pack(side=tk.LEFT, padx=8)

        # build the default mode's fields
        self._current_mode = "bc"  # default to BC (the most common task)
        self._on_mode_change(None)

    def _on_mode_change(self, _event: Optional[Any]) -> None:
        """Rebuild the per-mode field frame when the mode changes."""
        # destroy old fields
        for w in self._fields_frame.winfo_children():
            w.destroy()
        self._mode_frames.clear()

        label = self._mode_var.get()
        mode = self._mode_labels.get(label, "selfplay-collect")
        self._current_mode = mode

        if mode == "selfplay-collect":
            self._build_collect_fields()
        elif mode == "selfplay":
            self._build_ppo_fields()
        elif mode == "bc":
            self._build_bc_fields()
        elif mode == "play":
            self._build_play_fields()
        elif mode == "eval":
            self._build_eval_fields()
        elif mode == "league":
            self._build_league_fields()

    def _add_field(self, row: int, col: int, label: str, default: Any,
                   widget_type: str = "int", **kw) -> Any:
        """Add a labeled field to the current fields frame. Returns the tk var."""
        ttk.Label(self._fields_frame, text=label).grid(row=row, column=col, sticky="w", pady=2)
        if widget_type == "int":
            var = tk.IntVar(value=default)
            w = ttk.Entry(self._fields_frame, textvariable=var, width=kw.get("width", 8))
        elif widget_type == "float":
            var = tk.DoubleVar(value=default)
            w = ttk.Entry(self._fields_frame, textvariable=var, width=kw.get("width", 8))
        elif widget_type == "str":
            var = tk.StringVar(value=default)
            w = ttk.Entry(self._fields_frame, textvariable=var, width=kw.get("width", 20))
        elif widget_type == "bool":
            var = tk.BooleanVar(value=default)
            w = ttk.Checkbutton(self._fields_frame, textvariable=var)
        elif widget_type == "combo":
            # dropdown with a fixed value list (format / pairing / device)
            var = tk.StringVar(value=str(default))
            w = ttk.Combobox(
                self._fields_frame, textvariable=var,
                values=list(kw.get("values") or []),
                state="readonly", width=kw.get("width", 20),
            )
        else:
            raise ValueError(f"unknown widget_type {widget_type!r}")
        w.grid(row=row, column=col + 1, sticky="w", pady=2, padx=(4, 12))
        return var

    def _format_field(self, row: int, col: int, label: str = "Format:") -> Any:
        """A format dropdown fed by ``bot.teams.available_formats()`` — the
        single source of truth (D102): only formats with a usable team pool
        are selectable. Falls back to free text when no pool file exists yet."""
        from bot import teams
        try:
            values = teams.available_formats()
        except Exception:
            values = []
        default = "gen9ou" if "gen9ou" in values else (values[0] if values else "gen9ou")
        if not values:
            return self._add_field(row, col, label, default, "str")
        return self._add_field(row, col, label, default, "combo", values=values)

    def _ckpt_field(self, row: int, col: int, label: str = "Checkpoint:",
                    optional: bool = False) -> Any:
        """A checkpoint dropdown fed by ``launcher.browse_checkpoints()`` — the
        SAME list the Checkpoints tab shows (``scratch/*.pt`` +
        ``runs/*/checkpoints/*.pt``). Pick one instead of typing a path by hand.

        The box DISPLAYS the short, distinguishing name (relative to the project
        root) so the rows don't all read as the same truncated absolute path.
        Callers read it through :meth:`_resolve_ckpt` to get the full absolute
        path. The box stays EDITABLE so a custom path (relative or absolute) is
        still a valid fallback — :meth:`_resolve_ckpt` returns custom values
        as-is.

        ``optional=True`` (the "continue training / resume from" fields) leaves
        the box EMPTY by default — empty means "start from scratch"; the caller
        maps an empty value to null. Non-empty values are the listed
        (absolute) checkpoint paths."""
        try:
            cps = launcher.browse_checkpoints()
        except Exception:
            cps = []
        root = Path(launcher.ROOT).resolve()
        # Show the SHORT, distinguishing name (relative to the project root) so
        # the rows don't all read as the same truncated absolute path (the old
        # "unusable" bug). The full absolute path is stored in ``cmap`` and
        # resolved by _resolve_ckpt() at read time.
        entries: List[str] = []
        cmap: Dict[str, str] = {}
        for c in cps:
            p = c.get("path")
            if not p:
                continue
            ap = (Path(p) if Path(p).is_absolute() else (root / p)).resolve()
            try:
                display = ap.relative_to(root).as_posix()
            except ValueError:
                display = ap.name  # outside the project root -> just the file name
            if display in cmap:
                continue  # de-dup (the same file listed twice)
            entries.append(display)
            cmap[display] = str(ap)
        default = ("" if optional else (entries[0] if entries else ""))
        var = tk.StringVar(value=default)
        self._ckpt_maps.append((var, cmap))
        w = ttk.Combobox(self._fields_frame, textvariable=var, values=entries, width=46)
        w.grid(row=row, column=col + 1, sticky="we", pady=2, padx=(4, 12))
        return var

    def _resolve_ckpt(self, var: Any) -> str:
        """Map a checkpoint dropdown's SHORT display value back to its full
        absolute path. A custom-typed value (not in the list) is returned
        as-is, so a hand-typed relative/absolute path still works. Empty -> ""."""
        v = (var.get() or "").strip()
        if not v:
            return v
        for (var_ref, cmap) in self._ckpt_maps:
            if var_ref is var:
                return cmap.get(v, v)
        return v

    def _build_collect_fields(self) -> None:
        r = 0
        self._collect_n_battles = self._add_field(r, 0, "Battles:", 8, "int"); r += 1
        self._collect_fmt = self._format_field(r, 0); r += 1
        self._collect_n_servers = self._add_field(r, 0, "Servers:", COLLECT_DEFAULTS["n_servers"], "int"); r += 1
        self._collect_per_server = self._add_field(r, 0, "Per server:", COLLECT_DEFAULTS["per_server"], "int"); r += 1
        self._collect_port_base = self._add_field(r, 0, "Port base:", COLLECT_DEFAULTS["port_base"], "int"); r += 1
        self._collect_seed = self._add_field(r, 0, "Seed:", 42, "int"); r += 1
        self._collect_max_steps = self._add_field(r, 0, "Max steps:", 200, "int"); r += 1

    def _build_ppo_fields(self) -> None:
        r = 0
        self._ppo_fmt = self._format_field(r, 0); r += 1
        self._ppo_battles = self._add_field(r, 0, "Battles/round:", 1, "int"); r += 1
        self._ppo_rounds = self._add_field(r, 0, "Rounds:", 1, "int"); r += 1
        self._ppo_epochs = self._add_field(r, 0, "Epochs:", 1, "int"); r += 1
        self._ppo_lr = self._add_field(r, 0, "LR:", 3e-4, "float"); r += 1
        self._ppo_device = self._add_field(r, 0, "Device:", "cpu", "str"); r += 1
        # continue training from an existing checkpoint (empty = start from scratch)
        self._ppo_resume = self._ckpt_field(r, 0, "Resume from (optional):", optional=True); r += 1

    def _build_bc_fields(self) -> None:
        r = 0
        self._bc_fixture = self._add_field(r, 0, "Fixture path:", "bot/data/replays/gen9ou.jsonl", "str", width=40); r += 1
        self._bc_n = self._add_field(r, 0, "N battles:", 50, "int"); r += 1
        self._bc_epochs = self._add_field(r, 0, "Epochs:", 1, "int"); r += 1
        self._bc_batch = self._add_field(r, 0, "Batch size:", 8, "int"); r += 1
        self._bc_lr = self._add_field(r, 0, "LR:", 3e-3, "float"); r += 1
        self._bc_seed = self._add_field(r, 0, "Seed:", 42, "int"); r += 1
        self._bc_device = self._add_field(r, 0, "Device:", "cpu", "str"); r += 1
        # continue / fine-tune from an existing checkpoint (empty = random init)
        self._bc_resume = self._ckpt_field(r, 0, "Start from ckpt (optional):", optional=True); r += 1

    def _build_play_fields(self) -> None:
        r = 0
        self._play_ckpt = self._ckpt_field(r, 0); r += 1
        self._play_fmt = self._format_field(r, 0); r += 1
        self._play_bot_user = self._add_field(r, 0, "Bot user:", "checkpointbot", "str"); r += 1
        self._play_you = self._add_field(r, 0, "You:", "ttvJustinCase", "str"); r += 1
        self._play_device = self._add_field(r, 0, "Device:", "cpu", "str"); r += 1
        self._play_greedy = self._add_field(r, 0, "Greedy:", True, "bool"); r += 1
        ttk.Label(self._fields_frame, justify=tk.LEFT, wraplength=440,
                  text="On Start: the local server opens in your browser and the bot "
                       "challenges you there — just click accept. "
                       "(Fallback: log in and type  /challenge <Bot user>.)").grid(
                      row=r, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _build_eval_fields(self) -> None:
        r = 0
        self._eval_ckpt = self._ckpt_field(r, 0); r += 1
        self._eval_n = self._add_field(r, 0, "N battles:", 10, "int"); r += 1
        self._eval_fmt = self._format_field(r, 0); r += 1
        self._eval_seed = self._add_field(r, 0, "Seed:", 42, "int"); r += 1
        self._eval_device = self._add_field(r, 0, "Device:", "cpu", "str"); r += 1

    def _build_league_fields(self) -> None:
        """League (bot vs bot, D102): the roster lives in ``league/bots.json``
        (persistent — the config never carries a bot list). The roster panel
        below the fields shows the current bots and lets you ADD one (name +
        checkpoint path) before starting."""
        r = 0
        self._league_fmt = self._format_field(r, 0); r += 1
        self._league_pairing = self._add_field(
            r, 0, "Pairing:", "round_robin",
            "combo", values=["round_robin", "elo_weighted"]); r += 1
        self._league_best_of = self._add_field(r, 0, "Best of:", 1, "int"); r += 1
        self._league_greedy = self._add_field(r, 0, "Greedy:", True, "bool"); r += 1
        self._league_device = self._add_field(
            r, 0, "Device:", "cpu", "combo", values=["cpu", "cuda"]); r += 1
        self._league_n_servers = self._add_field(r, 0, "Servers:", 2, "int"); r += 1
        self._league_per_server = self._add_field(r, 0, "Per server:", 2, "int"); r += 1
        self._league_port_base = self._add_field(r, 0, "Port base:", 49960, "int"); r += 1
        self._league_max_steps = self._add_field(r, 0, "Max steps:", 200, "int"); r += 1
        self._league_seed = self._add_field(r, 0, "Seed:", 20261119, "int"); r += 1

        # ── roster panel ──────────────────────────────────────────────────
        roster_frame = ttk.LabelFrame(self._fields_frame, text="League roster", padding=4)
        roster_frame.grid(row=r, column=0, columnspan=2, sticky="we", pady=(8, 0))
        cols = ("name", "elo", "record", "checkpoint")
        self._league_tree = ttk.Treeview(
            roster_frame, columns=cols, show="headings", height=4)
        for c, w in (("name", 110), ("elo", 50), ("record", 70), ("checkpoint", 320)):
            self._league_tree.heading(c, text=c)
            self._league_tree.column(c, width=w)
        self._league_tree.pack(fill=tk.X)

        add_row = ttk.Frame(roster_frame)
        add_row.pack(fill=tk.X, pady=(4, 0))
        self._league_add_name = tk.StringVar(value="")
        self._league_add_ckpt = tk.StringVar(value="")
        ttk.Entry(add_row, textvariable=self._league_add_name, width=14).pack(side=tk.LEFT, padx=2)
        ttk.Entry(add_row, textvariable=self._league_add_ckpt, width=44).pack(side=tk.LEFT, padx=2)
        ttk.Button(add_row, text="Add bot", command=self._league_add_bot).pack(side=tk.LEFT, padx=4)
        ttk.Button(add_row, text="Refresh", command=self._league_refresh_roster).pack(side=tk.LEFT)
        self._league_roster_msg = ttk.Label(roster_frame, text="", foreground="red")
        self._league_roster_msg.pack(fill=tk.X, pady=(2, 0))
        self._league_refresh_roster()

    def _league_roster_path(self) -> Path:
        return launcher.ROOT / "league" / "bots.json"

    def _league_refresh_roster(self) -> None:
        """Re-read the roster JSON into the tree (read-only view)."""
        tree = getattr(self, "_league_tree", None)
        if tree is None:
            return
        tree.delete(*tree.get_children())
        try:
            from league.registry import BotRegistry
            bots = BotRegistry(self._league_roster_path()).all()
        except Exception as exc:
            self._league_roster_msg.config(
                text=f"roster unreadable: {type(exc).__name__}: {exc}")
            return
        for b in bots:
            tree.insert("", tk.END, values=(
                b.name, b.elo, f"{b.wins}W/{b.losses}L/{b.draws}D", b.ckpt_path))

    def _league_add_bot(self) -> None:
        """Add name + checkpoint to the roster (the persistent league state).
        The path is normalized relative to the project root — the league
        child runs with cwd=ROOT, so a relative path is the portable form."""
        name = self._league_add_name.get().strip()
        ckpt = self._league_add_ckpt.get().strip()
        if not name or not ckpt:
            self._league_roster_msg.config(text="both name and checkpoint are required")
            return
        try:
            from league.registry import BotRegistry
            rel = ckpt
            try:
                rel = os.path.relpath(Path(ckpt).resolve(), launcher.ROOT)
            except Exception:
                pass  # keep as typed
            reg = BotRegistry(self._league_roster_path())
            reg.add(name, rel)
            reg.save()
            self._league_add_name.set("")
            self._league_add_ckpt.set("")
            self._league_roster_msg.config(text="", foreground="red")
            self._league_refresh_roster()
        except Exception as exc:
            self._league_roster_msg.config(
                text=f"add failed: {type(exc).__name__}: {exc}", foreground="red")

    # ── replays tab ────────────────────────────────────────────────────────

    def _build_replays_tab(self, parent: ttk.Notebook) -> None:
        frame = ttk.Frame(parent, padding=8)
        parent.add(frame, text="Replay library")

        # summary
        self._replay_summary_label = ttk.Label(frame, text="Loading…")
        self._replay_summary_label.pack(fill=tk.X, pady=2)

        # controls (vars already created in build())
        ctrl = ttk.Frame(frame)
        ctrl.pack(fill=tk.X, pady=4)
        ttk.Label(ctrl, text="Show:").pack(side=tk.LEFT)
        ttk.Radiobutton(ctrl, text="Last N", variable=self._replay_order,
                        value="last").pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(ctrl, text="Top-ELO N", variable=self._replay_order,
                        value="top").pack(side=tk.LEFT, padx=4)
        ttk.Label(ctrl, text="N:").pack(side=tk.LEFT, padx=(8, 2))
        ttk.Spinbox(ctrl, from_=1, to=500, textvariable=self._replay_n,
                    width=5).pack(side=tk.LEFT)
        ttk.Button(ctrl, text="Refresh", command=self._refresh_replays).pack(side=tk.LEFT, padx=8)

        # battle list (read-only)
        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        cols = ("id", "format", "elo", "ts")
        self._replay_tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=12)
        for c, w in (("id", 200), ("format", 80), ("elo", 60), ("ts", 140)):
            self._replay_tree.heading(c, text=c)
            self._replay_tree.column(c, width=w)
        vsb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._replay_tree.yview)
        self._replay_tree.configure(yscrollcommand=vsb.set)
        self._replay_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        # D101 — open the selected replay in the browser (Chrome)
        view_row = ttk.Frame(frame)
        view_row.pack(fill=tk.X, pady=6)
        self._replay_open_btn = ttk.Button(view_row, text="Open in browser (Chrome)",
                                           command=self._open_replay_in_browser)
        self._replay_open_btn.pack(side=tk.LEFT)
        self._replay_status_label = ttk.Label(view_row, text="")
        self._replay_status_label.pack(side=tk.LEFT, padx=10)
        ttk.Label(frame, foreground="#666666",
                  text="Files land in runs/replays_view/. "
                      "Offline: drag the .html onto the Showdown client window.") \
            .pack(anchor="w")

    def _open_replay_in_browser(self) -> None:
        """D101 — export the selected replay as a client-accepted .html file
        and open it in the default browser (Chrome).

        Thread model (tkinter is NOT thread-safe — widget access happens ONLY
        in the main loop): the worker thread does the IO and puts a status
        message on a queue; :meth:`_pump_replay_msgs` (scheduled via
        ``root.after`` from the MAIN thread) drains it.
        """
        if self._replay_worker is not None and self._replay_worker.is_alive():
            self._replay_status_label.config(text="Still preparing… (one at a time)")
            return
        sel = self._replay_tree.selection()
        if not sel:
            self._replay_status_label.config(text="Select a replay row first.")
            return
        row_id = str(self._replay_tree.item(sel[0], "values")[0])
        self._replay_status_label.config(text=f"Preparing {row_id}…")

        def _work() -> None:
            try:
                from replays import ReplayLibrary, open_replay_in_browser
                rec = ReplayLibrary().get_by_id(row_id)
                if rec is None:
                    self._replay_msg_q.put(f"Replay {row_id} not found in the library.")
                    return
                path = open_replay_in_browser(rec)
                self._replay_msg_q.put(f"Opened: {path}")
            except Exception as exc:   # surface ANY failure in the status label
                self._replay_msg_q.put(f"Error: {exc}")

        self._replay_worker = threading.Thread(target=_work, daemon=True)
        self._replay_worker.start()
        self._pump_replay_msgs()

    def _pump_replay_msgs(self) -> None:
        """Main-loop drain of the replay worker's message queue (all tkinter
        calls happen here, on the main thread). Reschedules until the worker
        has delivered and the queue is empty."""
        try:
            while True:
                self._replay_status_label.config(text=self._replay_msg_q.get_nowait())
        except queue.Empty:
            pass
        except tk.TclError:
            return   # window closed mid-export: stop the pump (no reschedule)
        w = self._replay_worker
        if w is not None and (w.is_alive() or not self._replay_msg_q.empty()):
            self.root.after(50, self._pump_replay_msgs)

    def _refresh_replays(self) -> None:
        """Fetch the replay library summary + a bounded list."""
        try:
            from replays import ReplayLibrary
            lib = ReplayLibrary()
            self._replay_summary = lib.summary()
            order = self._replay_order.get()
            n = max(1, self._replay_n.get())
            self._replay_battles = lib.list_battles(n=n, order=order)
        except Exception as exc:
            self._replay_summary_label.config(text=f"Error: {exc}")
            return

        s = self._replay_summary
        self._replay_summary_label.config(
            text=f"Total: {s.get('total', 0)} battles  |  "
                 f"Formats: {s.get('formats', {})}")
        # update the tree
        self._replay_tree.delete(*self._replay_tree.get_children())
        for b in self._replay_battles:
            self._replay_tree.insert("", tk.END, values=(
                b.get("id", "?"), b.get("format", "?"),
                b.get("elo", ""), b.get("ts", "")))

    # ── checkpoints tab ────────────────────────────────────────────────────

    def _build_checkpoints_tab(self, parent: ttk.Notebook) -> None:
        frame = ttk.Frame(parent, padding=8)
        parent.add(frame, text="Checkpoints")

        ttk.Button(frame, text="Refresh", command=self._refresh_checkpoints).pack(anchor="e", pady=2)

        list_frame = ttk.Frame(frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=4)
        cols = ("name", "type", "n_battles", "top1", "verdict", "size_mb")
        self._ckpt_tree = ttk.Treeview(list_frame, columns=cols, show="headings", height=14)
        for c, w in (("name", 220), ("type", 60), ("n_battles", 70),
                     ("top1", 60), ("verdict", 60), ("size_mb", 60)):
            self._ckpt_tree.heading(c, text=c)
            self._ckpt_tree.column(c, width=w)
        vsb = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self._ckpt_tree.yview)
        self._ckpt_tree.configure(yscrollcommand=vsb.set)
        self._ckpt_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

    def _refresh_checkpoints(self) -> None:
        self._checkpoints = launcher.browse_checkpoints()
        self._ckpt_tree.delete(*self._ckpt_tree.get_children())
        for cp in self._checkpoints:
            self._ckpt_tree.insert("", tk.END, values=(
                cp.get("name", "?"), cp.get("type", "?"),
                cp.get("n_battles", ""), cp.get("top1", ""),
                cp.get("verdict", ""), cp.get("size_mb", ""),
            ))

    # ── log tab ────────────────────────────────────────────────────────────

    def _build_log_tab(self, parent: ttk.Notebook) -> None:
        frame = ttk.Frame(parent, padding=8)
        parent.add(frame, text="Log")

        self._log_text_widget = tk.Text(frame, wrap=tk.WORD, state=tk.DISABLED,
                                       font=("Consolas", 9))
        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self._log_text_widget.yview)
        self._log_text_widget.configure(yscrollcommand=vsb.set)
        self._log_text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)

        ttk.Button(frame, text="Refresh log", command=self._refresh_log).pack(anchor="e", pady=2)

    def _refresh_log(self) -> None:
        """Tail the active run's trainer.log (last 200 lines)."""
        st = launcher.status(runs_dir=self._runs_dir)
        run_id = st.get("run_id")
        if not run_id:
            self._set_log_text("No run found.")
            return
        log_path = self._runs_dir / run_id / "trainer.log"
        if not log_path.is_file():
            self._set_log_text(f"Run {run_id} has no log yet.")
            return
        try:
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-200:])
            self._set_log_text(tail or "(empty log)")
        except Exception as exc:
            self._set_log_text(f"Error reading log: {exc}")

    def _set_log_text(self, text: str) -> None:
        self._log_text_widget.config(state=tk.NORMAL)
        self._log_text_widget.delete("1.0", tk.END)
        self._log_text_widget.insert(tk.END, text)
        self._log_text_widget.see(tk.END)
        self._log_text_widget.config(state=tk.DISABLED)

    # ── bottom bar (stop) ──────────────────────────────────────────────────

    def _build_bottom_bar(self) -> None:
        bar = ttk.Frame(self.root, padding=6)
        bar.pack(fill=tk.X, side=tk.BOTTOM)

        self._stop_btn = ttk.Button(bar, text="■  Stop (graceful)",
                                    command=lambda: self._on_stop(force=False))
        self._stop_btn.pack(side=tk.LEFT)
        self._stop_force_btn = ttk.Button(bar, text="■  Stop (force)",
                                         command=lambda: self._on_stop(force=True))
        self._stop_force_btn.pack(side=tk.LEFT, padx=8)
        self._stop_msg = ttk.Label(bar, text="")
        self._stop_msg.pack(side=tk.LEFT, padx=8)

    # ── actions ────────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        """Build the config dict from the current mode's fields and call
        ``launcher.start_trainer``.

        The config is built in the MAIN thread (tkinter variables are not
        thread-safe — reading ``.get()`` from a bg thread is undefined
        behaviour). The ``start_trainer`` call (validate + spawn) runs in a
        background thread (it can take a moment)."""
        self._start_msg.config(text="Starting…")
        self._start_btn.config(state=tk.DISABLED)

        # build the config in the MAIN thread (safe tkinter var access)
        try:
            cfg = self._build_config()
        except Exception as exc:
            self._start_done(False, str(exc))
            return

        def _do():
            try:
                info = launcher.start_trainer(cfg, runs_dir=self._runs_dir)
                self.root.after(0, lambda: self._start_done(True,
                    f"Started {info['run_id']} (pid {info['pid']})"))
            except Exception as exc:
                self.root.after(0, lambda: self._start_done(False, str(exc)))

        threading.Thread(target=_do, daemon=True).start()

    def _start_done(self, ok: bool, msg: str) -> None:
        self._start_btn.config(state=tk.NORMAL)
        self._start_msg.config(text=msg, foreground="green" if ok else "red")

    def _build_config(self) -> Dict[str, Any]:
        """Read the current mode's fields into a config dict."""
        mode = getattr(self, "_current_mode", "selfplay-collect")
        if mode == "selfplay-collect":
            return {
                "mode": mode,
                "n_battles": self._collect_n_battles.get(),
                "battle_format": self._collect_fmt.get(),
                "n_servers": self._collect_n_servers.get(),
                "per_server": self._collect_per_server.get(),
                "port_base": self._collect_port_base.get(),
                "seed": self._collect_seed.get(),
                "max_steps": self._collect_max_steps.get(),
            }
        elif mode == "selfplay":
            return {
                "mode": mode,
                "battle_format": self._ppo_fmt.get(),
                "battles_per_round": self._ppo_battles.get(),
                "n_rounds": self._ppo_rounds.get(),
                "epochs": self._ppo_epochs.get(),
                "lr": self._ppo_lr.get(),
                "device": self._ppo_device.get(),
                # continue training from an existing checkpoint (None = from scratch)
                "resume_from": (self._resolve_ckpt(self._ppo_resume) or None),
            }
        elif mode == "bc":
            return {
                "mode": mode,
                "fixture": self._bc_fixture.get(),
                "n_battles": self._bc_n.get(),
                "max_decisions": 40,
                "heldout_frac": 0.5,
                "epochs": self._bc_epochs.get(),
                "batch_size": self._bc_batch.get(),
                "lr": self._bc_lr.get(),
                "device": self._bc_device.get(),
                "seed": self._bc_seed.get(),
                # fine-tune from an existing checkpoint (None = random init)
                "initial_ckpt": (self._resolve_ckpt(self._bc_resume) or None),
            }
        elif mode == "league":
            return {
                "mode": mode,
                "battle_format": self._league_fmt.get(),
                "pairing": self._league_pairing.get(),
                "best_of": self._league_best_of.get(),
                "greedy": self._league_greedy.get(),
                "device": self._league_device.get(),
                "n_servers": self._league_n_servers.get(),
                "per_server": self._league_per_server.get(),
                "port_base": self._league_port_base.get(),
                "max_steps": self._league_max_steps.get(),
                "seed": self._league_seed.get(),
            }
        elif mode == "play":
            return {
                "mode": mode,
                "ckpt_path": self._resolve_ckpt(self._play_ckpt),
                "n_battles": 0,
                "device": self._play_device.get(),
                "greedy": self._play_greedy.get(),
                "battle_format": self._play_fmt.get(),
                "bot_user": self._play_bot_user.get(),
                "you": self._play_you.get(),
            }
        elif mode == "eval":
            return {
                "mode": mode,
                "ckpt_path": self._eval_ckpt.get(),
                "n_battles": self._eval_n.get(),
                "device": self._eval_device.get(),
                "seed": self._eval_seed.get(),
                "battle_format": self._eval_fmt.get(),
            }
        raise ValueError(f"unknown mode {mode!r}")

    def _on_stop(self, force: bool) -> None:
        """Stop the active run (graceful or force). Runs in a background
        thread (graceful stop can wait up to 180s)."""
        self._stop_msg.config(text="Stopping…")
        self._stop_btn.config(state=tk.DISABLED)
        self._stop_force_btn.config(state=tk.DISABLED)

        def _do():
            try:
                info = launcher.stop_trainer(runs_dir=self._runs_dir, force=force)
                self.root.after(0, lambda: self._stop_done(True,
                    f"Stopped {info.get('run_id', '?')}"))
            except Exception as exc:
                self.root.after(0, lambda: self._stop_done(False, str(exc)))

        threading.Thread(target=_do, daemon=True).start()

    def _stop_done(self, ok: bool, msg: str) -> None:
        self._stop_btn.config(state=tk.NORMAL)
        self._stop_force_btn.config(state=tk.NORMAL)
        self._stop_msg.config(text=msg, foreground="green" if ok else "red")

    # ── poll loop ──────────────────────────────────────────────────────────

    def _poll(self) -> None:
        """Refresh the status bar. Called every POLL_MS via root.after()."""
        try:
            st = launcher.status(runs_dir=self._runs_dir)
        except Exception:
            st = {}

        state = st.get("state") or "idle"
        active = st.get("active", False)
        run_id = st.get("run_id") or ""
        rnd = st.get("round")
        planned = st.get("rounds_planned")
        battles_done = st.get("battles_done")

        status_text = f"Status: {state}"
        if active:
            status_text += "  ●"
        if run_id:
            status_text += f"  ({run_id})"
        if battles_done is not None:
            # self-play / league runs track progress by battles_done (round stays 0)
            status_text += f"  battles {battles_done}" + (f"/{planned}" if planned else "")
        elif rnd is not None and planned is not None:
            status_text += f"  round {rnd}/{planned}"
        self._status_label.config(text=status_text)

        hb = st.get("heartbeat_age_s")
        if hb is not None:
            self._progress_label.config(text=f"heartbeat {hb:.0f}s ago")
        else:
            self._progress_label.config(text="")

        cuda = st.get("cuda_available", False)
        self._cuda_label.config(
            text="CUDA ✓" if cuda else "CPU only")

        # schedule next poll
        if self.root is not None:
            self._poll_id = self.root.after(POLL_MS, self._poll)

    # ── entry point ────────────────────────────────────────────────────────

    def run(self) -> None:
        """Build the UI and start the tkinter main loop."""
        self.build()
        # initial data fetch
        self._refresh_replays()
        self._refresh_checkpoints()
        self._refresh_log()
        self.root.mainloop()


def main() -> None:
    app = DesktopApp()
    app.run()


if __name__ == "__main__":
    main()
