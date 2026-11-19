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

    def __init__(self, headless: bool = False):
        self.headless = headless
        self.root: Optional[tk.Tk] = None
        self._poll_id: Optional[str] = None
        self._runs_dir = launcher.default_runs_dir()

        # per-mode field frames (built lazily by _build_start_tab)
        self._mode_frames: Dict[str, ttk.Frame] = {}

        # replay library cache
        self._replay_summary: Dict[str, Any] = {}
        self._replay_battles: List[Dict[str, Any]] = []

        # checkpoint cache
        self._checkpoints: List[Dict[str, Any]] = []

        # log tail
        self._log_text = ""

        # tkinter variables (created in build(), after the root exists)
        self._mode_var: Optional[tk.StringVar] = None
        self._replay_order: Optional[tk.StringVar] = None
        self._replay_n: Optional[tk.IntVar] = None

    # ── build (create all widgets) ───────────────────────────────────────────

    def build(self) -> None:
        """Create the root window and all widgets. Safe to call headlessly."""
        self.root = tk.Tk()
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
        else:
            raise ValueError(f"unknown widget_type {widget_type!r}")
        w.grid(row=row, column=col + 1, sticky="w", pady=2, padx=(4, 12))
        return var

    def _build_collect_fields(self) -> None:
        r = 0
        self._collect_n_battles = self._add_field(r, 0, "Battles:", 8, "int"); r += 1
        self._collect_fmt = self._add_field(r, 0, "Format:", "gen9ou", "str"); r += 1
        self._collect_n_servers = self._add_field(r, 0, "Servers:", COLLECT_DEFAULTS["n_servers"], "int"); r += 1
        self._collect_per_server = self._add_field(r, 0, "Per server:", COLLECT_DEFAULTS["per_server"], "int"); r += 1
        self._collect_port_base = self._add_field(r, 0, "Port base:", COLLECT_DEFAULTS["port_base"], "int"); r += 1
        self._collect_seed = self._add_field(r, 0, "Seed:", 42, "int"); r += 1
        self._collect_max_steps = self._add_field(r, 0, "Max steps:", 200, "int"); r += 1

    def _build_ppo_fields(self) -> None:
        r = 0
        self._ppo_fmt = self._add_field(r, 0, "Format:", "gen9ou", "str"); r += 1
        self._ppo_battles = self._add_field(r, 0, "Battles/round:", 1, "int"); r += 1
        self._ppo_rounds = self._add_field(r, 0, "Rounds:", 1, "int"); r += 1
        self._ppo_epochs = self._add_field(r, 0, "Epochs:", 1, "int"); r += 1
        self._ppo_lr = self._add_field(r, 0, "LR:", 3e-4, "float"); r += 1
        self._ppo_device = self._add_field(r, 0, "Device:", "cpu", "str"); r += 1

    def _build_bc_fields(self) -> None:
        r = 0
        self._bc_fixture = self._add_field(r, 0, "Fixture path:", "bot/data/replays/gen9ou.jsonl", "str", width=40); r += 1
        self._bc_n = self._add_field(r, 0, "N battles:", 50, "int"); r += 1
        self._bc_epochs = self._add_field(r, 0, "Epochs:", 1, "int"); r += 1
        self._bc_batch = self._add_field(r, 0, "Batch size:", 8, "int"); r += 1
        self._bc_lr = self._add_field(r, 0, "LR:", 3e-3, "float"); r += 1
        self._bc_seed = self._add_field(r, 0, "Seed:", 42, "int"); r += 1
        self._bc_device = self._add_field(r, 0, "Device:", "cpu", "str"); r += 1

    def _build_play_fields(self) -> None:
        r = 0
        self._play_ckpt = self._add_field(r, 0, "Checkpoint:", "", "str", width=40); r += 1
        self._play_fmt = self._add_field(r, 0, "Format:", "gen9ou", "str"); r += 1
        self._play_bot_user = self._add_field(r, 0, "Bot user:", "checkpointbot", "str"); r += 1
        self._play_you = self._add_field(r, 0, "You:", "ttvJustinCase", "str"); r += 1
        self._play_device = self._add_field(r, 0, "Device:", "cpu", "str"); r += 1
        self._play_greedy = self._add_field(r, 0, "Greedy:", True, "bool"); r += 1

    def _build_eval_fields(self) -> None:
        r = 0
        self._eval_ckpt = self._add_field(r, 0, "Checkpoint:", "", "str", width=40); r += 1
        self._eval_n = self._add_field(r, 0, "N battles:", 10, "int"); r += 1
        self._eval_fmt = self._add_field(r, 0, "Format:", "gen9ou", "str"); r += 1
        self._eval_seed = self._add_field(r, 0, "Seed:", 42, "int"); r += 1
        self._eval_device = self._add_field(r, 0, "Device:", "cpu", "str"); r += 1

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
            }
        elif mode == "play":
            return {
                "mode": mode,
                "ckpt_path": self._play_ckpt.get(),
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

        status_text = f"Status: {state}"
        if active:
            status_text += "  ●"
        if run_id:
            status_text += f"  ({run_id})"
        if rnd is not None and planned is not None:
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
