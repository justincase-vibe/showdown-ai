#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Showdown AI - Training Launcher GUI.

Launch servers, configure the training run, and start it -- all from one window.
Uses tkinter (built-in, no extra install).

Usage:
    python bot/launcher.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, simpledialog

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PS_DIR = os.path.join(ROOT, "pokemon-showdown")
PYTHON = sys.executable
CKPT_DIR = os.path.join(HERE, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "ppo.pt")
SNAP_DIR = os.path.join(CKPT_DIR, "snapshots")

# Every tier the bot can train on. OU/RU/UU for gens 6-9, plus Random Battle for
# gens 6-9 (server-generated teams, so no Smogon pool needed for those).
ALL_FORMATS = [
    "gen6ou", "gen6ru", "gen6uu",
    "gen7ou", "gen7ru", "gen7uu",
    "gen8ou", "gen8ru", "gen8uu",
    "gen9ou", "gen9ru", "gen9uu",
    "gen6randombattle", "gen7randombattle", "gen8randombattle", "gen9randombattle",
]


def _apportion_preview(total: int, weights):
    """MIRROR of train.apportion -- used only to preview env allocation in the GUI.
    Keep in sync with bot/train.py::apportion (same proportional + largest-remainder logic)."""
    s = sum(weights) or 1.0
    raw = [total * w / s for w in weights]
    counts = [int(r) for r in raw]
    rem = total - sum(counts)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - counts[i], reverse=True)
    for i in range(rem):
        counts[order[i % len(order)]] += 1
    return counts

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "num_servers": 10,
    "base_port": 8000,
    "num_envs": 16,
    "opponent": "live",
    "buffer_size": 2048,
    "minibatch": 512,
    "ppo_epochs": 6,
    "lr": "1e-4",
    "max_battles": 5000,
    "device": "cuda",
    "hidden": "1024,2048,4096,4096,2048,1024",
    "team_mode": "mixed",
    "p_strong": "0.5",
    "log_every": 50,
    "seed": 0,
    "degen_sw_per_turn": "0.33",
}


# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------
def server_alive(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_servers(n: int, base_port: int) -> list[int]:
    """Launch n Showdown servers. Returns list of ports started."""
    ports = []
    for i in range(n):
        port = base_port + i
        if server_alive(port):
            ports.append(port)
            continue
        env = os.environ.copy()
        env["SHOWDOWN_PORT"] = str(port)
        subprocess.Popen(
            ["node", "pokemon-showdown"],
            cwd=PS_DIR,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == "nt" else 0,
        )
        ports.append(port)
    return ports


def wait_for_servers(ports: list[int], timeout: int = 45) -> list[int]:
    """Wait until all ports respond. Returns list of ports that are up."""
    deadline = time.time() + timeout
    up = []
    while time.time() < deadline:
        up = [p for p in ports if server_alive(p)]
        if len(up) == len(ports):
            return up
        time.sleep(1)
    return up


def kill_servers(ports: list[int]) -> None:
    """Kill node processes listening on the given ports."""
    for port in ports:
        # Try fuser first (Git Bash / Linux)
        try:
            subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        # Windows fallback: find PID via netstat
        try:
            out = subprocess.check_output(["netstat", "-aon"], text=True, timeout=5)
            for line in out.splitlines():
                if f":{port} " in line and "LISTENING" in line:
                    pid = line.split()[-1]
                    subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------
class LauncherApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Showdown AI - Training Launcher")
        root.geometry("920x900")
        root.minsize(840, 740)

        self.train_proc: subprocess.Popen | None = None
        self._progress_top = None

        self._build_ui()

    def _build_ui(self):
        # --- Top frame: server controls ---
        srv_frame = ttk.LabelFrame(self.root, text="Servers")
        srv_frame.pack(fill="x", padx=10, pady=5)

        srv_row = ttk.Frame(srv_frame)
        srv_row.pack(fill="x", padx=5, pady=3)

        ttk.Label(srv_row, text="Count:").pack(side="left")
        self.var_servers = tk.IntVar(value=DEFAULTS["num_servers"])
        ttk.Spinbox(srv_row, from_=1, to=25, textvariable=self.var_servers, width=4).pack(side="left", padx=3)

        ttk.Label(srv_row, text="Base port:").pack(side="left", padx=(10, 0))
        self.var_baseport = tk.IntVar(value=DEFAULTS["base_port"])
        ttk.Spinbox(srv_row, from_=7000, to=9999, textvariable=self.var_baseport, width=6).pack(side="left", padx=3)

        self.btn_start_srv = ttk.Button(srv_row, text="Start Servers", command=self._start_servers)
        self.btn_start_srv.pack(side="left", padx=10)
        self.btn_kill_srv = ttk.Button(srv_row, text="Kill Servers", command=self._kill_servers)
        self.btn_kill_srv.pack(side="left", padx=3)
        self.btn_check_srv = ttk.Button(srv_row, text="Check", command=self._check_servers)
        self.btn_check_srv.pack(side="left", padx=3)

        self.lbl_server_status = ttk.Label(srv_frame, text="[ ] No servers checked", foreground="gray")
        self.lbl_server_status.pack(anchor="w", padx=5, pady=2)

        # --- Formats & weights (crossable list) ---
        fmt_frame = ttk.LabelFrame(
            self.root, text="Formats   (tick to include · W = relative priority among the ticked)"
        )
        fmt_frame.pack(fill="x", padx=10, pady=5)
        self.fmt_vars: dict[str, tk.BooleanVar] = {}
        self.fmt_weights: dict[str, tk.StringVar] = {}
        for i, fmt in enumerate(ALL_FORMATS):
            cell = ttk.Frame(fmt_frame)
            cell.grid(row=(i // 2), column=(i % 2), sticky="w", padx=12, pady=1)
            self.fmt_vars[fmt] = tk.BooleanVar(value=True)  # default: all on
            ttk.Checkbutton(cell, text=fmt, variable=self.fmt_vars[fmt],
                            command=self._update_env_preview,
                            width=18).pack(side="left")
            ttk.Label(cell, text="  W:").pack(side="left")
            self.fmt_weights[fmt] = tk.StringVar(value="1")
            ent = ttk.Entry(cell, textvariable=self.fmt_weights[fmt], width=4)
            ent.pack(side="left", padx=2)
            ent.bind("<KeyRelease>", lambda _e: self._update_env_preview())
            ent.bind("<FocusOut>", lambda _e: self._update_env_preview())
        self.lbl_env_preview = ttk.Label(fmt_frame, text="", foreground="gray")
        self.lbl_env_preview.grid(row=(len(ALL_FORMATS) // 2), column=0, columnspan=2,
                                  sticky="w", padx=12, pady=3)

        # --- Middle frame: training params (collapsible) ---
        trn_outer = ttk.Frame(self.root)
        trn_outer.pack(fill="x", padx=10, pady=5)
        self._trn_expanded = True
        self.btn_trn_toggle = ttk.Button(trn_outer, text="▼ Training Parameters", command=self._toggle_train_params)
        self.btn_trn_toggle.pack(fill="x")
        self.trn_frame = ttk.LabelFrame(trn_outer, text="")
        self.trn_frame.pack(fill="both", expand=True, pady=(2, 0))
        trn_frame = self.trn_frame

        params = [
            ("Num envs", "num_envs", 0),
            ("Max battles", "max_battles", 1),
            ("Opponent", "opponent", 2),
            ("Buffer size", "buffer_size", 3),
            ("Minibatch", "minibatch", 4),
            ("PPO epochs", "ppo_epochs", 5),
            ("Learning rate", "lr", 6),
            ("Device", "device", 7),
            ("Hidden layers", "hidden", 8),
            ("Team mode", "team_mode", 9),
            ("Strong ratio (0-1)", "p_strong", 10),
            ("Log every", "log_every", 11),
            ("Seed", "seed", 12),
            ("Degen sw/turn", "degen_sw_per_turn", 13),
        ]

        self.vars: dict[str, tk.StringVar] = {}
        for label, key, row in params:
            ttk.Label(trn_frame, text=label + ":").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value=str(DEFAULTS[key]))
            self.vars[key] = var
            w = 40 if key == "hidden" else 14
            ent = ttk.Entry(trn_frame, textvariable=var, width=w)
            ent.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            if key == "num_envs":
                ent.bind("<KeyRelease>", lambda _e: self._update_env_preview())
                ent.bind("<FocusOut>", lambda _e: self._update_env_preview())

        trn_frame.columnconfigure(1, weight=1)

        self.var_fresh = tk.BooleanVar(value=False)  # default = CONTINUE from existing checkpoint
        ttk.Checkbutton(
            trn_frame,
            text="FRESH net (discard the existing checkpoint and train from scratch)",
            variable=self.var_fresh,
        ).grid(row=14, column=0, columnspan=2, sticky="w", padx=5, pady=4)

        self._update_env_preview()  # set the initial allocation preview

        # --- Bottom: action buttons ---
        act_frame = ttk.Frame(self.root)
        act_frame.pack(fill="x", padx=10, pady=5)

        self.btn_train = ttk.Button(act_frame, text="Start Training", command=self._start_training)
        self.btn_train.pack(side="left", padx=5)
        self.btn_stop = ttk.Button(act_frame, text="Stop", command=self._stop_training, state="disabled")
        self.btn_stop.pack(side="left", padx=5)
        self.btn_eval = ttk.Button(act_frame, text="Eval (5 battles)", command=self._run_eval)
        self.btn_eval.pack(side="left", padx=5)
        ttk.Label(act_frame, text="Replays:").pack(side="left", padx=(10, 0))
        self.var_replay_count = tk.IntVar(value=5)
        ttk.Spinbox(act_frame, from_=1, to=100, textvariable=self.var_replay_count, width=3).pack(side="left")
        self.btn_replays = ttk.Button(act_frame, text="Open Replays", command=self._open_replays)
        self.btn_replays.pack(side="left", padx=5)
        self.btn_progress = ttk.Button(act_frame, text="Progress", command=self._open_progress)
        self.btn_progress.pack(side="left", padx=5)

        self.lbl_status = ttk.Label(act_frame, text="Ready", foreground="blue")
        self.lbl_status.pack(side="right", padx=10)

        # --- Snapshot management (save the current net, compare, or fall back) ---
        snap_frame = ttk.LabelFrame(self.root, text="Net snapshots (save / compare / fall back)")
        snap_frame.pack(fill="x", padx=10, pady=5)
        ttk.Button(snap_frame, text="Save snapshot", command=self._save_snapshot).pack(side="left", padx=5)
        ttk.Button(snap_frame, text="Restore snapshot...", command=self._restore_snapshot).pack(side="left", padx=5)
        ttk.Button(snap_frame, text="Net info", command=self._net_info).pack(side="left", padx=5)
        ttk.Button(snap_frame, text="Compare vs snapshot...", command=self._compare_net).pack(side="left", padx=5)
        ttk.Label(snap_frame, text="  saved to: checkpoints/snapshots/", foreground="gray").pack(side="right", padx=10)

        # --- Log area ---
        log_frame = ttk.LabelFrame(self.root, text="Output")
        log_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.log = scrolledtext.ScrolledText(log_frame, height=12, font=("Consolas", 9), state="disabled")
        self.log.pack(fill="both", expand=True)

    def _log(self, msg: str):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")
        self.root.update_idletasks()

    # --- Net snapshots (save / restore / inspect / compare) -----------------
    @staticmethod
    def _list_snapshots() -> list[str]:
        """Snapshot files, newest first."""
        if not os.path.isdir(SNAP_DIR):
            return []
        files = [f for f in os.listdir(SNAP_DIR) if f.endswith(".pt")]
        files.sort(key=lambda f: os.path.getmtime(os.path.join(SNAP_DIR, f)), reverse=True)
        return files

    def _pick_snapshot(self, title: str, prompt: str) -> "str | None":
        """Modal picker over the snapshot folder. Returns the chosen filename or None."""
        snaps = self._list_snapshots()
        if not snaps:
            messagebox.showinfo("No snapshots", "No snapshots in checkpoints/snapshots/ yet.\nUse 'Save snapshot' first.", parent=self.root)
            return None
        top = tk.Toplevel(self.root)
        top.title(title)
        top.transient(self.root)
        top.grab_set()
        ttk.Label(top, text=prompt).pack(padx=12, pady=(10, 4))
        lb = tk.Listbox(top, height=min(12, len(snaps)), width=48, selectmode="single", font=("Consolas", 9))
        for f in snaps:
            lb.insert("end", f)
        lb.selection_set(0)
        lb.activate(0)
        lb.pack(padx=12, pady=4)
        state = {"ok": False}

        def ok():
            if lb.curselection():
                state["ok"] = True
            top.destroy()

        def cancel():
            top.destroy()

        btns = ttk.Frame(top)
        btns.pack(pady=8)
        ttk.Button(btns, text="OK", command=ok).pack(side="left", padx=4)
        ttk.Button(btns, text="Cancel", command=cancel).pack(side="left", padx=4)
        top.wait_window()
        if not state["ok"]:
            return None
        return snaps[lb.curselection()[0]]

    def _save_snapshot(self):
        """Copy the current net to checkpoints/snapshots/<name>.pt (original untouched)."""
        if not os.path.isfile(CKPT_PATH):
            messagebox.showwarning("No checkpoint", "No ppo.pt found. Train first.", parent=self.root)
            return
        default_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = simpledialog.askstring("Save snapshot", "Snapshot name (no extension):", initialvalue=default_name, parent=self.root)
        if name is None:
            return
        name = name.strip().replace("\\", "_").replace("/", "_").replace("..", "_")
        if not name:
            return
        os.makedirs(SNAP_DIR, exist_ok=True)
        dest = os.path.join(SNAP_DIR, name + ".pt")
        if os.path.isfile(dest) and not messagebox.askyesno("Overwrite", f"Snapshot '{name}.pt' already exists. Overwrite?", parent=self.root):
            return
        self.lbl_status.config(text="Saving snapshot...", foreground="orange")

        def work():
            try:
                shutil.copy2(CKPT_PATH, dest)
            except Exception as exc:
                self.root.after(0, lambda: self._log(f"!! Snapshot failed (is a save in progress?): {exc}"))
                self.root.after(0, lambda: self.lbl_status.config(text="Snapshot failed", foreground="red"))
                return
            self.root.after(0, lambda: self._log(f"Snapshot saved: {name}.pt   (original ppo.pt untouched)"))
            self.root.after(0, lambda: self.lbl_status.config(text=f"Snapshot saved: {name}", foreground="green"))

        threading.Thread(target=work, daemon=True).start()

    def _restore_snapshot(self):
        """Fall back: copy a snapshot over ppo.pt (auto-backing up the current net first)."""
        pick = self._pick_snapshot("Restore snapshot", "Pick a snapshot to restore into ppo.pt (the current net is auto-backed-up first):")
        if not pick:
            return
        src = os.path.join(SNAP_DIR, pick)
        if not messagebox.askyesno(
            "Restore snapshot",
            f"Replace the current net (ppo.pt) with:\n  {pick}\n\nThe current ppo.pt will be auto-backed-up to snapshots/ first.\nContinue?",
            parent=self.root,
        ):
            return
        self.lbl_status.config(text="Restoring...", foreground="orange")

        def work():
            try:
                os.makedirs(SNAP_DIR, exist_ok=True)
                backup = os.path.join(SNAP_DIR, "pre_restore_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pt")
                shutil.copy2(CKPT_PATH, backup)
                shutil.copy2(src, CKPT_PATH)
            except Exception as exc:
                self.root.after(0, lambda: self._log(f"!! Restore failed (is a save in progress?): {exc}"))
                self.root.after(0, lambda: self.lbl_status.config(text="Restore failed", foreground="red"))
                return
            self.root.after(0, lambda: self._log(f"Restored {pick} -> ppo.pt   (previous net backed up as {os.path.basename(backup)})"))
            self.root.after(0, lambda: self.lbl_status.config(text=f"Restored: {pick}", foreground="green"))

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _net_metrics(ck: dict) -> dict:
        """Global parameter stats over actor + critic (CPU tensors)."""
        import torch
        flat = []
        for key in ("actor", "critic"):
            sd = ck.get(key) or {}
            for v in sd.values():
                if isinstance(v, torch.Tensor):
                    flat.append(v.detach().float())
        if not flat:
            return {}
        cat = torch.cat([p.reshape(-1) for p in flat])
        return {
            "params": int(cat.numel()),
            "mean": float(cat.mean()),
            "std": float(cat.std()),
            "norm": float(torch.linalg.norm(cat)),
            "n_updates": ck.get("n_updates", "n/a"),
        }

    @staticmethod
    def _fmt_metrics(label: str, m: dict) -> str:
        if not m:
            return f"[{label}]  (no parameters found)"
        return (f"[{label}]  n_updates={m['n_updates']}  params={m['params']:,}  "
                f"mean={m['mean']:.5f}  std={m['std']:.5f}  ||w||={m['norm']:.4f}")

    def _net_info(self):
        if not os.path.isfile(CKPT_PATH):
            messagebox.showwarning("No checkpoint", "No ppo.pt found. Train first.", parent=self.root)
            return
        self.lbl_status.config(text="Reading net...", foreground="orange")

        def work():
            import torch
            ck = torch.load(CKPT_PATH, map_location="cpu")
            txt = self._fmt_metrics("ppo.pt", self._net_metrics(ck))
            self.root.after(0, lambda: self._log(txt))
            self.root.after(0, lambda: self.lbl_status.config(text="Done", foreground="blue"))

        threading.Thread(target=work, daemon=True).start()

    @staticmethod
    def _report_compare(name: str, cur: dict, old: dict) -> str:
        import torch

        def tensors(ck: dict) -> dict:
            out = {}
            for key in ("actor", "critic"):
                for n, v in (ck.get(key) or {}).items():
                    if isinstance(v, torch.Tensor):
                        out[f"{key}.{n}"] = v.detach().float()
            return out

        c, o = tensors(cur), tensors(old)
        common = sorted(set(c) & set(o))
        if not common:
            return f"[compare ppo.pt vs {name}]  no common parameters found (shape mismatch?)"
        diffs = {t: float(torch.linalg.norm(c[t] - o[t])) for t in common}
        total = float(sum(d * d for d in diffs.values()) ** 0.5)
        cu, ou = cur.get("n_updates", "n/a"), old.get("n_updates", "n/a")
        delta = (cu - ou) if isinstance(cu, int) and isinstance(ou, int) else "n/a"
        top = sorted(diffs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        lines = [
            f"[compare  ppo.pt  vs  {name}]",
            f"  n_updates: {ou}  ->  {cu}   (delta {delta})",
            f"  total parameter L2 change: {total:.6f}   across {len(common)} tensors",
            "  top-3 most-changed layers:",
        ]
        for t, d in top:
            lines.append(f"    {t}: {d:.6f}")
        if set(c) - set(o):
            lines.append(f"  (note: {len(set(c) - set(o))} tensor(s) only in current net)")
        return "\n".join(lines)

    def _compare_net(self):
        if not os.path.isfile(CKPT_PATH):
            messagebox.showwarning("No checkpoint", "No ppo.pt found. Train first.", parent=self.root)
            return
        pick = self._pick_snapshot("Compare net vs snapshot", "Pick a snapshot to compare the CURRENT net (ppo.pt) against:")
        if not pick:
            return
        self.lbl_status.config(text="Comparing...", foreground="orange")

        def work():
            import torch
            cur = torch.load(CKPT_PATH, map_location="cpu")
            old = torch.load(os.path.join(SNAP_DIR, pick), map_location="cpu")
            txt = self._report_compare(pick, cur, old)
            self.root.after(0, lambda: self._log(txt))
            self.root.after(0, lambda: self.lbl_status.config(text="Done", foreground="blue"))
            self.root.after(0, lambda: messagebox.showinfo("Compare result", txt, parent=self.root))

        threading.Thread(target=work, daemon=True).start()

    # --- Progress window (separate window, keeps the main log clean) ---
    def _open_progress(self):
        def work():
            from progress import render
            text = render(CKPT_DIR, os.path.join(HERE, "run_gpu.log"), 60)
            self.root.after(0, lambda: self._progress_window(text))
        threading.Thread(target=work, daemon=True).start()

    def _progress_window(self, text: str):
        if getattr(self, "_progress_top", None) is not None:
            try:
                self._progress_top.destroy()
            except Exception:
                pass
        top = tk.Toplevel(self.root)
        self._progress_top = top
        top.title("Showdown AI - Progress")
        top.geometry("780x540")
        top.transient(self.root)

        body = tk.Text(top, font=("Consolas", 10), state="disabled", wrap="none",
                       background="#1e1e1e", foreground="#e6e6e6")
        vsb = ttk.Scrollbar(top, command=body.yview)
        body.configure(yscrollcommand=vsb.set)
        body.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        btns = ttk.Frame(top)
        btns.pack(fill="x", pady=(4, 2))

        def refresh():
            def w2():
                from progress import render
                t2 = render(CKPT_DIR, os.path.join(HERE, "run_gpu.log"), 60)
                self.root.after(0, lambda: self._set_progress_text(body, t2))
            threading.Thread(target=w2, daemon=True).start()

        def copy_all():
            try:
                self.root.clipboard_clear()
                self.root.clipboard_append(body.get("1.0", "end-1c"))
            except Exception:
                pass

        ttk.Button(btns, text="Refresh", command=refresh).pack(side="left", padx=5, pady=5)
        ttk.Button(btns, text="Copy", command=copy_all).pack(side="left", padx=5)
        ttk.Button(btns, text="Close", command=top.destroy).pack(side="right", padx=5)

        self._set_progress_text(body, text)

    def _set_progress_text(self, body: tk.Text, text: str):
        body.configure(state="normal")
        body.delete("1.0", "end")
        body.insert("1.0", text)
        body.configure(state="disabled")
        body.see("1.0")

    # --- Server actions ---
    def _start_servers(self):
        n = self.var_servers.get()
        base = self.var_baseport.get()
        ports = list(range(base, base + n))
        self._log(f"Starting {n} servers on ports {ports[0]}-{ports[-1]}...")
        self.lbl_status.config(text="Starting servers...", foreground="orange")

        def work():
            start_servers(n, base)
            up = wait_for_servers(ports, timeout=45)
            self.root.after(0, lambda: self._servers_started(up, ports))

        threading.Thread(target=work, daemon=True).start()

    def _servers_started(self, up: list[int], all_ports: list[int]):
        if len(up) == len(all_ports):
            self.lbl_server_status.config(text=f"[OK] {len(up)}/{len(all_ports)} servers UP", foreground="green")
            self._log(f"All {len(up)} servers are up.")
        else:
            missing = [p for p in all_ports if p not in up]
            self.lbl_server_status.config(
                text=f"[!!] {len(up)}/{len(all_ports)} up | missing: {missing}", foreground="red"
            )
            self._log(f"WARNING: {len(missing)} server(s) failed to start: {missing}")
        self.lbl_status.config(text="Servers ready", foreground="blue")

    def _kill_servers(self):
        n = self.var_servers.get()
        base = self.var_baseport.get()
        ports = list(range(base, base + n))
        self._log(f"Killing servers on ports {ports[0]}-{ports[-1]}...")
        kill_servers(ports)
        time.sleep(1)
        alive = [p for p in ports if server_alive(p)]
        if alive:
            self._log(f"Still alive: {alive}")
            self.lbl_server_status.config(text=f"[!!] {len(alive)} still running", foreground="red")
        else:
            self._log("All servers killed.")
            self.lbl_server_status.config(text="[ ] No servers running", foreground="gray")

    def _check_servers(self):
        n = self.var_servers.get()
        base = self.var_baseport.get()
        ports = list(range(base, base + n))
        up = [p for p in ports if server_alive(p)]
        if up:
            self.lbl_server_status.config(text=f"[OK] {len(up)}/{len(ports)} UP: {up}", foreground="green")
            self._log(f"Servers up: {up}")
        else:
            self.lbl_server_status.config(text=f"[!!] 0/{len(ports)} up", foreground="red")
            self._log("No servers responding.")

    # --- Training ---
    def _selected_formats(self):
        """Return (formats, weights) lists from the crossable list (ticked rows only),
        in ALL_FORMATS order. Weights default to 1 when blank/invalid; 0 keeps the tier
        but gives it no envs (use uncheck to fully exclude)."""
        fmts, wts = [], []
        for fmt in ALL_FORMATS:
            if not self.fmt_vars[fmt].get():
                continue
            raw = self.fmt_weights[fmt].get().strip()
            try:
                w = float(raw) if raw else 1.0
            except ValueError:
                w = 1.0
            fmts.append(fmt)
            wts.append(f"{max(0.0, w):g}")
        return fmts, wts

    def _toggle_train_params(self):
        self._trn_expanded = not self._trn_expanded
        if self._trn_expanded:
            self.btn_trn_toggle.config(text="▼ Training Parameters")
            self.trn_frame.pack(fill="both", expand=True, pady=(2, 0))
        else:
            self.btn_trn_toggle.config(text="▶ Training Parameters")
            self.trn_frame.pack_forget()

    def _update_env_preview(self):
        """Live preview: how Num envs split across the ticked formats; flag any tier
        that would get 0 envs (silent no-train) in red."""
        try:
            total = int(self.vars["num_envs"].get())
        except (ValueError, tk.TclError):
            total = 0
        checked = [f for f in ALL_FORMATS if self.fmt_vars[f].get()]
        if not checked:
            self.lbl_env_preview.config(text="(no formats ticked)", foreground="gray")
            return
        weights = []
        for f in checked:
            raw = self.fmt_weights[f].get().strip()
            try:
                w = float(raw) if raw else 1.0
            except ValueError:
                w = 1.0
            weights.append(max(0.0, w))
        alloc = _apportion_preview(total, weights)
        zero = [f for f, c in zip(checked, alloc) if c == 0]
        base = "envs -> " + "  ".join(f"{f}:{c}" for f, c in zip(checked, alloc))
        if zero:
            self.lbl_env_preview.config(
                text=base + f"   !! 0 envs (NOT training): {', '.join(zero)} -- raise Num envs or lower other W",
                foreground="red")
        else:
            self.lbl_env_preview.config(text=base, foreground="green")

    def _build_train_cmd(self) -> list[str]:
        v = {k: var.get().strip() for k, var in self.vars.items()}
        n = self.var_servers.get()
        base = self.var_baseport.get()
        ports = ",".join(str(base + i) for i in range(n))

        fmts, wts = self._selected_formats()
        if not fmts:
            return []

        cmd = [
            PYTHON, os.path.join(HERE, "train.py"),
            "--formats", ",".join(fmts),
            "--weights", ",".join(wts),
            "--num-envs", v["num_envs"],
            "--opponent", v["opponent"],
            "--buffer-size", v["buffer_size"],
            "--minibatch", v["minibatch"],
            "--ppo-epochs", v["ppo_epochs"],
            "--lr", v["lr"],
            "--max-battles", v["max_battles"],
            "--device", v["device"],
            "--hidden", v["hidden"],
            "--server-ports", ports,
            "--team-mode", v["team_mode"],
            "--p-strong", v["p_strong"],
            "--log-every", v["log_every"],
            "--seed", v["seed"],
            "--degen-sw-per-turn", v["degen_sw_per_turn"],
        ]
        # Continue from the existing checkpoint by default; skip only when starting fresh.
        if not self.var_fresh.get():
            ckpt = os.path.join(HERE, "checkpoints", "ppo.pt")
            if os.path.isfile(ckpt):
                cmd += ["--resume", ckpt]
        return cmd

    def _start_training(self):
        if not self._selected_formats()[0]:
            messagebox.showwarning("No formats", "Tick at least one format to train.", parent=self.root)
            return

        if self.var_fresh.get():
            ckpt = os.path.join(HERE, "checkpoints", "ppo.pt")
            if os.path.isfile(ckpt) and not messagebox.askyesno(
                "Start FRESH net?",
                f"This will DISCARD the existing net at:\n  {ckpt}\nand train from scratch.\n\nAre you sure?",
            ):
                return

        n = self.var_servers.get()
        base = self.var_baseport.get()
        ports = list(range(base, base + n))
        up = [p for p in ports if server_alive(p)]

        if len(up) < len(ports):
            missing = [p for p in ports if p not in up]
            if not messagebox.askyesno(
                "Servers not ready",
                f"Only {len(up)}/{len(ports)} servers are up.\nMissing ports: {missing}\n\nStart anyway?"
            ):
                return

        cmd = self._build_train_cmd()
        self._log(f"\n{'='*60}")
        self._log("LAUNCHING TRAINING")
        self._log(f"{'='*60}")
        self._log(f"  battles: {self.vars['max_battles'].get()}")
        self._log(f"  envs:    {self.vars['num_envs'].get()}")
        fmts, wts = self._selected_formats()
        self._log(f"  formats: {len(fmts)} -> {', '.join(fmts)}")
        self._log(f"  weights: {', '.join(wts)}")
        self._log(f"  device:  {self.vars['device'].get()}")
        self._log(f"  ports:   {ports[0]}-{ports[-1]}")
        self._log(f"{'='*60}\n")

        self.btn_train.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.lbl_status.config(text="Training...", foreground="orange")

        log_file = open(os.path.join(HERE, "run_gpu.log"), "w", encoding="utf-8")

        def run():
            self.train_proc = subprocess.Popen(
                cmd, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            assert self.train_proc.stdout
            for line in self.train_proc.stdout:
                log_file.write(line)
                log_file.flush()
                self.root.after(0, lambda l=line.rstrip(): self._log(l))
            self.train_proc.wait()
            code = self.train_proc.returncode
            log_file.close()
            self.root.after(0, lambda: self._training_done(code))

        threading.Thread(target=run, daemon=True).start()

    def _training_done(self, code: int):
        self.btn_train.config(state="normal")
        self.btn_stop.config(state="disabled")
        if code == 0:
            self.lbl_status.config(text="DONE", foreground="green")
            self._log("\nTraining completed successfully.")
            messagebox.showinfo("Done", "Training run finished!")
        else:
            self.lbl_status.config(text=f"ERROR (exit {code})", foreground="red")
            self._log(f"\nTraining exited with code {code}. Check log above.")
            messagebox.showerror("Error", f"Training exited with code {code}")

    def _stop_training(self):
        if self.train_proc:
            self._log("Stopping training...")
            self.train_proc.terminate()
            try:
                self.train_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.train_proc.kill()
            self.lbl_status.config(text="Stopped", foreground="red")

    def _run_eval(self):
        n = self.var_servers.get()
        base = self.var_baseport.get()
        port = base  # use first server
        if not server_alive(port):
            messagebox.showwarning("No server", f"No server on port {port}. Start one first.")
            return

        ckpt = os.path.join(HERE, "checkpoints", "ppo.pt")
        if not os.path.isfile(ckpt):
            messagebox.showwarning("No checkpoint", "No ppo.pt found. Train first.")
            return

        hidden = self.vars["hidden"].get()
        device = self.vars["device"].get()
        cmd = [
            PYTHON, os.path.join(HERE, "quick_eval.py"),
            "--model", ckpt,
            "--battles", "5",
            "--device", device,
            "--hidden", hidden,
            "--formats", "gen7ou,gen9ou",
            "--server-ports", str(port),
        ]
        self._log(f"\n--- EVAL (5 battles vs random) ---")
        self.lbl_status.config(text="Evaluating...", foreground="orange")

        def run():
            proc = subprocess.Popen(
                cmd, cwd=ROOT,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace",
            )
            assert proc.stdout
            for line in proc.stdout:
                self.root.after(0, lambda l=line.rstrip(): self._log(l))
            proc.wait()
            self.root.after(0, lambda: self._eval_done(proc.returncode))

        threading.Thread(target=run, daemon=True).start()

    def _eval_done(self, code: int):
        self.lbl_status.config(
            text="Eval done" if code == 0 else "Eval error",
            foreground="green" if code == 0 else "red"
        )

    def _open_replays(self):
        import glob
        import re
        import webbrowser
        replays_dir = os.path.join(HERE, "replays")
        if not os.path.isdir(replays_dir):
            messagebox.showwarning("No replays", "Replays folder not found.")
            return
        files = glob.glob(os.path.join(replays_dir, "*.html"))
        if not files:
            messagebox.showwarning("No replays", "No replay files found.")
            return

        def battle_num(path):
            # Trailing "-<number>.html" is the battle id (format names also contain
            # digits, e.g. gen7ou, so match the LAST number before .html only).
            m = re.search(r"-(\d+)\.html$", os.path.basename(path))
            return int(m.group(1)) if m else -1

        # Each battle has 2 files (one per agent). Keep ONE file per battle number
        # (the newest), then order battles by number descending -> distinct battles.
        best = {}
        for f in files:
            bn = battle_num(f)
            if bn < 0:
                continue
            if bn not in best or os.path.getmtime(f) > os.path.getmtime(best[bn]):
                best[bn] = f
        ordered = sorted(best.values(), key=battle_num, reverse=True)

        count = max(1, int(self.var_replay_count.get()))
        pick = ordered[:count]
        self._log(f"\nOpening {len(pick)} most recent DISTINCT replays (of {len(ordered)} battles):")
        for f in pick:
            self._log(f"  {os.path.basename(f)}")
        for f in pick:
            webbrowser.open("file://" + f.replace(os.sep, "/"))
        self._log("Opened in browser.")


# ---------------------------------------------------------------------------
def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
