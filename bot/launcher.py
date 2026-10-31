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
import subprocess
import sys
import threading
import time
import urllib.request

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PS_DIR = os.path.join(ROOT, "pokemon-showdown")
PYTHON = sys.executable

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULTS = {
    "num_servers": 10,
    "base_port": 8000,
    "formats": "gen6ou,gen6ru,gen6uu,gen7ou,gen7ru,gen7uu,gen8ou,gen8ru,gen8uu,gen9ou,gen9ru,gen9uu",
    "num_envs": 10,
    "opponent": "live",
    "buffer_size": 2048,
    "minibatch": 512,
    "ppo_epochs": 6,
    "lr": "1e-4",
    "max_battles": 5000,
    "device": "cuda",
    "hidden": "1024,2048,4096,4096,2048,1024",
    "team_mode": "random",
    "log_every": 50,
    "seed": 0,
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
        root.geometry("820x720")
        root.minsize(700, 600)

        self.train_proc: subprocess.Popen | None = None

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

        # --- Middle frame: training params ---
        trn_frame = ttk.LabelFrame(self.root, text="Training Parameters")
        trn_frame.pack(fill="both", expand=True, padx=10, pady=5)

        params = [
            ("Formats", "formats", 0),
            ("Num envs", "num_envs", 1),
            ("Max battles", "max_battles", 2),
            ("Opponent", "opponent", 3),
            ("Buffer size", "buffer_size", 4),
            ("Minibatch", "minibatch", 5),
            ("PPO epochs", "ppo_epochs", 6),
            ("Learning rate", "lr", 7),
            ("Device", "device", 8),
            ("Hidden layers", "hidden", 9),
            ("Team mode", "team_mode", 10),
            ("Log every", "log_every", 11),
            ("Seed", "seed", 12),
        ]

        self.vars: dict[str, tk.StringVar] = {}
        for label, key, row in params:
            ttk.Label(trn_frame, text=label + ":").grid(row=row, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value=str(DEFAULTS[key]))
            self.vars[key] = var
            w = 70 if key in ("formats", "hidden") else 14
            ttk.Entry(trn_frame, textvariable=var, width=w).grid(row=row, column=1, sticky="ew", padx=5, pady=2)

        trn_frame.columnconfigure(1, weight=1)

        # --- Bottom: action buttons ---
        act_frame = ttk.Frame(self.root)
        act_frame.pack(fill="x", padx=10, pady=5)

        self.btn_train = ttk.Button(act_frame, text="Start Training", command=self._start_training)
        self.btn_train.pack(side="left", padx=5)
        self.btn_stop = ttk.Button(act_frame, text="Stop", command=self._stop_training, state="disabled")
        self.btn_stop.pack(side="left", padx=5)
        self.btn_eval = ttk.Button(act_frame, text="Eval (5 battles)", command=self._run_eval)
        self.btn_eval.pack(side="left", padx=5)
        self.btn_replays = ttk.Button(act_frame, text="Open Last 5 Replays", command=self._open_replays)
        self.btn_replays.pack(side="left", padx=5)

        self.lbl_status = ttk.Label(act_frame, text="Ready", foreground="blue")
        self.lbl_status.pack(side="right", padx=10)

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
    def _build_train_cmd(self) -> list[str]:
        v = {k: var.get().strip() for k, var in self.vars.items()}
        n = self.var_servers.get()
        base = self.var_baseport.get()
        ports = ",".join(str(base + i) for i in range(n))

        cmd = [
            PYTHON, os.path.join(HERE, "train.py"),
            "--formats", v["formats"],
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
            "--log-every", v["log_every"],
            "--seed", v["seed"],
        ]
        return cmd

    def _start_training(self):
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
        self._log(f"  formats: {self.vars['formats'].get()[:60]}...")
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
        replays_dir = os.path.join(HERE, "replays")
        if not os.path.isdir(replays_dir):
            messagebox.showwarning("No replays", "Replays folder not found.")
            return
        files = glob.glob(os.path.join(replays_dir, "*.html"))
        if not files:
            messagebox.showwarning("No replays", "No replay files found.")
            return
        # Sort by modification time, newest first
        files.sort(key=os.path.getmtime, reverse=True)
        last5 = files[:5]
        self._log(f"\nOpening last {len(last5)} replays:")
        for f in last5:
            name = os.path.basename(f)
            size_kb = os.path.getsize(f) // 1024
            self._log(f"  {name} ({size_kb} KB)")
        # Open in default browser (Windows)
        import webbrowser
        for f in last5:
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
