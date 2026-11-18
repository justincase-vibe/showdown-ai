"""P4.3 — the training GUI + its infrastructure (PLAN.md D77–D90).

Architecture (the user's tool: a web page to launch GPU training):

  gui/runstate.py     the on-disk RUN protocol — status / heartbeat / metrics /
                      stop flag / log / checkpoints under ``runs/<run-id>/``.
                      NOTHING about a live run lives in the GUI process's
                      memory, so a GUI restart never loses sight of training.
  gui/launcher.py     start/stop/inspect the trainer as a DETACHED child
                      process (new process group, no console window, log to
                      file) — killing or restarting the GUI cannot kill the
                      training. Supports both ``selfplay`` (PPO) and ``bc``
                      (imitation) modes (D88).
  gui/trainer_main.py the self-play SUBPROCESS entry: ``python -m
                      gui.trainer_main --config <run-dir>/config.json
                      --run-dir <run-dir>`` — GPU guard, net init (random or
                      ``PokeNet.load`` resume), the round loop with
                      per-battle heartbeats + graceful stop, metrics stream,
                      checkpointing, loud final status.
  gui/bc_main.py      the BC/imitation SUBPROCESS entry (D87): same runstate
                      protocol, maps the D86 stop/heartbeat hooks onto
                      runstate, writes the S4.3 verdict as the final metrics
                      row.
  gui/server.py       FastAPI app: REST control + WebSocket live-metrics
                      stream + the static dashboard. The server is a thin
                      control plane — it NEVER trains. Supports both modes
                      (D88).

Rules honored here: the agent NEVER runs training (the user launches it via
this GUI); the GUI's own code paths are CPU-verifiable (tests run a 1-battle
CPU smoke through the real subprocess to prove the loop works).
"""
