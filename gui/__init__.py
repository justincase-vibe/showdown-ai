"""P4.3 — the self-play training GUI + its infrastructure (PLAN.md D77/D79).

Architecture (the user's tool: a web page to launch GPU self-play training):

  gui/runstate.py     the on-disk RUN protocol — status / heartbeat / metrics /
                      stop flag / log / checkpoints under ``runs/<run-id>/``.
                      NOTHING about a live run lives in the GUI process's
                      memory, so a GUI restart never loses sight of training.
  gui/launcher.py     start/stop/inspect the trainer as a DETACHED child
                      process (new process group, no console window, log to
                      file) — killing or restarting the GUI cannot kill the
                      training.
  gui/trainer_main.py the training SUBPROCESS entry: ``python -m
                      gui.trainer_main --config <run-dir>/config.json
                      --run-dir <run-dir>`` — GPU guard, net init (random or
                      ``PokeNet.load`` resume), the round loop with
                      per-battle heartbeats + graceful stop, metrics stream,
                      checkpointing, loud final status.
  gui/server.py       FastAPI app: REST control + WebSocket live-metrics
                      stream + the static dashboard. The server is a thin
                      control plane — it NEVER trains.

Rules honored here: the agent NEVER runs training (the user launches it via
this GUI); the GUI's own code paths are CPU-verifiable (tests run a 1-battle
CPU smoke through the real subprocess to prove the loop works).
"""
