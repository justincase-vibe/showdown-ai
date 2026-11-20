"""``python -m gui`` — launch the training console.

    python -m gui            -> http://127.0.0.1:8700
    GUI_PORT=9000 python -m gui

The server is the thin control plane only (gui.server) — training runs in
the detached ``gui.trainer_main`` subprocess it spawns, so closing this
console never stops a training run.
"""
import os

import uvicorn


def main() -> None:
    port = int(os.environ.get("GUI_PORT", "8700"))
    uvicorn.run("gui.server:app", host="127.0.0.1", port=port, log_level="info")


if __name__ == "__main__":
    main()
