# `_legacy/bot/` — RETIRED first-generation bot (DO NOT USE)

These files are the **original PPO bot** (`bot/` generation: `policy.py` MaskedPPO,
`train.py` trainer, `env.py` env, `obs.py` flat-892 obs, `launcher.py` the old tkinter
GUI, plus the standalone scripts `play.py` / `quick_eval.py` / `imitate.py` /
`state_from_log.py` / `progress.py` / `repro_train.py` / `debug_forfeit.py` /
`run_gpu_watch.py` / `test_decide.py` / `test_mask_fix.py`).

They have been **superseded** by the current generation and quarantined here so the
project has **ONE workflow** instead of two:

| Concern            | Retired (here)          | **Current (use this)**                                  |
|--------------------|-------------------------|---------------------------------------------------------|
| Policy / PPO       | `policy.py`             | `net/ppo.py` + `net/trainer.py` (`SelfPlayTrainer`)     |
| Environment        | `env.py`                | `net/env.py` (`ShowdownEnv`)                            |
| Observation        | `obs.py` (flat 892-d)   | `belief/` + `net/encoder.py` (belief-state embeddings)   |
| Trainer entry      | `train.py`              | `gui/trainer_main.py` (drives `net.trainer`)            |
| GUI (desktop)      | `launcher.py` (tkinter) | `python -m gui.desktop` (Mission Control)               |
| Imitation          | `imitate.py`            | `gui/bc_main.py` (drives `net/bc.py`)                   |
| Self-play league   | (n/a)                   | `league/` (round-robin + ELO)                           |

## What stayed in `bot/` (still load-bearing — do NOT touch)
The current generation still reads these from `bot/`:
- `teams.py` — format list + team builders (used by the native GUI's format dropdown)
- `team_pools.json`, `data/`, `replays/`, `checkpoints/` — **data** (replays, teams, nets)
- `datamine/` — the HuggingFace replay downloader (wrapped by `corpus/pull.py`)
- `smogon_teams.py`, `teamgen.py`, `build_movesets.py` — the team-pool scrapers/builders
- `action_space.py`, `reveal.py` — small shared helpers still imported by `datamine/parse.py`
- `requirements.txt`, `replay_diag.py` (replay health-check), `play_online.log`/etc.

## Reverting
These files were **moved**, not deleted. To restore one:
`mv _legacy/bot/<file> bot/<file>`
</new_content>
            </invoke>
