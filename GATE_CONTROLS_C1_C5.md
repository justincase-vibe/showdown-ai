# Gate Controls C1–C5, Verdict Rules, and Work Decisions

**Canonical spec.** Source: Claude review (`gate_eval_review_and_controls.md`, relayed
by the user) + the user's DECISION message. Date: 2026-11-24. If chat context is
lost, THIS document + `OPEN_DESIGN_CHOICES.md` are the source of truth.

---

## 0. Work order (user DECISION — binding)

1. **(b) Run the gate NOW** with controls C1–C5. If the controls say
   **YARDSTICK_INVALID**, fix the yardstick first and re-run before anything else.
2. **(a) In parallel, on a branch, CPU-light only:** PPO fix pack =
   advantage normalization + minibatch PPO (**minibatch 256, 4 epochs, stop at
   approx-KL > 0.03**) + **at least 50 battles per update**. Shaping is a
   SEPARATE later commit. Gate for (a): **learnability test** (random-only
   opponent, 50 battles/round; win rate vs random must exceed **85% within about
   15 rounds**, else it is a bug).
3. **(c) Then BC warm start (Arm B)** on top of the fixed PPO, ping-pong turns
   filtered. Build the data pipeline in parallel; run the collection only when
   the servers are free.
4. **Compare fixed PPO from scratch vs BC+PPO** with the same controls, at
   least **400 battles per comparison**.

**Constraints:** Do NOT edit `gui/m1_gate_main.py` or `tools/gate_eval.py`
while the gate runs. Report numbers at each step. GPU runs are launched by the
USER via paste; the LLM works CPU-only.

---

## 1. Gate controls C1–C5 (verbatim)

Run each with the SAME seeds/teams and at least 200 battles, unless noted:

- **C1** random-init net (fresh `make_random_net`, same architecture) vs
  `strategy` AND `strategy_posterior`, greedy AND sampled.
- **C2** the `random` pool bot vs `strategy` AND `strategy_posterior`.
- **C3** the old `heuristic` bot vs `strategy` AND `strategy_posterior`.
- **C4** mirror: `strategy` vs `strategy`, and `strategy_posterior` vs
  `strategy` (expect about 50%; checks p1/p2 side bias).
- **C5** silent fallbacks of the strategy bot: count how often `decide()`
  returns `acts[0]` (ctx is None), hits the POV except (defaults to p1), or
  passes. Log counts per 100 battles. They must be zero or near zero.

> Use `fitness.harness` (bot on p1 vs reference on p2) for the bot controls.
> Extend `tools.gate_eval` with a `--controls` option for C1.

**Implementation note:** in THIS codebase the `random` pool member IS a fresh
`make_random_net` (`fitness/ladder.random_reference`), so C1 ≡ C2 mechanically —
they are still reported as separate rows (same seed space, identical results
expected). The shared parallel loop (`run_eval_battles` with `my_ref`) keeps
the bot on p1 vs the reference on p2, the same placement `fitness.harness`
uses, but parallel instead of sequential. C4/C5 are implemented via the
`SUBJECT_TO_POOL` extension and `fitness.strategy_agent` counters.

---

## 2. Gate rule changes in `tools/gate_eval.py` (verbatim)

- Print the Wilson lower 95% bound per **(opponent × mode)** AND **pooled per
  opponent**.
- The verdict is the **WORST of the pooled bounds of `strategy` and
  `strategy_posterior`** (not strategy_posterior only).
- Print a **warning** when greedy and sampled differ by more than **10 points**.
- Thresholds: lower bound **>= 0.40 PASS, >= 0.25 CONTINUE, else STOP**, but
  only if the validity rules below hold.

## 3. Validity rules (print them in the verdict) (verbatim)

- **YARDSTICK_INVALID** if C2 (random bot) beats the strategy yardstick more
  than **25%** of the time, OR if the C1 random-init net's **UPPER 95% bound is
  >= 0.40** (a random net could pass the gate, so the gate cannot discriminate).
  Fix the yardstick first.
- Otherwise the net's pooled win rate must **exceed C1's by at least 10 points
  with non-overlapping 95% intervals**, or the verdict is **capped at
  CONTINUE**.

## 4. Reporting (verbatim)

- Test BOTH `m1gate_round80` and `m1gate_round70` (pre-registered; do NOT
  cherry-pick). Fresh seed, n-per-pair 200, both modes.
- Per (opponent × mode) W/L/D with Wilson bounds, pooled bounds, all controls,
  the validity verdict.
- Always add the sentence: **"in-distribution only (same 25-team pool as
  training); not a generalization claim"**.
- Do NOT quote the round-70 0.72 from the earlier eval (selected as the best of
  8 noisy evals: winner's curse).

## 5. Stats corrections for earlier reports (verbatim)

- n=100 gives a 95% Wilson interval of about **±0.10** (not ±0.19).
- Report the cumulative pool win rate **per opponent and per window**
  (rounds 1–20 vs 61–80), not as one number.
- Measured (from `runs/20261123T195442Z/metrics.jsonl`): yardstick vs strategy
  r01–30: 0.547 (lo 0.423) · r31–60: 0.553 (lo 0.430) · r61–80: 0.670
  (lo 0.519). Training-pool WR by window: 0.432 → 0.605 → 0.618.

---

## 6. Implementation status (keep updated)

| # | item | state |
|---|------|-------|
| 1 | `--subject` any pool member (C1–C4 expressible) | DONE (commit a6d9165 + follow-up) |
| 2 | `run_eval_battles(my_ref=...)` shared loop, bot on p1 | DONE (a6d9165) |
| 3 | per (opp×mode) bounds + pooled per opp | DONE (a6d9165) |
| 4 | verdict = worst of POOLED bounds of strategy AND strategy_posterior | DONE |
| 5 | warning when greedy vs sampled differ > 10 pts | DONE |
| 6 | validity rules (YARDSTICK_INVALID / 10-pt + CI-overlap cap) | DONE |
| 7 | C5 strategy-bot silent-fallback counters (per-pair delta) | DONE |
| 8 | in-distribution sentence in report + result.json | DONE (a6d9165) |
| 9 | both r70 + r80 gated | DONE (multi-ckpt) |
| 10 | PPO fix pack (adv-norm + minibatch 256 + KL 0.03 stop) | IN PROGRESS on `bionic/ppo-fix-pack` |
| 11 | battles/round >= 50 for the fixed-PPO runs | TODO (m1gate config; frozen while gate runs) |
| 12 | learnability test (vs random, 50/round, >85% in ~15 rounds) | TODO (needs servers) |
| 13 | BC warm start pipeline (Arm B, ping-pong filtered) | TODO (pipeline code first) |
| 14 | fixed-PPO vs BC+PPO comparison, >=400 battles, same controls | TODO |
| 15 | reward shaping | SEPARATE later commit (explicit) |

## 7. The gate pastes (two-stage — controls decide validity first)

**Run 1 — controls C1, C3, C4a, C4b** (C2 ≡ C1 in this codebase; its row is
filled by the validity fallback). ~2400 battles ≈ 8 h at 6×2.

```cmd
cd /d "C:\Users\JustinCase\Documents\Projects\showdown AI-imitation" && python -m tools.gate_eval --subject random_init,heuristic,strategy:strategy,strategy_posterior:strategy --opps strategy,strategy_posterior --n-per-pair 200 --modes greedy,sampled --device cuda --seed 20261125 --servers 6 --per-server 2 --port-base 49980
```

**Run 2 — the gate (only if Run 1 did NOT say YARDSTICK_INVALID)** — both
checkpoints + C1 in the same run (fills the validity check). ~1600 battles ≈ 5.3 h.

```cmd
cd /d "C:\Users\JustinCase\Documents\Projects\showdown AI-imitation" && python -m tools.gate_eval --subject random_init --ckpt runs/20261123T195442Z/checkpoints/m1gate_round70.pt,runs/20261123T195442Z/checkpoints/m1gate_round80.pt --opps strategy,strategy_posterior --n-per-pair 200 --modes greedy,sampled --device cuda --seed 20261125 --servers 6 --per-server 2 --port-base 49980
```

Reading: every entry gets per-mode bounds, pooled bounds, FINAL verdict;
the VALIDITY block prints C1/C2 rows + YARDSTICK status; C5 deltas print per
pair. Stop = close the console window.

## 8. Environment notes

- Windows Defender / "Application Control policy" blocked the pyarrow `_parquet`
  DLL once (2 user notifications) → `net/tests/test_s42.py` failed collection
  that day. System issue, not project code.
- RAM: 31 GB; 6×2 (12 concurrent) is the working size; 8×2 hit the RAM max;
  4×2 is the safe fallback. ~5 battles/min at 6×2.
- Freeze list (do not commit): `bot/requirements.txt`, `bot/team_pools.json`,
  `check_battles.ps1`, `start_selfplay.bat`, `start_servers_10.bat`,
  `scratch_learnset_probe.js`, `scratch_team.json`.
- Git: `git add <specific files>` only. Remote: `justincase-vibe/showdown-ai`.
