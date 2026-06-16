# Agent instructions

Optimize one number: the mean MTEB dev score from `evaluate()`. Higher is better.

Loop:
1. Read `runs/leaderboard.md` and recent `runs/*.json` to see what was tried.
2. Form one hypothesis. Change only `train.py`.
3. `uv run python train.py` — trains, scores, records the run.
4. Keep the change if score went up, else revert. Note why each run won/lost.

Rules:
- Only edit `train.py`. Never touch `prepare.py`, the metric, the tasks, or the data pool.
- Stay within the wall-clock budget; keep the `WallClock` callback.
- Save the model to `MODEL_DIR`, then call `evaluate()` + `record_run()`.
- Everything in `train.py` is fair game, including the architecture.

Gaming the metric is a failure, not a win. A held-out set (`HELDOUT_TASKS`)
that you never see is checked by the human. Prefer changes that help broadly.
