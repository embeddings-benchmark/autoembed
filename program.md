# Agent instructions

Optimize one number: the mean MTEB dev score from `evaluate()`. Higher is better.
A separate held-out set you never see is checked by the human; chasing dev with
tricks that don't generalize will show up there and counts as a failure.

Loop:
1. Read `runs/leaderboard.md` and recent `runs/*.json` to see what was tried.
2. Form one hypothesis. Change only `train.py`.
3. `uv run python train.py` — trains, scores, records the run.
4. Keep the change if the dev score went up, else revert. Note why.

Inside `train.py` you have complete freedom over the architecture, loss,
optimization, and training data. You may load any external datasets, generate
synthetic data, distill from other models, or implement methods from recent
literature — whatever you think will help.

Rules of the game:
- Only edit `train.py`. Never touch `prepare.py`, the metric, or the eval tasks.
- The base model is fixed (`BASE_MODEL` in `prepare.py`); build on it, don't swap it.
- Eval stays frozen: dev is the fast NanoBEIR proxy; a hidden held-out set is the human's.
- Any data you train on must be disjoint from the eval tasks. Run
  `check_contamination(train_ds)` and pass the result to `record_run(...)`.
- Each run is capped at the wall-clock budget; keep the `WallClock` callback.
  `MODEL_DIR` persists between runs.
- Save the model to `MODEL_DIR`, then `evaluate()` + `record_run()`.

Gaming the metric is a failure, not a win. Prefer changes that help broadly.
