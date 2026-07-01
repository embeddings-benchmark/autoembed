# Task

Train a text-embedding model that scores as high as possible on the evaluation,
starting from the fixed base model in `task.py` (`BASE_MODEL`). Save your best model
to `final_model/` as a sentence-transformers model; we score it once you stop.

You have complete freedom over the method — architecture, loss, optimization, and
training data (your own datasets, synthetic data, hard negatives, methods from recent
papers). How you iterate is up to you.

Tools you can use:
- `uv run --no-sync python -c "from task import evaluate; print(evaluate('final_model'))"`
  — score `final_model/` on the dev proxy (the number you're optimizing).
- `check_contamination(train_ds)` in `task.py` — overlap of your training data
  (anchor/positive/negative columns) with the eval.
- `bash timer.sh` — remaining budget.

## Rules
- Only fine-tune `BASE_MODEL`; don't start from another already-trained model.
- Don't modify `task.py` or the evaluation.
- Don't train on the eval tasks' data (check with `check_contamination`).
- A hidden held-out set is scored by the human; tricks that don't generalize show
  up there and count as failure.
- Never ask for feedback — decide and act. We score `final_model/` when you stop.
