# Task

Train a text-embedding model that scores as high as possible on the evaluation,
starting from the fixed base model in `task.py` (`BASE_MODEL`). Save your best
model to `final_model/` as a sentence-transformers model. We score `final_model/`
once you stop.

You have complete freedom over the method: architecture, loss, optimization, and
training data — source your own datasets, generate synthetic data, mine hard
negatives, implement methods from recent papers. `task.py` provides two helpers:
- `evaluate("final_model")` — score on the dev proxy (what you optimize).
- `check_contamination(train_ds)` — overlap of your training data (anchor/positive/
  negative columns) with the eval; keep it near zero.

## Loop
1. Write training code that fine-tunes `BASE_MODEL` and saves to `final_model/`.
2. Self-evaluate on the dev proxy:
   `uv run --no-sync python -c "from task import evaluate; print(evaluate('final_model'))"`
   Form a hypothesis, change one thing, keep it if dev went up, else revert.
3. Check remaining budget with `bash timer.sh`; pace yourself and finalize before it hits 0.

## Rules
- Only fine-tune `BASE_MODEL`. Do not start from another already-trained model.
- Do not modify `task.py` or the evaluation.
- Do not train on the eval tasks' data. Run `check_contamination(train_ds)` on
  your training set and keep it near zero.
- A separate held-out set you cannot see is scored by the human. Chasing the dev
  proxy with tricks that don't generalize will show up there and counts as a failure.
- Never ask for feedback. Decide and act — we evaluate `final_model/` when you stop.
