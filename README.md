# autoembed

Can an agent train a good embedding model on its own? `autoembed` is a small
benchmark harness for *agentic* embedding-model training: an agent edits one
file, trains under a fixed budget, is scored on a frozen MTEB eval, reads the
result, and iterates. The measurement is frozen; the method is free, and a
held-out set the agent never sees detects gaming.

## Design

- **Fixed (frozen harness — `prepare.py`):** the base model, the training
  budget, the eval tasks, the contamination check, and bookkeeping. The agent
  may read this but must not edit it.
- **Free (the agent's job — `train.py`):** architecture, loss, optimization, and
  data — including external datasets, synthetic data, hard negatives, and
  methods from recent papers. The agent rewrites this each iteration.
- **Eval:** a fast NanoBEIR retrieval proxy (dev) that the agent optimizes, plus
  a hidden held-out set (`eval_heldout.py`) only the human runs.
- **Integrity:** every training set is checked for overlap with the eval tasks
  (`check_contamination`); the held-out set is never exposed to the agent.

Principle: **freeze the measurement, free the method.**

## Files

- `prepare.py` — frozen: base model, data pool, dev eval, contamination, bookkeeping.
- `train.py` — the agent edits this: the trainer (method + architecture + data).
- `program.md` — agent instructions (the rules of the game).
- `eval_heldout.py` — human-run held-out eval + leakage audit (agent never sees it).
- `eval_ref.py` — score reference models (e.g. `all-mpnet-base-v2`) on the dev proxy.
- `gpu.sh` — environment-specific Slurm launcher (example; adapt to your setup).

## Run

```bash
uv sync
uv run python train.py         # one experiment (trains, scores, records)
uv run python eval_heldout.py  # held-out generalization (human-run)
```

Drive the loop with an agent:

```bash
claude "Read program.md and kick off a new experiment."
```

GPU note: `pyproject.toml` pins torch to a CUDA 12.8 wheel index; adjust it to
your driver (see the comment there).
