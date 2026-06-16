# autoembed

A minimal auto-research loop for embedding models, scored on MTEB. An agent
edits `train.py`, trains on a fixed budget, scores on a fixed MTEB subset, reads
the result, iterates. Mirrors [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

- `prepare.py` — frozen: data pool, MTEB dev/held-out tasks, the metric, bookkeeping.
- `train.py` — the agent edits this: method + architecture.
- `program.md` — you edit this: agent instructions.

Principle: freeze the measurement, free the method. Held-out tasks detect gaming.

## Run (on the H100 box)

```bash
uv sync
uv run python train.py                         # one experiment
claude "Read program.md and kick off a new experiment."   # the loop
```
