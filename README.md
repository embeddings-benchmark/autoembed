# autoembed

Can an agent train a good embedding model on its own? `autoembed` is a benchmark
for *agentic* embedding-model training: a CLI agent (Claude Code, Codex, Gemini…)
is given a fixed base model, a wall-clock budget on one GPU, and a dev eval to
query — and must produce the best embedding model it can. The harness then scores
the agent's `final_model/` on a frozen eval, including a held-out set the agent
never sees.

The measurement is frozen; the method is free.

## How it works

1. `run_task.sh <agent>` seeds a fresh workdir with the **agent-facing files only**
   (`task.py`, `instructions.md`, `timer.sh`), sets a deadline, and launches the
   agent via `agents/<agent>/solve.sh`.
2. The agent reads `instructions.md`, writes its own training code (using the
   helpers in `task.py`), self-evaluates on the **dev proxy** (`task.evaluate`),
   paces itself with `bash timer.sh`, and saves its best model to `final_model/`.
3. When the agent stops, `score.py` runs **outside** the workdir and scores
   `final_model/` on the dev proxy, the **hidden held-out** set, and a
   contamination audit.

Integrity is structural, not honor-system: the scorer (`score.py`), which defines
and runs the held-out set, never enters the agent's workdir, and the harness scores
from its own copy — so the agent can't see or tamper with the official metric.

## Files

| | |
|---|---|
| `task.py` | fixed base model, dev eval, contamination check (the agent gets a copy) |
| `instructions.md` | the prompt given to the agent |
| `agents/<agent>/solve.sh` | per-agent launchers (claude · codex · gemini) |
| `run_task.sh` | orchestrator: seed workdir → run agent → score |
| `timer.sh` | remaining-budget query the agent calls |
| `score.py` | harness scoring: dev + hidden held-out + contamination; defines the held-out tasks (harness-only) |
| `reference.py` | reference-model anchors on the dev proxy |
| `Dockerfile` | container env; run with `docker run --gpus all` |

## Run

Needs a local NVIDIA GPU, the agent CLI installed + authenticated (API key or
subscription), and `uv`.

```bash
uv sync
./run_task.sh claude              # native: run Claude Code for the default budget
HOURS=10 ./run_task.sh codex      # 10-hour budget with Codex
MODE=docker ./run_task.sh gemini  # isolated container run (needs the agent CLI in the image)
```

GPU note: `pyproject.toml` pins torch to a CUDA 12.8 wheel index; adjust it to
your driver (see the comment there).
