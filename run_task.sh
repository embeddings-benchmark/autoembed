#!/usr/bin/env bash
# Run an agent end-to-end on the embedding-training task, then score its
# final_model on the frozen eval. Works on any box with a local GPU.
#
#   ./run_task.sh <agent> [agent_config] [hours]
#   agents: claude | codex | gemini   (see agents/<agent>/solve.sh)
#
# Requires a local NVIDIA GPU, the agent CLI installed + authenticated (API key
# env var or subscription), and either Docker with the NVIDIA runtime or a native
# uv env. Set MODE=native to run without Docker.
set -euo pipefail

AGENT="${1:?usage: run_task.sh <agent> [config] [hours]}"
AGENT_CONFIG="${2:-}"
HOURS="${3:-${HOURS:-3}}"
MODE="${MODE:-native}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Fresh agent workdir: only the agent-facing files. heldout.py / score.py
# stay out, so the held-out tasks are never visible to the agent.
WORK="$(mktemp -d)"
cp "$ROOT"/{task.py,instructions.md,timer.sh,pyproject.toml,uv.lock} "$WORK"/
cp "$ROOT/agents/$AGENT/solve.sh" "$WORK/solve.sh"
mkdir -p "$WORK/final_model"

export PROMPT="$(cat "$ROOT/instructions.md")"
export AGENT_CONFIG
export DEADLINE=$(( $(date +%s) + HOURS * 3600 ))

echo ">> agent=$AGENT config=${AGENT_CONFIG:-default} budget=${HOURS}h workdir=$WORK mode=$MODE"
if [ "$MODE" = native ]; then
  ( cd "$WORK" && bash solve.sh )
else
  docker run --rm --gpus all -v "$WORK":/work -w /work \
    -e PROMPT -e AGENT_CONFIG -e DEADLINE -e AUTOEMBED_BASE_MODEL \
    -e ANTHROPIC_API_KEY -e OPENAI_API_KEY -e GEMINI_API_KEY \
    autoembed bash solve.sh
fi

echo ">> scoring $WORK/final_model"
( cd "$ROOT" && uv run python score.py "$WORK/final_model" )
