#!/usr/bin/env bash
# Run an agent on the task, then score final_model. Local NVIDIA GPU required.
#   ./run_task.sh <agent> [config] [hours]      agents: claude | codex | gemini
# MODE=native (default) uses the repo's venv; MODE=docker uses the autoembed image.
set -euo pipefail

AGENT="${1:?usage: run_task.sh <agent> [config] [hours]}"
AGENT_CONFIG="${2:-}"
HOURS="${3:-${HOURS:-3}}"
MODE="${MODE:-native}"
ROOT="$(cd "$(dirname "$0")" && pwd)"

# Agent workdir: only agent-facing files. score.py / heldout tasks stay in the repo.
WORK="$(mktemp -d)"
cp "$ROOT"/{task.py,instructions.md,timer.sh,check_cuda.py,pyproject.toml,uv.lock} "$WORK"/
cp "$ROOT/agents/$AGENT/solve.sh" "$WORK/solve.sh"
mkdir -p "$WORK/final_model"

export PROMPT="$(cat "$ROOT/instructions.md")"
export AGENT_CONFIG
export DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
LIMIT="$(( HOURS * 3600 + 300 ))s"   # hard cap: budget + 5min grace

sandbox() {   # run "$1" in the sandbox; the venv comes from the environment
  if [ "$MODE" = native ]; then
    ( cd "$WORK" && export UV_PROJECT_ENVIRONMENT="$ROOT/.venv" && bash -c "$1" )
  else
    docker run --rm --gpus all -v "$WORK":/work -w /work \
      -e PROMPT -e AGENT_CONFIG -e DEADLINE -e AUTOEMBED_BASE_MODEL \
      -e UV_PROJECT_ENVIRONMENT=/opt/autoembed/.venv \
      -e ANTHROPIC_API_KEY -e OPENAI_API_KEY -e GEMINI_API_KEY \
      autoembed bash -c "$1"
  fi
}

echo ">> agent=$AGENT config=${AGENT_CONFIG:-default} budget=${HOURS}h mode=$MODE workdir=$WORK"

sandbox 'uv run --no-sync python check_cuda.py' \
  || { echo "!! no CUDA GPU visible — aborting"; exit 1; }

set +e
sandbox "timeout --signal=TERM --kill-after=30s $LIMIT bash solve.sh"
rc=$?
set -e
[ "$rc" -eq 124 ] && echo ">> agent hit the ${HOURS}h budget (killed)"
echo ">> agent exit=$rc; scoring final_model"

( cd "$ROOT" && uv run python score.py "$WORK/final_model" )
