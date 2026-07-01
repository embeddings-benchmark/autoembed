#!/usr/bin/env bash
# Run an agent on the task, then score final_model. Local NVIDIA GPU required.
#   ./run_task.sh <agent> [config] [hours]      agents: claude | codex | gemini
# MODE=native (default) uses the repo's venv; MODE=docker uses the autoembed image.
# Per-run artifacts land in results/<id>/: prompt, trace, final_model, scores, meta.
set -euo pipefail

AGENT="${1:?usage: run_task.sh <agent> [config] [hours]}"
AGENT_CONFIG="${2:-}"
HOURS="${3:-${HOURS:-3}}"
MODE="${MODE:-native}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
BASE_MODEL="${AUTOEMBED_BASE_MODEL:-microsoft/mpnet-base}"

RUN_ID="$(date +%Y%m%d-%H%M%S)_${AGENT}"
RESULTS="$ROOT/results/$RUN_ID"
mkdir -p "$RESULTS"

WORK="$(mktemp -d)"
cp "$ROOT"/{task.py,instructions.md,timer.sh,check_cuda.py,pyproject.toml,uv.lock} "$WORK"/
cp "$ROOT/agents/$AGENT/solve.sh" "$WORK/solve.sh"
mkdir -p "$WORK/final_model"

export PROMPT="$(cat "$ROOT/instructions.md")"
export AGENT_CONFIG
export DEADLINE=$(( $(date +%s) + HOURS * 3600 ))
LIMIT="$(( HOURS * 3600 + 300 ))s"   # hard cap: budget + 5min grace
printf '%s' "$PROMPT" > "$RESULTS/prompt.txt"

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

echo ">> agent=$AGENT config=${AGENT_CONFIG:-default} base=$BASE_MODEL budget=${HOURS}h mode=$MODE"
echo ">> results=$RESULTS"

sandbox 'uv run --no-sync python check_cuda.py' \
  || { echo "!! no CUDA GPU visible — aborting"; exit 1; }

START=$(date +%s)
set +e
sandbox "timeout --signal=TERM --kill-after=30s $LIMIT bash solve.sh" 2>&1 \
  | python3 "$ROOT/timestamp_lines.py" | tee "$RESULTS/trace.log"
rc=${PIPESTATUS[0]}
set -e
END=$(date +%s)
[ "$rc" -eq 124 ] && echo ">> agent hit the ${HOURS}h budget (killed)"

python3 "$ROOT/parse_trace.py" "$RESULTS/trace.log" "$RESULTS/trace.md" || true

if [ -n "$(ls -A "$WORK/final_model" 2>/dev/null)" ]; then
  echo ">> scoring final_model"
  ( cd "$ROOT" && uv run python score.py "$WORK/final_model" "$RESULTS/scores.json" ) \
    | tee "$RESULTS/score.log"
  cp -r "$WORK/final_model" "$RESULTS/final_model"
else
  echo ">> no final_model produced — skipping scoring"
fi

BUDGET_HIT=$([ "$rc" -eq 124 ] && echo true || echo false)
RUN_ID="$RUN_ID" AGENT="$AGENT" AGENT_CONFIG="$AGENT_CONFIG" BASE_MODEL="$BASE_MODEL" \
HOURS="$HOURS" MODE="$MODE" DURATION="$((END - START))" RC="$rc" BUDGET_HIT="$BUDGET_HIT" \
python3 - "$RESULTS/meta.json" <<'PY'
import json, os, sys
json.dump({
    "run_id": os.environ["RUN_ID"], "agent": os.environ["AGENT"],
    "agent_config": os.environ["AGENT_CONFIG"], "base_model": os.environ["BASE_MODEL"],
    "budget_hours": int(os.environ["HOURS"]), "mode": os.environ["MODE"],
    "duration_s": int(os.environ["DURATION"]), "agent_exit": int(os.environ["RC"]),
    "budget_hit": os.environ["BUDGET_HIT"] == "true",
}, open(sys.argv[1], "w"), indent=2)
PY

echo ">> done: $RESULTS"
