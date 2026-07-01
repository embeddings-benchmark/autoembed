#!/bin/bash
# Launch Claude Code on the task. Reads $PROMPT, runs in the current workdir.
unset OPENAI_API_KEY GEMINI_API_KEY
export CLAUDE_CONFIG_DIR="$PWD/.claude-agent"
printf '%s' "$PROMPT" | claude --bare --print --verbose \
    --model "${AGENT_CONFIG:-claude-opus-4-8}" \
    --output-format stream-json --dangerously-skip-permissions
