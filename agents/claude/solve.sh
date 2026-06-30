#!/bin/bash
# Launch Claude Code on the task. Reads $PROMPT, runs in the current workdir.
unset OPENAI_API_KEY GEMINI_API_KEY
printf '%s' "$PROMPT" | claude --print --verbose --model "${AGENT_CONFIG:-claude-opus-4-8}" \
    --output-format stream-json --dangerously-skip-permissions
