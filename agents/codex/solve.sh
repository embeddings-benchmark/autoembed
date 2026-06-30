#!/bin/bash
# Launch Codex CLI on the task. Reads $PROMPT, runs in the current workdir.
unset ANTHROPIC_API_KEY GEMINI_API_KEY
printf '%s' "$PROMPT" | codex --search exec --json --skip-git-repo-check --yolo \
    --model "${AGENT_CONFIG:-gpt-5.3-codex}"
