#!/bin/bash
# Launch Gemini CLI on the task. Reads $PROMPT, runs in the current workdir.
unset ANTHROPIC_API_KEY OPENAI_API_KEY
export GEMINI_SANDBOX="false"
gemini --yolo --model "${AGENT_CONFIG:-gemini-3-pro}" --output-format stream-json -p "$PROMPT"
