#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export FLYGPT_GENERATOR_URL="${FLYGPT_GENERATOR_URL:-http://127.0.0.1:11434/v1/chat/completions}"
export FLYGPT_GENERATOR_MODEL="${FLYGPT_GENERATOR_MODEL:-qwen2.5:0.5b-instruct}"
export FLYGPT_GENERATOR_PROVIDER="${FLYGPT_GENERATOR_PROVIDER:-ollama-local}"
export FLYGPT_GENERATOR_TIMEOUT="${FLYGPT_GENERATOR_TIMEOUT:-60}"
export FLYGPT_GENERATOR_TEMPERATURE="${FLYGPT_GENERATOR_TEMPERATURE:-0.35}"
export FLYGPT_GENERATOR_MAX_TOKENS="${FLYGPT_GENERATOR_MAX_TOKENS:-384}"

echo "🪰 FlyGPT v0.7 backend"
echo "Generator: $FLYGPT_GENERATOR_PROVIDER / $FLYGPT_GENERATOR_MODEL"
echo "URL: $FLYGPT_GENERATOR_URL"
echo

if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
else
  PYTHON="${PYTHON:-python}"
fi

if ! "$PYTHON" -c "import uvicorn" >/dev/null 2>&1; then
  echo "❌ uvicorn is not installed for: $PYTHON"
  echo "Install project dependencies first, or create .venv."
  exit 1
fi

echo "Python: $PYTHON"
exec "$PYTHON" -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
