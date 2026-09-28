#!/usr/bin/env bash
set -euo pipefail

MODEL="${FLYGPT_LOCAL_MODEL:-qwen2.5:0.5b-instruct}"

if ! command -v ollama >/dev/null 2>&1; then
  echo "==> Installing Ollama from the official installer..."
  curl -fsSL https://ollama.com/install.sh | sh
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "==> Starting Ollama..."
  nohup ollama serve >/tmp/flygpt-ollama.log 2>&1 &

  for _ in $(seq 1 30); do
    if curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! curl -fsS http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  echo "Ollama did not start. Check /tmp/flygpt-ollama.log" >&2
  exit 1
fi

echo "==> Pulling local generator model: $MODEL"
ollama pull "$MODEL"

echo
echo "✅ Local generator is ready"
echo "Model: $MODEL"
echo "Endpoint: http://127.0.0.1:11434/v1/chat/completions"
echo
echo "Next:"
echo "  bash scripts/run_backend_v0_7_local.sh"
