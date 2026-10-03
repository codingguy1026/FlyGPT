#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

MODEL="${FLYGPT_LOCAL_MODEL:-qwen2.5:0.5b-instruct}"

if [[ -f ".env" ]]; then
  echo "🔐 Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if [[ -z "${NEUPRINT_TOKEN:-}" && -z "${NEUPRINT_APPLICATION_CREDENTIALS:-}" ]]; then
  echo "⚠️  neuPrint token is not set. MaleCNS queries will return a service error until you add one to .env."
fi
OLLAMA_URL="http://127.0.0.1:11434/api/tags"

BACKEND_PID=""
FRONTEND_PID=""
OLLAMA_PID=""

cleanup() {
  trap - INT TERM EXIT
  echo
  echo "🧹 Stopping FlyGPT..."

  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
  fi

  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
  fi

  # Only stop Ollama when this script started it.
  if [[ -n "$OLLAMA_PID" ]] && kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID" 2>/dev/null || true
  fi

  wait 2>/dev/null || true
}

trap cleanup INT TERM EXIT

echo "🪰 FlyGPT v0.8.0 full stack"
echo

if ! command -v ollama >/dev/null 2>&1; then
  echo "❌ Ollama is not installed."
  echo "Run: bash scripts/setup_ollama_generator.sh"
  exit 1
fi

if ! curl -fsS "$OLLAMA_URL" >/dev/null 2>&1; then
  echo "🧠 Starting Ollama on :11434..."
  ollama serve >/tmp/flygpt-ollama.log 2>&1 &
  OLLAMA_PID=$!

  for _ in $(seq 1 30); do
    if curl -fsS "$OLLAMA_URL" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
fi

if ! curl -fsS "$OLLAMA_URL" >/dev/null 2>&1; then
  echo "❌ Ollama did not start. Check /tmp/flygpt-ollama.log"
  exit 1
fi

if ! ollama list 2>/dev/null | awk 'NR > 1 {print $1}' | grep -Fxq "$MODEL"; then
  echo "📦 Local model not found. Pulling $MODEL..."
  ollama pull "$MODEL"
fi

if [[ ! -x ".venv/bin/python" ]]; then
  echo "❌ Python virtual environment not found at .venv/"
  echo "Create/install the backend environment first."
  exit 1
fi

if ! .venv/bin/python -c "import uvicorn" >/dev/null 2>&1; then
  echo "❌ uvicorn is not installed in .venv"
  echo "Install the backend dependencies first."
  exit 1
fi

if [[ ! -f "frontend/package.json" ]]; then
  echo "❌ frontend/package.json not found."
  exit 1
fi

if [[ ! -d "frontend/node_modules" ]]; then
  echo "📦 Frontend dependencies not found. Running npm install..."
  (cd frontend && npm install)
fi

echo
echo "🚀 Starting backend  : http://127.0.0.1:8000"
bash scripts/run_backend_v0_7_local.sh &
BACKEND_PID=$!

echo "🎨 Starting frontend : http://127.0.0.1:3000"
(
  cd frontend
  npm run dev
) &
FRONTEND_PID=$!

echo
echo "✅ FlyGPT is launching"
echo "   Ollama   :11434"
echo "   Backend  :8000"
echo "   Frontend :3000"
echo
echo "Press Ctrl+C once to stop the stack."
echo

# Exit if either app process dies; the trap cleans up the other one.
wait -n "$BACKEND_PID" "$FRONTEND_PID"
STATUS=$?

if [[ $STATUS -ne 0 ]]; then
  echo "⚠️ One FlyGPT process exited with status $STATUS."
fi

exit "$STATUS"
