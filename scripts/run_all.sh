#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [[ -f ".env" ]]; then
  echo "🔐 Loading .env"
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

MODEL="${FLYGPT_LOCAL_MODEL:-qwen2.5:0.5b-instruct}"

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
  echo "🐍 Creating Python virtual environment..."
  python3 -m venv .venv
fi

REQ_HASH="$(sha256sum requirements.txt | awk '{print $1}')"
REQ_MARKER=".venv/.flygpt-requirements.sha256"
INSTALLED_HASH="$(cat "$REQ_MARKER" 2>/dev/null || true)"

if [[ "$REQ_HASH" != "$INSTALLED_HASH" ]] || ! .venv/bin/python -c "import uvicorn, fastapi, neuprint, pandas, jinja2" >/dev/null 2>&1; then
  echo "📦 Backend dependencies changed or are incomplete. Installing requirements..."
  .venv/bin/python -m pip install -r requirements.txt
  printf '%s\n' "$REQ_HASH" > "$REQ_MARKER"
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

echo "⏳ Waiting for backend health check..."
BACKEND_READY=0
for _ in $(seq 1 40); do
  if curl -fsS http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
    BACKEND_READY=1
    break
  fi
  if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
    echo "❌ Backend exited before port 8000 became ready."
    echo "   Check the backend error printed above."
    exit 1
  fi
  sleep 0.5
done

if [[ "$BACKEND_READY" -ne 1 ]]; then
  echo "❌ Backend did not become ready on port 8000."
  exit 1
fi

echo "✅ Backend ready"

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
