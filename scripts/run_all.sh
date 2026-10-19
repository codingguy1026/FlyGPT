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

STABLE_MODE=0
for argument in "$@"; do
  case "$argument" in
    --local-only)
      export FLYGPT_LOCAL_ONLY=1
      ;;
    --stable)
      STABLE_MODE=1
      ;;
    *)
      echo "Usage: bash scripts/run_all.sh [--local-only] [--stable]" >&2
      exit 2
      ;;
  esac
done

# In stable mode, disable the backend file watcher and serve a built Next.js
# frontend rather than running two development/watch processes.
export FLYGPT_STABLE_MODE="$STABLE_MODE"

# Explicit local-only mode never invokes a hosted language model. It also
# rejects stale hosted URLs rather than silently routing to them.
case "${FLYGPT_LOCAL_ONLY:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    export FLYGPT_LOCAL_ONLY=1
    export FLYGPT_GENERATOR_URL="${FLYGPT_GENERATOR_URL:-http://127.0.0.1:11434/v1/chat/completions}"
    export FLYGPT_GENERATOR_MODEL="${FLYGPT_GENERATOR_MODEL:-${FLYGPT_LOCAL_MODEL:-qwen3:4b-instruct}}"
    export FLYGPT_GENERATOR_PROVIDER=ollama-local
    if ! python3 -c 'import sys; from local_generator_policy import is_local_generator_url; sys.exit(0 if is_local_generator_url(sys.argv[1]) else 1)' "$FLYGPT_GENERATOR_URL"; then
      echo "❌ FLYGPT_LOCAL_ONLY forbids hosted generator URLs: $FLYGPT_GENERATOR_URL" >&2
      exit 1
    fi
    unset GEMINI_API_KEY FLYGPT_GENERATOR_API_KEY FLYGPT_GENERATOR_FALLBACK_MODEL FLYGPT_GENERATOR_REASONING_EFFORT
    echo "🔒 LOCAL ONLY: hosted LLM generation disabled"
    ;;
esac

# If a Gemini API key is present and no explicit generator endpoint was chosen,
# use Gemini 3.8 Flash through Google's OpenAI-compatible chat endpoint.
if [[ -n "${GEMINI_API_KEY:-}" && -z "${FLYGPT_GENERATOR_URL:-}" ]]; then
  export FLYGPT_GENERATOR_URL="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
  export FLYGPT_GENERATOR_MODEL="${FLYGPT_GENERATOR_MODEL:-gemini-3.8-flash}"
  export FLYGPT_GENERATOR_PROVIDER="gemini"
  export FLYGPT_GENERATOR_API_KEY="${FLYGPT_GENERATOR_API_KEY:-$GEMINI_API_KEY}"
  export FLYGPT_GENERATOR_REASONING_EFFORT="${FLYGPT_GENERATOR_REASONING_EFFORT:-low}"
  export FLYGPT_GENERATOR_FALLBACK_MODEL="${FLYGPT_GENERATOR_FALLBACK_MODEL:-gemini-3.7-flash}"
  export FLYGPT_GENERATOR_RETRY_DELAY="${FLYGPT_GENERATOR_RETRY_DELAY:-0.6}"
fi

MODEL="${FLYGPT_GENERATOR_MODEL:-${FLYGPT_LOCAL_MODEL:-qwen3:4b-instruct}}"
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"

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

echo "🪰 FlyGPT full stack"
if [[ "$STABLE_MODE" -eq 1 ]]; then
  echo "🛡️ Stable mode: production frontend, backend without auto-reload"
else
  echo "🛠️ Development mode: hot reload enabled"
fi
echo

USE_OLLAMA=1
if [[ -n "${FLYGPT_GENERATOR_URL:-}" ]]; then
  case "$FLYGPT_GENERATOR_URL" in
    http://127.0.0.1:11434/*|http://localhost:11434/*) ;;
    *) USE_OLLAMA=0 ;;
  esac
fi

if [[ "$USE_OLLAMA" -eq 1 ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "❌ Ollama is not installed."
    echo "Run: bash scripts/setup_ollama_generator.sh"
    exit 1
  fi

  if ! curl -fsS "$OLLAMA_URL" >/dev/null 2>&1; then
    echo "🧠 Starting Ollama on :11434... (keep-alive: $OLLAMA_KEEP_ALIVE)"
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

  echo "🔥 Preloading $MODEL so the first chat does not pay the cold-start cost..."
  curl -fsS http://127.0.0.1:11434/api/generate \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"${MODEL}\",\"prompt\":\"\",\"keep_alive\":\"${OLLAMA_KEEP_ALIVE}\"}" \
    >/dev/null || echo "⚠️  Model preload failed; FlyGPT can still start, but the first response may be slow."
else
  echo "☁️  Hosted generator: ${FLYGPT_GENERATOR_PROVIDER:-compatible-http} / ${FLYGPT_GENERATOR_MODEL:-unset}"
  if [[ -n "${FLYGPT_GENERATOR_FALLBACK_MODEL:-}" ]]; then
    echo "🛟 Generator fallback: ${FLYGPT_GENERATOR_FALLBACK_MODEL}"
  fi
  if [[ "${FLYGPT_GENERATOR_PROVIDER:-}" == "gemini" && -z "${FLYGPT_GENERATOR_API_KEY:-}" ]]; then
    echo "⚠️  Gemini is selected but no API key is configured."
  fi
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
  .venv/bin/python -m pip install --no-cache-dir -r requirements.txt
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

# Existing servers may serve stale code even when their ports return 200.
# Next.js also falls back to 3001 when port 3000 is occupied, which makes a
# previously opened Codespaces -3000 URL appear outdated.
for port in 3000 8000; do
  if python3 -c 'import socket,sys; s=socket.socket(); s.settimeout(0.5); busy=(s.connect_ex(("127.0.0.1",int(sys.argv[1])))==0); s.close(); sys.exit(0 if busy else 1)' "$port"; then
    echo "❌ Port $port is already in use. Stop the previous make run/server first."
    echo "   Inspect: ss -ltnp | grep :$port"
    echo "   Tip: make doctor checks the existing servers without stopping them."
    exit 1
  fi
done

if [[ "$STABLE_MODE" -eq 1 ]]; then
  echo "🏗️ Building the frontend before starting the backend to avoid build-time memory contention..."
  (cd frontend && NEXT_TELEMETRY_DISABLED=1 npm run build)
fi

echo
echo "🚀 Starting backend  : http://127.0.0.1:8000"
bash scripts/run_backend_v0_7_local.sh &
BACKEND_PID=$!

echo "⏳ Waiting for backend health check..."
BACKEND_READY=0
for _ in $(seq 1 180); do
  if curl -fsS --max-time 2 http://127.0.0.1:8000/api/health >/dev/null 2>&1; then
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
  echo "❌ Backend did not become ready on port 8000 after 90 seconds."
  echo "   Try: curl -i http://127.0.0.1:8000/api/health"
  exit 1
fi

echo "✅ Backend ready"

echo "🎨 Starting frontend : http://127.0.0.1:3000"
(
  cd frontend
  # Run Next.js as the tracked process, not as a child of npm, so cleanup
  # can stop the server itself rather than leaving a stale port listener.
  if [[ "$STABLE_MODE" -eq 1 ]]; then
    exec ./node_modules/.bin/next start --hostname 0.0.0.0 --port 3000
  else
    exec ./node_modules/.bin/next dev --hostname 0.0.0.0 --port 3000
  fi
) &
FRONTEND_PID=$!

echo "⏳ Waiting for frontend and same-origin API rewrite..."
FRONTEND_READY=0
for _ in $(seq 1 90); do
  if curl -fsS --max-time 2 http://127.0.0.1:3000/ >/dev/null 2>&1 && \
     curl -fsS --max-time 2 http://127.0.0.1:3000/api/health >/dev/null 2>&1; then
    FRONTEND_READY=1
    break
  fi
  if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
    echo "❌ Next.js frontend exited before port 3000 became ready."
    echo "   Inspect the npm/Next.js error printed above. Run 'make doctor'."
    exit 1
  fi
  sleep 1
done
if [[ "$FRONTEND_READY" -ne 1 ]]; then
  echo "❌ Frontend or frontend API proxy did not respond on port 3000 after 90 seconds."
  echo "   Run: make doctor"
  exit 1
fi

echo
echo "✅ FlyGPT frontend and API proxy ready on port 3000"
if [[ "$USE_OLLAMA" -eq 1 ]]; then
  echo "   Generator: Ollama :11434 / $MODEL"
else
  echo "   Generator: ${FLYGPT_GENERATOR_PROVIDER:-compatible-http} / ${FLYGPT_GENERATOR_MODEL:-unset}"
fi
echo "   Backend  :8000"
echo "   Frontend :3000"
if [[ -n "${CODESPACE_NAME:-}" ]]; then
  echo "   Codespaces preview: https://${CODESPACE_NAME}-3000.${GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN:-app.github.dev}/"
  echo "   If this external link shows 404 while localhost returns 200, run 'make doctor'"
  echo "   and open the CURRENT 3000 forwarded URL from the VS Code PORTS tab."
fi
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
