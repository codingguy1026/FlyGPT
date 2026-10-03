#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
TRAIN_ENV=".venv-train"

if [[ ! -x "$TRAIN_ENV/bin/python" ]]; then
  echo "🏗️ Creating FlyGPT training environment at $TRAIN_ENV..."
  "$PYTHON_BIN" -m venv "$TRAIN_ENV"
fi

if ! "$TRAIN_ENV/bin/python" -c "import torch" >/dev/null 2>&1; then
  echo "📦 Installing CPU-only PyTorch for router training..."
  "$TRAIN_ENV/bin/python" -m pip install --no-cache-dir     --index-url https://download.pytorch.org/whl/cpu     "torch>=2.4,<3"
fi

"$TRAIN_ENV/bin/python" - <<'PY'
import torch
print(f"✅ training torch={torch.__version__} cuda={torch.cuda.is_available()}")
PY
