#!/usr/bin/env bash
# Enable v0.5 full MaleCNS category-free policy as the MAIN ordinary-chat
# router. No Ollama/Qwen/local language model is downloaded or started.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

MODEL="artifacts/papiti_v0_5_category_free_166700.pt"
RUN_ID="43168156480"
ARTIFACT="papiti-v0-5-category-free-166700-experiment"
SELECTED="papiti-v0.5-category-free-166700.pt"
SHA256="75bfe6703e5eccc0d13b5ef030ddb631addbdfe4be81470375c72ce32615e28f"

if [[ ! -s "$MODEL" ]]; then
  if ! command -v gh >/dev/null 2>&1; then
    echo "❌ Install the checkpoint manually or run in authenticated GitHub Codespaces." >&2
    exit 1
  fi
  temp="$(mktemp -d)"
  trap 'rm -rf "$temp"' EXIT
  echo "📥 Downloading verified v0.5 model from successful training run..."
  gh run download "$RUN_ID" -R codingguy1026/FlyGPT -n "$ARTIFACT" -D "$temp"
  mkdir -p artifacts
  install -m 600 "$temp/$SELECTED" "$MODEL"
fi

echo "$SHA256  $MODEL" | sha256sum -c -

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "❌ .venv/bin/python missing. Create the normal FlyGPT environment first." >&2
  exit 1
fi
if ! "$PY" -c 'import torch' >/dev/null 2>&1; then
  echo "📦 Installing CPU PyTorch (library only, NOT a language model)."
  "$PY" -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
fi

"$PY" - <<'PY'
from papiti_v05_runtime import PapitiV05PolicyRuntime
model = PapitiV05PolicyRuntime(
    "artifacts/papiti_v0_5_category_free_166700.pt"
)
result = model.decide("사과와 바나나의 공통점이 뭐야?")
assert model.n_neurons == 166700
assert result["contract"] == "papiti_v0_5_tool_policy_v1"
assert isinstance(result["selected_capabilities"], list)
print("✅ Verified 166,700 MaleCNS neurons and category-free policy inference.")
PY

python3 - <<'PY'
import json
import os
from pathlib import Path

env = Path(".env")
previous_file = Path("data/papiti_v05_previous_env.json")
keys = {
    "FLYGPT_V05_ACTIVE",
    "FLYGPT_V05_MODEL_PATH",
    "FLYGPT_GENERATOR_MODE",
    "FLYGPT_LOCAL_ONLY",
}
lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
def line_key(line):
    return line.lstrip().removeprefix("export ").split("=", 1)[0].strip()

previous = [line for line in lines if line_key(line) in keys]
if not previous_file.exists():
    previous_file.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(previous_file, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as fp:
        json.dump({"previous": previous}, fp, ensure_ascii=False)
kept = [line for line in lines if line_key(line) not in keys]
kept += [
    "FLYGPT_V05_ACTIVE=1",
    "FLYGPT_V05_MODEL_PATH=artifacts/papiti_v0_5_category_free_166700.pt",
    "FLYGPT_GENERATOR_MODE=http",
    "FLYGPT_LOCAL_ONLY=0",
]
temp = env.with_name(".env.papiti-v05.tmp")
temp.write_text("\n".join(kept) + "\n", encoding="utf-8")
os.chmod(temp, 0o600)
os.replace(temp, env)
PY

echo "🪰 Main chat router: MaleCNS v0.5 (category-free, 166700 neurons)"
echo "☁️  External API mode selected; NO local language-model installation."
if ! grep -Eq '^[[:space:]]*(export[[:space:]]+)?GEMINI_API_KEY=.+$' .env; then
  echo "⚠️ Gemini key missing. Add GEMINI_API_KEY=... privately to .env for natural answers."
fi
echo "✅ Restart server with 'make run'. Rollback with 'make disable-papiti-v05'."
