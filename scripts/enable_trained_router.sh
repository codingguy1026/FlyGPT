#!/usr/bin/env bash
# Explicit opt-in installation of the already trained MaleCNS research checkpoint.
# The model is selected by INTERNAL validation (seed 44), not the public test
# score. This script does not touch source control or a production deployment.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
MODEL="artifacts/fly_router_malecns_residual_seed44.pt"
PINNED_SHA="06642d649163b6636ed3b70069a9fafcc7fcc91c7fa37850a35ba82df2cb4c46"
RUN_ID="38858319727"
ARTIFACT="flygpt-research-only-trained-malecns"

if ! command -v gh >/dev/null 2>&1; then
  echo "❌ GitHub CLI 'gh' is required (already provided in GitHub Codespaces)."
  exit 1
fi
if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "❌ Please authenticate GitHub CLI in your Codespace (gh auth login)."
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "🪰 Downloading the original training archive from GitHub Actions…"
gh run download "$RUN_ID" -R codingguy1026/FlyGPT \
  -n "$ARTIFACT" -D "$tmp" >/dev/null
printf '%s  %s\n' "$PINNED_SHA" "$tmp/male_cns_seed44.pt" | sha256sum -c -

# Keep the old checkpoint untouched. Install next to it.
mkdir -p artifacts
install -m 644 "$tmp/male_cns_seed44.pt" "$MODEL"

# Prefer existing runtime for checks. Isolated training venv is a valid backup.
check_python=""
for candidate in .venv/bin/python .venv-train/bin/python python3; do
  if "$candidate" -c "import torch" >/dev/null 2>&1; then
    check_python="$candidate"
    break
  fi
done
if [[ -z "$check_python" ]]; then
  echo "❌ Install CPU-only PyTorch in the FlyGPT runtime (make train-setup)."
  exit 1
fi

echo "🔬 Verifying the exact MaleCNS graph and trained model inference…"
FLYGPT_ALLOW_RESEARCH_ROUTER=1 "$check_python" - <<'PY'
from router_runtime import FlyRouterRuntime
router = FlyRouterRuntime("artifacts/fly_router_malecns_residual_seed44.pt")
result = router.predict("파이썬으로 웹페이지를 만드는 코드 예제를 작성해줘", top_k=3)
assert router.residual_research is True
assert router.n_nodes == 256 and router.src.numel() == 948
assert len(result["trace"]) == 5  # input, three graph steps, blended readout
assert result["trace"][-1]["stage"] == "residual_blend"
assert len(result["top_routes"]) == 3
print("✅ Integrity and inference passed. Predicted route:", result["route"])
PY

# Runtime .venv (FastAPI) needs PyTorch, not just .venv-train.
if [[ ! -x .venv/bin/python ]]; then
  echo "🐍 Creating the FastAPI runtime environment…"
  python3 -m venv .venv
fi
if ! .venv/bin/python -c "import torch" >/dev/null 2>&1; then
  echo "📦 Installing CPU-only PyTorch for the website backend…"
  .venv/bin/python -m pip install "torch>=2.4,<3" \
    --index-url https://download.pytorch.org/whl/cpu
fi

# Preserve every unrelated .env setting; store only the previous two keys in
# an ignored local rollback marker. Never print .env or credentials.
python3 - <<'PY'
from pathlib import Path
import json
import os

path = Path(".env")
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
keys = ("FLYGPT_MODEL_PATH", "FLYGPT_ALLOW_RESEARCH_ROUTER")
before = [
    line for line in lines
    if line.lstrip().startswith(tuple("export " + x + "=" for x in keys))
    or line.lstrip().startswith(tuple(x + "=" for x in keys))
]
marker = Path("data/trained_router_previous_env.json")
marker.parent.mkdir(parents=True, exist_ok=True)
if not marker.exists():
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"previous_lines": before}, f)
filtered = [line for line in lines if line not in before]
filtered.extend((
    "export FLYGPT_ALLOW_RESEARCH_ROUTER=1",
    "export FLYGPT_MODEL_PATH=artifacts/fly_router_malecns_residual_seed44.pt",
))
temp = path.with_suffix(path.suffix + ".tmp")
temp.write_text("\n".join(filtered) + "\n", encoding="utf-8")
os.chmod(temp, 0o600)
os.replace(temp, path)
PY
echo "✅ Selected seed-44 trained MaleCNS checkpoint for NEXT FlyGPT backend start."
echo "   Restart the running backend (make stable, or make run) to activate."
echo "   Roll back with: make disable-trained-router"
