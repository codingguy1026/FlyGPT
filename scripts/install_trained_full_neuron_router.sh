#!/usr/bin/env bash
# Install the REAL trained 166,700-neuron MaleCNS router from a verified
# GitHub Actions artifact. Never infer one model's accuracy from a 3-seed mean.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODEL="artifacts/fly_router_malecns_all_neurons.pt"
RUN_ID="38915242351"
ARTIFACT="real-full-MaleCNS-3-seed-models"
# Seed 44 was selected using INTERNAL VALIDATION (highest of 3), not test labels.
SELECTED="malecns_all_neurons_seed44.pt"
PINNED_SHA256="0a410dd31fc6cf682af2de1f79b2f7f981d218d4154290e282b9c28957318901"
if ! command -v gh >/dev/null 2>&1; then
  echo "❌ GitHub CLI gh is required; use GitHub Codespaces." >&2
  exit 1
fi
if ! gh auth status -h github.com >/dev/null 2>&1; then
  echo "❌ Authenticate GitHub CLI before downloading model artifact." >&2
  exit 1
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "📥 Retrieving authentic full-neuron model from the successful run…"
gh run download "$RUN_ID" -R codingguy1026/FlyGPT -n "$ARTIFACT" -D "$tmp"
echo "$PINNED_SHA256  $tmp/$SELECTED" | sha256sum -c -
mkdir -p artifacts
cp "$tmp/$SELECTED" "$MODEL"
echo "✅ Full-neuron model integrity verified. Preparing optional CPU PyTorch…" 
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
if ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
  .venv/bin/python -m pip install 'torch>=2.4,<3' --index-url https://download.pytorch.org/whl/cpu
fi
# This activates only after the FULL model was loaded and verified.
bash scripts/enable_full_neuron_router.sh
echo "🪰 Ready. Restart the FlyGPT backend (make stable or make run)."
echo "    Previous model: make disable-full-neuron-router"
