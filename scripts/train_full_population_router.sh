#!/usr/bin/env bash
# Offline two-stage routing training on entire annotated MaleCNS graph.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [[ ! -s "data/malecns_full/connectome-weights-male-cns-v1.0-minconf-0.5.feather" ]] ||
   [[ ! -s "data/malecns_full/body-annotations-male-cns-v1.0-minconf-0.5.feather" ]]; then
  echo "❌ Missing full-graph dataset. Run: make download-full-neuron-data"
  exit 1
fi
if [[ ! -x .venv-train/bin/python ]]; then
  echo "❌ Missing training venv. Run: make train-setup"
  exit 1
fi
.venv-train/bin/python -c 'import torch, pyarrow, numpy' >/dev/null 2>&1 || {
  echo "❌ Install pyarrow: .venv-train/bin/python -m pip install -r requirements-full-connectome.txt"
  exit 1
}
echo "🧠 Phase 1: pretrain embeddings from all annotated neurons and directed connections"
.venv-train/bin/python training/train_full_connectome.py \
  --epochs "${FLYGPT_FULL_STRUCTURE_EPOCHS:-1}" \
  --embedding-dim 16 \
  --batch-size 32768 \
  --out artifacts/malecns_full_structural_embeddings.pt
echo "🪰 Phase 2: train route classifier using all structural neuron embeddings"
.venv-train/bin/python training/full_population_router.py \
  --structural artifacts/malecns_full_structural_embeddings.pt \
  --epochs "${FLYGPT_FULL_ROUTER_EPOCHS:-80}" \
  --out artifacts/fly_router_malecns_all_neurons.pt
echo "✅ Whole-neuron research checkpoint created. NOT deployed."
