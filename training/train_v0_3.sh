#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SCAFFOLD="training/flywire_scaffold.json"
OUT="artifacts/fly_router_v0_3.pt"

if [[ ! -f "$SCAFFOLD" ]]; then
  echo "Missing $SCAFFOLD"
  echo "Build it first with training/build_scaffold.py using your local FlyWire parts."
  exit 1
fi

python training/train_router.py \
  --dataset training/teacher_seed.jsonl \
  --scaffold "$SCAFFOLD" \
  --out "$OUT" \
  --epochs 100

python training/eval_router.py \
  --model "$OUT" \
  --dataset training/regression_v0_3.jsonl
