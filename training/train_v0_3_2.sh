#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SCAFFOLD="training/flywire_scaffold.json"
OLD="artifacts/fly_router_v0_3_1.pt"
OUT="artifacts/fly_router_v0_3_2.pt"
GRATITUDE="training/regression_v0_3_2_gratitude.jsonl"
FULL="training/regression_v0_3_1.jsonl"

if [[ ! -f "$SCAFFOLD" ]]; then
  echo "No scaffold found; building one from local FlyWire parts..."
  python training/build_scaffold.py \
    --data-dir data/flywire_parts \
    --out "$SCAFFOLD" \
    --nodes 256 \
    --edges 4096 \
    --sample-rows 200000
fi

if [[ -f "$OLD" ]]; then
  echo
  echo "=== v0.3.1 gratitude baseline ==="
  python training/eval_router.py \
    --model "$OLD" \
    --dataset "$GRATITUDE"
fi

echo
echo "=== training v0.3.2: gratitude patch ==="
python training/train_router.py \
  --dataset training/teacher_seed.jsonl \
  --scaffold "$SCAFFOLD" \
  --out "$OUT" \
  --epochs 100 \
  --vectorizer-version v2

echo
echo "=== v0.3.2 brain integrity check ==="
python training/diagnose_brain.py \
  --model "$OUT" \
  --scaffold "$SCAFFOLD"

echo
echo "=== v0.3.2 gratitude regression ==="
python training/eval_router.py \
  --model "$OUT" \
  --dataset "$GRATITUDE"

echo
echo "=== v0.3.2 full held-out regression ==="
python training/eval_router.py \
  --model "$OUT" \
  --dataset "$FULL"
