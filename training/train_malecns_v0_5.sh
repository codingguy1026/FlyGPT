#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

RUNTIME_PYTHON="${RUNTIME_PYTHON:-.venv/bin/python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-.venv-train/bin/python}"

SCAFFOLD="training/malecns_scaffold.json"
OUT="artifacts/fly_router_malecns_v0_5.pt"
MERGED="/tmp/flygpt_teacher_malecns_v0_5.jsonl"

BASE="training/teacher_seed.jsonl"
CURR34="training/curriculum_v0_3_4.jsonl"
CURR35="training/curriculum_v0_3_5_short.jsonl"
CURR36="training/curriculum_v0_3_6_context.jsonl"
CURR37="training/curriculum_v0_3_7_yugeok.jsonl"
CURR38="training/curriculum_v0_3_8_hard_negatives.jsonl"
CURR39="training/curriculum_v0_3_9.jsonl"
CURR50="training/curriculum_v0_5_realworld.jsonl"

REG50="training/regression_v0_5_realworld.jsonl"
PARA="training/regression_v0_3_9_paraphrase.jsonl"
HARD="training/regression_v0_3_8_hard_negatives.jsonl"
YUGEOK="training/regression_v0_3_7_yugeok.jsonl"
CONTEXT="training/regression_v0_3_6_context.jsonl"
SHORT="training/regression_v0_3_5_short.jsonl"
CORE="training/regression_v0_3_4_core.jsonl"
PRIOR="training/regression_v0_3_1.jsonl"
GRATITUDE="training/regression_v0_3_2_gratitude.jsonl"

if [[ ! -x "$TRAIN_PYTHON" ]] || ! "$TRAIN_PYTHON" -c "import torch" >/dev/null 2>&1; then
  echo "❌ Training environment is not ready."
  echo "Run: make train-setup"
  exit 1
fi

if [[ "${REFRESH_SCAFFOLD:-0}" == "1" || ! -s "$SCAFFOLD" ]]; then
  if [[ ! -x "$RUNTIME_PYTHON" ]]; then
    echo "❌ Runtime environment not found: $RUNTIME_PYTHON"
    echo "Run: make run"
    exit 1
  fi
  if [[ -z "${NEUPRINT_TOKEN:-}" && -z "${NEUPRINT_APPLICATION_CREDENTIALS:-}" ]]; then
    echo "❌ neuPrint credentials are required to build the scaffold."
    echo "Add NEUPRINT_TOKEN to .env, or reuse the committed $SCAFFOLD."
    exit 1
  fi

  echo "🧠 Building fresh MaleCNS v1.0 scaffold..."
  "$RUNTIME_PYTHON" training/build_scaffold.py \
    --out "$SCAFFOLD" \
    --nodes 256 \
    --edges 4096 \
    --candidate-edges 32768
else
  echo "🧠 Reusing frozen scaffold: $SCAFFOLD"
  echo "   Set REFRESH_SCAFFOLD=1 to rebuild it from neuPrint."
fi

cat "$BASE" "$CURR34" "$CURR35" "$CURR36" "$CURR37" "$CURR38" "$CURR39" "$CURR50" > "$MERGED"

echo
echo "🎓 Training FlyGPT MaleCNS router v0.5..."
"$TRAIN_PYTHON" training/train_router.py \
  --dataset "$MERGED" \
  --scaffold "$SCAFFOLD" \
  --out "$OUT" \
  --epochs 220 \
  --patience 30 \
  --lr 0.0015 \
  --weight-decay 0.0004 \
  --val-ratio 0.22 \
  --vectorizer-version v6 \
  --vocab-size 8192 \
  --label-smoothing 0.04 \
  --margin 0.30 \
  --margin-weight 0.10 \
  --feature-dropout 0.08 \
  --selection-metric balanced_accuracy \
  --gate-target-precision 0.97 \
  --gate-min-coverage 0.55

echo
echo "🩺 Checking MaleCNS brain integrity..."
"$TRAIN_PYTHON" training/diagnose_brain.py \
  --model "$OUT" \
  --scaffold "$SCAFFOLD"

for DATASET in "$REG50" "$PARA" "$HARD" "$YUGEOK" "$CONTEXT" "$SHORT" "$CORE" "$PRIOR" "$GRATITUDE"; do
  echo
  echo "=== regression: $DATASET ==="
  "$TRAIN_PYTHON" training/eval_router.py --model "$OUT" --dataset "$DATASET"
done

echo
echo "✅ MaleCNS v0.5 training complete: $OUT"
echo "🪰 v0.5 adds noisy-feature training, margin regularization, v6 text features,"
echo "   temperature calibration, and a learned confidence gate."
echo
echo "💾 If the regression suite looks good:"
echo "   git add $OUT $SCAFFOLD"
echo "   git commit -m 'model: add MaleCNS v0.5 calibrated router checkpoint'"
