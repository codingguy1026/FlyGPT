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
OUT="artifacts/fly_router_malecns_v0_4.pt"
MERGED="/tmp/flygpt_teacher_malecns_v0_4.jsonl"

BASE="training/teacher_seed.jsonl"
CURR34="training/curriculum_v0_3_4.jsonl"
CURR35="training/curriculum_v0_3_5_short.jsonl"
CURR36="training/curriculum_v0_3_6_context.jsonl"
CURR37="training/curriculum_v0_3_7_yugeok.jsonl"
CURR38="training/curriculum_v0_3_8_hard_negatives.jsonl"
CURR39="training/curriculum_v0_3_9.jsonl"

PARA="training/regression_v0_3_9_paraphrase.jsonl"
HARD="training/regression_v0_3_8_hard_negatives.jsonl"
YUGEOK="training/regression_v0_3_7_yugeok.jsonl"
CONTEXT="training/regression_v0_3_6_context.jsonl"
SHORT="training/regression_v0_3_5_short.jsonl"
CORE="training/regression_v0_3_4_core.jsonl"
PRIOR="training/regression_v0_3_1.jsonl"
GRATITUDE="training/regression_v0_3_2_gratitude.jsonl"

if [[ ! -x "$RUNTIME_PYTHON" ]]; then
  echo "❌ Runtime environment not found: $RUNTIME_PYTHON"
  echo "Run: make run"
  exit 1
fi

if [[ ! -x "$TRAIN_PYTHON" ]] || ! "$TRAIN_PYTHON" -c "import torch" >/dev/null 2>&1; then
  echo "❌ Training environment is not ready."
  echo "Run: make train-setup"
  exit 1
fi

if [[ -z "${NEUPRINT_TOKEN:-}" && -z "${NEUPRINT_APPLICATION_CREDENTIALS:-}" ]]; then
  echo "❌ neuPrint credentials are not configured."
  echo "Add NEUPRINT_TOKEN to .env, then rerun."
  exit 1
fi

echo "🧠 Building fresh MaleCNS v1.0 scaffold..."
"$RUNTIME_PYTHON" training/build_scaffold.py   --out "$SCAFFOLD"   --nodes 256   --edges 4096   --candidate-edges 32768

cat "$BASE" "$CURR34" "$CURR35" "$CURR36" "$CURR37" "$CURR38" "$CURR39" > "$MERGED"

echo
echo "🎓 Training FlyGPT MaleCNS router v0.4..."
"$TRAIN_PYTHON" training/train_router.py   --dataset "$MERGED"   --scaffold "$SCAFFOLD"   --out "$OUT"   --epochs 160   --patience 15   --vectorizer-version v5   --vocab-size 8192

echo
echo "🩺 Checking MaleCNS brain integrity..."
"$TRAIN_PYTHON" training/diagnose_brain.py   --model "$OUT"   --scaffold "$SCAFFOLD"

for DATASET in "$PARA" "$HARD" "$YUGEOK" "$CONTEXT" "$SHORT" "$CORE" "$PRIOR" "$GRATITUDE"; do
  echo
  echo "=== regression: $DATASET ==="
  "$TRAIN_PYTHON" training/eval_router.py --model "$OUT" --dataset "$DATASET"
done

echo
echo "✅ MaleCNS v0.4 training complete: $OUT"
echo "💾 Keep the checkpoint this time:"
echo "   git add $OUT $SCAFFOLD"
echo "   git commit -m 'model: add MaleCNS v0.4 router checkpoint'"
