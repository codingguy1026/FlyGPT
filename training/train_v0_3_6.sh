#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SCAFFOLD="training/malecns_scaffold.json"
BASE="training/teacher_seed.jsonl"
CURR34="training/curriculum_v0_3_4.jsonl"
CURR35="training/curriculum_v0_3_5_short.jsonl"
CURR36="training/curriculum_v0_3_6_context.jsonl"
MERGED="/tmp/flygpt_teacher_v0_3_6.jsonl"

OLD="artifacts/fly_router_v0_3_5.pt"
OUT="artifacts/fly_router_v0_3_6.pt"

CONTEXT="training/regression_v0_3_6_context.jsonl"
SHORT="training/regression_v0_3_5_short.jsonl"
CORE="training/regression_v0_3_4_core.jsonl"
PRIOR="training/regression_v0_3_1.jsonl"
GRATITUDE="training/regression_v0_3_2_gratitude.jsonl"

cat "$BASE" "$CURR34" "$CURR35" "$CURR36" > "$MERGED"

echo "=== v0.3.6 curriculum size ==="
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
p = Path("/tmp/flygpt_teacher_v0_3_6.jsonl")
rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
counts = Counter(row["route"] for row in rows)
print(f"examples={len(rows)}")
for route, count in sorted(counts.items()):
    print(f"{route}: {count}")
PY

if [[ ! -f "$SCAFFOLD" ]]; then
  echo
  echo "No scaffold found; building one from Janelia MaleCNS v1.0 via neuPrint..."
  python training/build_scaffold.py \
    --data-dir data/flywire_parts \
    --out "$SCAFFOLD" \
    --nodes 256 \
    --edges 4096 \
    --sample-rows 200000
fi

if [[ -f "$OLD" ]]; then
  echo
  echo "=== v0.3.5 baseline on contextual regression ==="
  python training/eval_router.py \
    --model "$OLD" \
    --dataset "$CONTEXT"
fi

echo
echo "=== training v0.3.6: contextual word-pair routing ==="
python training/train_router.py \
  --dataset "$MERGED" \
  --scaffold "$SCAFFOLD" \
  --out "$OUT" \
  --epochs 120 \
  --vectorizer-version v5 \
  --vocab-size 8192

echo
echo "=== v0.3.6 brain integrity check ==="
python training/diagnose_brain.py \
  --model "$OUT" \
  --scaffold "$SCAFFOLD"

echo
echo "=== v0.3.6 contextual held-out regression ==="
python training/eval_router.py \
  --model "$OUT" \
  --dataset "$CONTEXT"

echo
echo "=== v0.3.6 short-form regression ==="
python training/eval_router.py \
  --model "$OUT" \
  --dataset "$SHORT"

echo
echo "=== v0.3.6 broad core regression ==="
python training/eval_router.py \
  --model "$OUT" \
  --dataset "$CORE"

echo
echo "=== v0.3.6 prior regression ==="
python training/eval_router.py \
  --model "$OUT" \
  --dataset "$PRIOR"

echo
echo "=== v0.3.6 gratitude regression ==="
python training/eval_router.py \
  --model "$OUT" \
  --dataset "$GRATITUDE"
