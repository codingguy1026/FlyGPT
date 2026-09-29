#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SCAFFOLD="training/flywire_scaffold.json"
BASE="training/teacher_seed.jsonl"
CURR34="training/curriculum_v0_3_4.jsonl"
CURR35="training/curriculum_v0_3_5_short.jsonl"
CURR36="training/curriculum_v0_3_6_context.jsonl"
CURR37="training/curriculum_v0_3_7_yugeok.jsonl"
MERGED="/tmp/flygpt_teacher_v0_3_7.jsonl"

OLD="artifacts/fly_router_v0_3_6.pt"
OUT="artifacts/fly_router_v0_3_7.pt"

YUGEOK="training/regression_v0_3_7_yugeok.jsonl"
CONTEXT="training/regression_v0_3_6_context.jsonl"
SHORT="training/regression_v0_3_5_short.jsonl"
CORE="training/regression_v0_3_4_core.jsonl"
PRIOR="training/regression_v0_3_1.jsonl"
GRATITUDE="training/regression_v0_3_2_gratitude.jsonl"

cat "$BASE" "$CURR34" "$CURR35" "$CURR36" "$CURR37" > "$MERGED"

echo "=== v0.3.7 curriculum size ==="
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
p = Path("/tmp/flygpt_teacher_v0_3_7.jsonl")
rows = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
counts = Counter(row["route"] for row in rows)
print(f"examples={len(rows)}")
for route, count in sorted(counts.items()):
    print(f"{route}: {count}")
PY

if [[ ! -f "$SCAFFOLD" ]]; then
  echo
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
  echo "=== v0.3.6 baseline on yugeok regression ==="
  python training/eval_router.py \
    --model "$OLD" \
    --dataset "$YUGEOK"
fi

echo
echo "=== training v0.3.7: adversarial intent boundaries ==="
python training/train_router.py \
  --dataset "$MERGED" \
  --scaffold "$SCAFFOLD" \
  --out "$OUT" \
  --epochs 140 \
  --vectorizer-version v5 \
  --vocab-size 8192

echo
echo "=== v0.3.7 brain integrity check ==="
python training/diagnose_brain.py \
  --model "$OUT" \
  --scaffold "$SCAFFOLD"

for DATASET in "$YUGEOK" "$CONTEXT" "$SHORT" "$CORE" "$PRIOR" "$GRATITUDE"; do
  echo
  echo "=== regression: $DATASET ==="
  python training/eval_router.py --model "$OUT" --dataset "$DATASET"
done
