#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

SCAFFOLD="training/flywire_scaffold.json"
BASE="training/teacher_seed.jsonl"
CURRICULUM="training/curriculum_v0_3_4.jsonl"
MERGED="/tmp/flygpt_teacher_v0_3_4.jsonl"
OLD="artifacts/fly_router_v0_3_3.pt"
OUT="artifacts/fly_router_v0_3_4.pt"
CORE="training/regression_v0_3_4_core.jsonl"
PRIOR="training/regression_v0_3_1.jsonl"
GRATITUDE="training/regression_v0_3_2_gratitude.jsonl"

cat "$BASE" "$CURRICULUM" > "$MERGED"

echo "=== v0.3.4 curriculum size ==="
python - <<'PY'
import json
from collections import Counter
from pathlib import Path
p = Path("/tmp/flygpt_teacher_v0_3_4.jsonl")
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
  echo "=== v0.3.3 baseline on new core regression ==="
  python training/eval_router.py \
    --model "$OLD" \
    --dataset "$CORE"
fi

echo
echo "=== training v0.3.4: broad conversation curriculum ==="
python training/train_router.py \
  --dataset "$MERGED" \
  --scaffold "$SCAFFOLD" \
  --out "$OUT" \
  --epochs 120 \
  --vectorizer-version v3

echo
echo "=== v0.3.4 brain integrity check ==="
python training/diagnose_brain.py \
  --model "$OUT" \
  --scaffold "$SCAFFOLD"

echo
echo "=== v0.3.4 core held-out regression ==="
python training/eval_router.py \
  --model "$OUT" \
  --dataset "$CORE"

echo
echo "=== v0.3.4 prior held-out regression ==="
python training/eval_router.py \
  --model "$OUT" \
  --dataset "$PRIOR"

echo
echo "=== v0.3.4 gratitude regression ==="
python training/eval_router.py \
  --model "$OUT" \
  --dataset "$GRATITUDE"
