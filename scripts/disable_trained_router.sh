#!/usr/bin/env bash
# Reversible opt-out without deleting the trained file or exposing .env secrets.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 - <<'PY'
from pathlib import Path
import json
import os

path = Path(".env")
marker = Path("data/trained_router_previous_env.json")
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
keys = ("FLYGPT_MODEL_PATH", "FLYGPT_ALLOW_RESEARCH_ROUTER")
removed = []
preserved = []
for line in lines:
    name = line.strip().removeprefix("export ").split("=", 1)[0].strip()
    (removed if name in keys else preserved).append(line)
if marker.exists():
    saved = json.loads(marker.read_text(encoding="utf-8"))
    preserved.extend(saved["previous_lines"])
    marker.unlink()
tmp = path.with_suffix(path.suffix + ".tmp")
tmp.write_text("\n".join(preserved) + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
os.replace(tmp, path)
PY
echo "✅ Restored previous router settings. Restart backend to use the prior model."
