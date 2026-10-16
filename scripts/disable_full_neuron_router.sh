#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 - <<'PY'
from pathlib import Path
import json, os
path = Path(".env")
marker = Path("data/full_neuron_previous_env.json")
keys = {"FLYGPT_MODEL_PATH","FLYGPT_ENABLE_FULL_POPULATION_ROUTER","FLYGPT_ALLOW_RESEARCH_ROUTER"}
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
def key(line): return line.lstrip().removeprefix("export ").split("=", 1)[0].strip()
keep = [s for s in lines if key(s) not in keys]
if marker.exists():
    before = json.loads(marker.read_text(encoding="utf-8"))["previous"]
    keep.extend(before)
    marker.unlink()
temp = path.with_suffix(path.suffix + ".tmp")
temp.write_text("\n".join(keep) + "\n", encoding="utf-8")
os.chmod(temp, 0o600)
os.replace(temp, path)
PY
echo "✅ Previous router settings restored; restart the backend to apply."
