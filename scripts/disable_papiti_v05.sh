#!/usr/bin/env bash
# Restore the previous chat/router-generator switch values only.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 - <<'PY'
import json
import os
from pathlib import Path

env = Path(".env")
previous_file = Path("data/papiti_v05_previous_env.json")
keys = {
    "FLYGPT_V05_ACTIVE", "FLYGPT_V05_MODEL_PATH",
    "FLYGPT_GENERATOR_MODE", "FLYGPT_LOCAL_ONLY",
}
lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
def line_key(line):
    return line.lstrip().removeprefix("export ").split("=", 1)[0].strip()

kept = [line for line in lines if line_key(line) not in keys]
previous = []
if previous_file.is_file():
    previous = json.loads(previous_file.read_text(encoding="utf-8")).get("previous", [])
kept.extend(previous)
temp = env.with_name(".env.papiti-v05.tmp")
temp.write_text("\n".join(kept) + "\n", encoding="utf-8")
os.chmod(temp, 0o600)
os.replace(temp, env)
if previous_file.is_file():
    previous_file.unlink()
PY
echo "✅ Papiti v0.5 main router disabled; previous generator mode restored."
echo "Restart with make run to use the legacy router again."
