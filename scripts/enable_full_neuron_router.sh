#!/usr/bin/env bash
# Enables only a VERIFIED real >100k neuron checkpoint; no silent fallback.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
MODEL="artifacts/fly_router_malecns_all_neurons.pt"
if [[ ! -s "$MODEL" ]]; then
  echo "❌ No trained full-neuron model. Run: make train-full-neuron-router"
  exit 1
fi
python_bin=""
for candidate in .venv/bin/python .venv-train/bin/python python3; do
  if "$candidate" -c 'import torch' >/dev/null 2>&1; then python_bin="$candidate"; break; fi
done
if [[ -z "$python_bin" ]]; then
  echo "❌ PyTorch is required in the FlyGPT Python runtime."
  exit 1
fi
FLYGPT_ENABLE_FULL_POPULATION_ROUTER=1 "$python_bin" - <<'PY'
from full_population_runtime import FullPopulationRouterRuntime
model = FullPopulationRouterRuntime("artifacts/fly_router_malecns_all_neurons.pt")
out = model.predict("안녕, 파피티!")
assert model.n_nodes >= 100000 and model.n_edges >= model.n_nodes
assert model.graph()["full_population_summary"] is True
assert out["all_curated_neurons_attended"] is True
print(f"✅ Model verified: {model.n_nodes:,} annotated neurons, {model.n_edges:,} eligible edge rows")
PY
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c 'import torch' >/dev/null 2>&1; then
  echo "❌ Website backend .venv lacks PyTorch. Set up the runtime first; configuration unchanged."
  exit 1
fi
python3 - <<'PY'
from pathlib import Path
import json, os
path = Path(".env")
marker = Path("data/full_neuron_previous_env.json")
keys = {"FLYGPT_MODEL_PATH","FLYGPT_ENABLE_FULL_POPULATION_ROUTER","FLYGPT_ALLOW_RESEARCH_ROUTER"}
lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
def key(line): return line.lstrip().removeprefix("export ").split("=", 1)[0].strip()
keep = [s for s in lines if key(s) not in keys]
previous = [s for s in lines if key(s) in keys]
marker.parent.mkdir(parents=True, exist_ok=True)
if not marker.exists():
    fd = os.open(marker, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as out:
        json.dump({"previous": previous}, out)
keep += ["export FLYGPT_ENABLE_FULL_POPULATION_ROUTER=1",
         "export FLYGPT_MODEL_PATH=artifacts/fly_router_malecns_all_neurons.pt"]
temp = path.with_suffix(path.suffix + ".tmp")
temp.write_text("\n".join(keep) + "\n", encoding="utf-8")
os.chmod(temp, 0o600)
os.replace(temp, path)
PY
echo "✅ All-neuron structural router selected for NEXT FlyGPT backend startup."
echo "   Restart app with make stable, or make run. Rollback: make disable-full-neuron-router"
