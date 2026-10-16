.PHONY: run local stable doctor stability reclaim disk train-setup train-v0.4 train-v0.5 train-role-experiment compare-role-experiment train-full-structural enable-trained-router disable-trained-router download-full-neuron-data train-full-neuron-router enable-full-neuron-router disable-full-neuron-router help stop

run:
	@bash scripts/run_all.sh

local:
	@bash scripts/run_all.sh --local-only

stable:
	@bash scripts/run_all.sh --stable

doctor:
	@python3 scripts/diagnose_codespaces.py

stability:
	@python3 -m scripts.check_stability

train-setup:
	@bash scripts/setup_train_env.sh

train-v0.4:
	@bash training/train_malecns_v0_4.sh

train-v0.5:
	@bash training/train_malecns_v0_5.sh

# Offline experiment only. Never changes the app's default router checkpoint.
train-role-experiment:
	@ROLE_EXPERIMENT=1 bash training/train_malecns_v0_5.sh

compare-role-experiment:
	@.venv-train/bin/python training/compare_role_aware.py --baseline artifacts/fly_router_malecns_v0_5.pt --candidate artifacts/fly_router_malecns_role_experiment.pt

# Download archived, checksum-pinned MaleCNS residual checkpoint and opt in locally.
# No deployment is triggered; restart the backend after either command.
enable-trained-router:
	@bash scripts/enable_trained_router.sh

disable-trained-router:
	@bash scripts/disable_trained_router.sh

# All-neuron mode is offline and opt-in. The official graph is >1GB;
# never download or train it during /api/chat requests or server boot.
download-full-neuron-data:
	@bash scripts/download_all_malecns_data.sh

train-full-neuron-router:
	@bash scripts/train_full_population_router.sh

enable-full-neuron-router:
	@bash scripts/enable_full_neuron_router.sh

disable-full-neuron-router:
	@bash scripts/disable_full_neuron_router.sh

# Explicit offline command. Never run the full ~1 GB source scan on app startup.
train-full-structural:
	@.venv-train/bin/python training/train_full_connectome.py --epochs 1

reclaim:
	@echo "🧹 Reclaiming safe FlyGPT development space..."
	@rm -rf frontend/.next
	@rm -rf data/flywire_parts
	@rm -f data/proofread_connections_783.feather
	@rm -rf "$$HOME/.cache/pip"
	@echo "✅ Cleared Next.js build cache, obsolete FlyWire data, and pip cache."
	@echo "Tip: run 'make disk' to inspect the largest remaining directories."

disk:
	@echo "Disk free:"
	@df -h .
	@echo
	@echo "FlyGPT disk usage:"
	@du -sh .venv .venv-train frontend/node_modules frontend/.next data training/malecns_scaffold.json artifacts "$$HOME/.cache/pip" "$$HOME/.ollama" 2>/dev/null || true

help:
	@echo "FlyGPT development commands"
	@echo "  make run         Start configured generator, backend, and frontend"
	@echo "  make local       Strictly local LLM via Ollama (ignore hosted API keys)"
	@echo "  make stable      Build + serve production frontend without backend hot reload"
	@echo "  make doctor      Diagnose frontend, backend, proxy and Codespaces URL"
	@echo "  make stability   Sample frontend/backend/proxy six times for intermittent failures"
	@echo "  make train-setup Create isolated CPU PyTorch training environment"
	@echo "  make train-v0.4  Build MaleCNS scaffold and train/evaluate router v0.4"
	@echo "  make train-v0.5  Train calibrated v0.5 router on the frozen MaleCNS scaffold"
	@echo "  make train-role-experiment  Train isolated neuron-role-aware candidate (needs neuPrint)"
	@echo "  make compare-role-experiment  Compare candidate and baseline on held-out regression sets"
	@echo "  make enable-trained-router  Verify/download 473-example-trained residual MaleCNS checkpoint"
	@echo "  make disable-trained-router Restore previous local router settings"
	@echo "  make download-full-neuron-data Download official 1.1GB full MaleCNS graph (explicit only)"
	@echo "  make train-full-neuron-router Train exact whole-population structural routing model offline"
	@echo "  make enable-full-neuron-router Verify+select the all-neuron model for next site startup"
	@echo "  make disable-full-neuron-router Return to the previous router model"
	@echo "  make train-full-structural  Run one full offline MaleCNS structural training epoch"
	@echo "  make reclaim     Clear safe rebuildable caches and obsolete FlyWire data"
	@echo "  make disk        Show disk usage for common large FlyGPT paths"
	@echo "  Ctrl+C           Stop the running stack"

stop:
	@echo "FlyGPT is managed by make run; press Ctrl+C in that terminal to stop it."
