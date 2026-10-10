.PHONY: run local reclaim disk train-setup train-v0.4 train-v0.5 help stop

run:
	@bash scripts/run_all.sh

local:
	@bash scripts/run_all.sh --local-only

train-setup:
	@bash scripts/setup_train_env.sh

train-v0.4:
	@bash training/train_malecns_v0_4.sh

train-v0.5:
	@bash training/train_malecns_v0_5.sh

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
	@echo "  make train-setup Create isolated CPU PyTorch training environment"
	@echo "  make train-v0.4  Build MaleCNS scaffold and train/evaluate router v0.4"
	@echo "  make train-v0.5  Train calibrated v0.5 router on the frozen MaleCNS scaffold"
	@echo "  make reclaim     Clear safe rebuildable caches and obsolete FlyWire data"
	@echo "  make disk        Show disk usage for common large FlyGPT paths"
	@echo "  Ctrl+C           Stop the running stack"

stop:
	@echo "FlyGPT is managed by make run; press Ctrl+C in that terminal to stop it."
