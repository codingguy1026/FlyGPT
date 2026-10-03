.PHONY: run stop help

run:
	@bash scripts/run_all.sh

help:
	@echo "FlyGPT development commands"
	@echo "  make run   Start Ollama, backend, and frontend"
	@echo "  Ctrl+C     Stop the stack"

stop:
	@echo "FlyGPT is managed by make run; press Ctrl+C in that terminal to stop it."
