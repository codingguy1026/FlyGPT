#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

echo "=== v0.4.0 generator integration tests ==="
python -m unittest   tests/test_dispatcher.py   tests/test_generator_runtime.py   tests/test_memory_store.py   tests/test_math_fast_path.py

echo
echo "=== canned handler tombstone check ==="
if grep -nE 'def _code_handler|def _general_handler|def _summarize_handler|def _pending_handler' dispatcher.py; then
  echo "FAIL: legacy canned handlers still exist"
  exit 1
fi

echo "PASS: legacy canned response handlers are gone"

echo
echo "=== compile check ==="
python -m py_compile dispatcher.py generator_runtime.py app.py memory_store.py

echo
echo "RESULT: v0.4.0 pipeline checks passed"
