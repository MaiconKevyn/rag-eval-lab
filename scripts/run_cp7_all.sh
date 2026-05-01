#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
BENCHMARK_PATH="${1:-data/benchmark/benchmark_v1_157d1689_2026-04-25.json}"
MAX_QUESTIONS="${MAX_QUESTIONS:-150}"

CONFIGS=(
  "configs/exp_001_llamaindex.yaml"
  "configs/exp_002_llamaindex.yaml"
  "configs/exp_003_llamaindex.yaml"
  "configs/exp_004_llamaindex.yaml"
)

for config in "${CONFIGS[@]}"; do
  echo ""
  echo "==> Running CP7 for ${config}"
  "${PYTHON_BIN}" scripts/run_llamaindex_experiment.py \
    --config "${config}" \
    --benchmark "${BENCHMARK_PATH}" \
    --max-questions "${MAX_QUESTIONS}" \
    --force
done
