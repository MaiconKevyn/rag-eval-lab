#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
RUNS=(
  "data/runs/exp_001_llamaindex/run_results.json"
  "data/runs/exp_002_llamaindex/run_results.json"
  "data/runs/exp_003_llamaindex/run_results.json"
  "data/runs/exp_004_llamaindex/run_results.json"
)

for run in "${RUNS[@]}"; do
  echo ""
  echo "==> Evaluating ${run}"
  "${PYTHON_BIN}" scripts/evaluate_run.py --run "${run}" --batch --force
done

echo ""
echo "==> Logging all experiments to MLflow"
"${PYTHON_BIN}" scripts/log_to_mlflow.py --tracking-uri sqlite:///mlflow.db

echo ""
echo "==> Generating framework comparison report"
"${PYTHON_BIN}" scripts/generate_report.py
