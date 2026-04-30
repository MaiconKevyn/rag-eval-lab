#!/usr/bin/env bash
# Run CP3 (ingest + RAG) for all 4 experiments in parallel.
# Each experiment logs to logs/<exp_id>.log.
#
# Usage:
#   bash scripts/run_cp3_all.sh [benchmark_path]
set -uo pipefail

BENCHMARK_PATH="${1:-$(ls -t data/benchmark/benchmark_v1_*.json 2>/dev/null | head -1)}"
MAX_QUESTIONS="${MAX_QUESTIONS:-150}"

if [[ -z "$BENCHMARK_PATH" ]]; then
  echo "ERROR: No benchmark found in data/benchmark/. Run generate_benchmark.py first." >&2
  exit 1
fi

echo "Using benchmark: $BENCHMARK_PATH"
mkdir -p logs

configs=(
  "configs/exp_001_chunk256_ada.yaml"
  "configs/exp_002_chunk512_ada.yaml"
  "configs/exp_003_chunk128_topk10.yaml"
  "configs/exp_004_chunk256_topk3.yaml"
)

pids=()
exp_names=()

for config in "${configs[@]}"; do
  exp_name=$(basename "$config" .yaml)
  logfile="logs/${exp_name}_cp3.log"
  exp_names+=("$exp_name")

  echo "==> Starting $exp_name  →  $logfile"
  {
    echo "--- Ingest: $config ---"
    python scripts/ingest_corpus.py --config "$config" --rebuild
    echo "--- RAG run: $config ---"
    python scripts/run_experiment.py --config "$config" --benchmark "$BENCHMARK_PATH" --max-questions "$MAX_QUESTIONS" --force
  } >"$logfile" 2>&1 &
  pids+=($!)
done

echo ""
echo "All 4 experiments running in parallel. Waiting for completion..."
echo ""

failed=()
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[OK]   ${exp_names[$i]}"
  else
    echo "[FAIL] ${exp_names[$i]}  →  logs/${exp_names[$i]}_cp3.log"
    failed+=("${exp_names[$i]}")
  fi
done

echo ""
if [[ ${#failed[@]} -eq 0 ]]; then
  echo "All CP3 experiments completed successfully."
else
  echo "ERROR: ${#failed[@]} experiment(s) failed: ${failed[*]}"
  exit 1
fi
