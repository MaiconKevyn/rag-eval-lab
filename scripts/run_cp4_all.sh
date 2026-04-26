#!/usr/bin/env bash
# Run CP4 LLM-as-a-Judge evaluation for all 4 experiments in parallel.
# Each experiment logs to logs/<exp_id>_cp4.log.
#
# Usage:
#   bash scripts/run_cp4_all.sh
#   MODEL=gpt-4o N_REPS=1 bash scripts/run_cp4_all.sh
set -uo pipefail

MODEL="${MODEL:-gpt-4o-mini}"
N_REPS="${N_REPS:-3}"

experiments=(
  exp_001_chunk256_ada
  exp_002_chunk512_ada
  exp_003_chunk128_topk10
  exp_004_chunk256_topk3
)

mkdir -p logs

pids=()

for exp in "${experiments[@]}"; do
  run_file="data/runs/$exp/run_results.json"
  if [[ ! -f "$run_file" ]]; then
    echo "SKIP $exp — $run_file not found (run CP3 first)"
    continue
  fi

  logfile="logs/${exp}_cp4.log"
  echo "==> Starting $exp  →  $logfile"
  python scripts/evaluate_run.py \
    --run "$run_file" \
    --model "$MODEL" \
    --n-reps "$N_REPS" \
    --force \
    >"$logfile" 2>&1 &
  pids+=($!)
done

if [[ ${#pids[@]} -eq 0 ]]; then
  echo "No experiments to evaluate. Run CP3 first."
  exit 1
fi

echo ""
echo "${#pids[@]} evaluations running in parallel. Waiting for completion..."
echo ""

failed=()
i=0
for exp in "${experiments[@]}"; do
  run_file="data/runs/$exp/run_results.json"
  [[ ! -f "$run_file" ]] && continue
  if wait "${pids[$i]}"; then
    echo "[OK]   $exp"
  else
    echo "[FAIL] $exp  →  logs/${exp}_cp4.log"
    failed+=("$exp")
  fi
  ((i++)) || true
done

echo ""
if [[ ${#failed[@]} -eq 0 ]]; then
  echo "All CP4 evaluations completed successfully. Results in data/evals/."
else
  echo "ERROR: ${#failed[@]} evaluation(s) failed: ${failed[*]}"
  exit 1
fi
