"""Log a completed RAG experiment run to MLflow.

One MLflow run per experiment_id. Parameters come from config_snapshot
(run_results.json), metrics from metrics.json. Artifacts: both JSON files
and the YAML config if it exists.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import mlflow

from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

_MLFLOW_EXPERIMENT = "rag-eval-lab"
_RUNS_DIR = Path("data/runs")
_EVALS_DIR = Path("data/evals")
_CONFIGS_DIR = Path("configs")


def _detect_framework(experiment_id: str) -> str:
    return "llamaindex" if experiment_id.endswith("_llamaindex") else "vanilla"


def _comparison_key(experiment_id: str) -> str:
    match = re.match(r"^(exp_\d+)", experiment_id)
    return match.group(1) if match else experiment_id.removesuffix("_llamaindex")


def log_experiment(experiment_id: str, tracking_uri: str = "sqlite:///mlflow.db") -> str:
    """Log one experiment to MLflow. Returns the MLflow run_id."""
    run_path = _RUNS_DIR / experiment_id / "run_results.json"
    metrics_path = _EVALS_DIR / experiment_id / "metrics.json"

    if not run_path.exists():
        raise FileNotFoundError(f"run_results.json not found: {run_path}")
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")

    run_data = json.loads(run_path.read_text())
    metrics_data = json.loads(metrics_path.read_text())

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    empty_context_rate = _compute_empty_context_rate(run_data)

    with mlflow.start_run(run_name=experiment_id) as run:
        _log_params(run_data)
        _log_metrics(metrics_data, run_data, empty_context_rate)
        _log_tags(run_data, metrics_data)
        _log_artifacts(experiment_id, run_path, metrics_path)

        run_id = run.info.run_id
        log.info("Logged %s → MLflow run_id=%s", experiment_id, run_id)
        return run_id


def log_all_experiments(tracking_uri: str = "sqlite:///mlflow.db") -> dict[str, str]:
    """Log all experiments that have both run_results.json and metrics.json."""
    results: dict[str, str] = {}
    for run_dir in sorted(_RUNS_DIR.iterdir()):
        exp_id = run_dir.name
        metrics_path = _EVALS_DIR / exp_id / "metrics.json"
        if not (run_dir / "run_results.json").exists():
            log.debug("Skipping %s — no run_results.json", exp_id)
            continue
        if not metrics_path.exists():
            log.debug("Skipping %s — no metrics.json", exp_id)
            continue
        run_id = log_experiment(exp_id, tracking_uri=tracking_uri)
        results[exp_id] = run_id
    return results


# ── helpers ──────────────────────────────────────────────────────────────────

def _log_params(run_data: dict[str, Any]) -> None:
    cfg = run_data.get("config_snapshot", {})
    chunking = cfg.get("chunking", {})
    embedding = cfg.get("embedding", {})
    retrieval = cfg.get("retrieval", {})
    generation = cfg.get("generation", {})

    mlflow.log_params({
        "framework": _detect_framework(run_data.get("experiment_id", "")),
        "chunk_size": chunking.get("chunk_size"),
        "chunk_overlap": chunking.get("chunk_overlap"),
        "embedding_model": embedding.get("model"),
        "top_k": retrieval.get("top_k"),
        "score_threshold": retrieval.get("score_threshold"),
        "generation_model": generation.get("model"),
        "temperature": generation.get("temperature"),
        "max_tokens": generation.get("max_tokens"),
        "n_questions": run_data.get("n_questions"),
        "description": cfg.get("description", ""),
    })


def _log_metrics(
    metrics_data: dict[str, Any],
    run_data: dict[str, Any],
    empty_context_rate: float,
) -> None:
    agg = metrics_data.get("aggregated", {})
    for metric_name, stats in agg.items():
        for stat_key in ("mean", "median", "std", "min", "max"):
            value = stats.get(stat_key)
            if value is not None:
                mlflow.log_metric(f"{metric_name}_{stat_key}", value)

    mlflow.log_metrics({
        "empty_context_rate": empty_context_rate,
        "total_cost_usd": run_data.get("total_cost_usd", 0.0),
        "total_prompt_tokens": run_data.get("total_prompt_tokens", 0),
        "total_completion_tokens": run_data.get("total_completion_tokens", 0),
        "n_evaluated": metrics_data.get("n_evaluated", 0),
        "n_skipped": metrics_data.get("n_skipped", 0),
    })


def _log_tags(run_data: dict[str, Any], metrics_data: dict[str, Any]) -> None:
    exp_id = run_data.get("experiment_id", "")
    mlflow.set_tags({
        "framework": _detect_framework(exp_id),
        "comparison_key": _comparison_key(exp_id),
        "benchmark_version": run_data.get("benchmark_version", ""),
        "judge_model": metrics_data.get("judge_model", ""),
        "n_reps": metrics_data.get("n_reps", ""),
        "started_at": run_data.get("started_at", ""),
        "finished_at": run_data.get("finished_at", ""),
    })


def _log_artifacts(
    experiment_id: str,
    run_path: Path,
    metrics_path: Path,
) -> None:
    mlflow.log_artifact(str(run_path), artifact_path="run")
    mlflow.log_artifact(str(metrics_path), artifact_path="eval")

    config_path = _CONFIGS_DIR / f"{experiment_id}.yaml"
    if config_path.exists():
        mlflow.log_artifact(str(config_path), artifact_path="config")


def _compute_empty_context_rate(run_data: dict[str, Any]) -> float:
    results = run_data.get("results", [])
    if not results:
        return 0.0
    empty = sum(1 for r in results if not r.get("retrieved_context"))
    return round(empty / len(results), 4)
