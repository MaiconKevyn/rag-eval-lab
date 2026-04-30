"""Aggregation of per-question judge scores into experiment-level statistics."""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_eval_lab.evaluation.judge import LLMJudge, QuestionScores
from rag_eval_lab.rag.runner import RunResults
from rag_eval_lab.utils.io import write_json
from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

_EVALS_DIR = Path("data/evals")


@dataclass
class MetricStats:
    mean: float
    std: float
    median: float
    min: float
    max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _stats(values: list[float]) -> MetricStats:
    if not values:
        return MetricStats(mean=0.0, std=0.0, median=0.0, min=0.0, max=0.0)
    return MetricStats(
        mean=round(statistics.mean(values), 4),
        std=round(statistics.stdev(values) if len(values) > 1 else 0.0, 4),
        median=round(statistics.median(values), 4),
        min=round(min(values), 4),
        max=round(max(values), 4),
    )


def aggregate(scores: list[QuestionScores]) -> dict[str, MetricStats]:
    return {
        "faithfulness": _stats([s.faithfulness for s in scores]),
        "answer_relevancy": _stats([s.answer_relevancy for s in scores]),
        "context_recall": _stats([s.context_recall for s in scores]),
        "composite": _stats([s.composite for s in scores]),
    }


def evaluate_run(
    run_results: RunResults,
    judge: LLMJudge,
    *,
    run_path: str | Path,
    max_questions: int | None = None,
) -> dict[str, Any]:
    """Score every question in run_results and return the evaluation report dict."""
    questions = run_results.results
    if max_questions is not None:
        questions = questions[:max_questions]

    from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

    scores: list[QuestionScores] = []
    n_skipped = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("Evaluating", total=len(questions))
        for q in questions:
            try:
                scores.append(judge.score(q))
            except Exception as exc:
                log.warning("Judge failed for qa_id=%s: %s", q.qa_id, exc)
                n_skipped += 1
            progress.advance(task)

    agg = aggregate(scores)
    report: dict[str, Any] = {
        "experiment_id": run_results.experiment_id,
        "run_path": str(run_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge.model,
        "n_reps": judge.n_reps,
        "n_evaluated": len(scores),
        "n_skipped": n_skipped,
        "aggregated": {k: v.to_dict() for k, v in agg.items()},
        "per_question": [s.to_dict() for s in scores],
    }

    cost = run_results.total_cost_usd  # for log context
    log.info(
        "Evaluation complete: %d scored, %d skipped | "
        "faithfulness=%.2f  relevancy=%.2f  recall=%.2f  composite=%.2f",
        len(scores),
        n_skipped,
        agg["faithfulness"].mean,
        agg["answer_relevancy"].mean,
        agg["context_recall"].mean,
        agg["composite"].mean,
    )
    return report


def evaluate_run_batch(
    run_results: RunResults,
    judge: LLMJudge,
    *,
    run_path: str | Path,
    max_questions: int | None = None,
    poll_interval: int = 30,
) -> dict[str, Any]:
    """Score questions via the OpenAI Batch API (separate limits, 50% cheaper).

    Submits all prompts as a single batch job, polls until complete, then
    parses results. Suitable for large evaluations that would hit the RPD limit.
    """
    questions = run_results.results
    if max_questions is not None:
        questions = questions[:max_questions]

    n_requests = len(questions) * 3 * judge.n_reps
    log.info(
        "Building %d batch requests (%d questions × 3 metrics × %d reps)…",
        n_requests, len(questions), judge.n_reps,
    )
    requests = judge.build_batch_requests(questions)

    batch_id = judge._client.submit_batch(
        requests,
        description=f"eval_{run_results.experiment_id}",
    )
    log.info(
        "Batch %s submitted (%d requests). Polling every %ds…",
        batch_id, len(requests), poll_interval,
    )

    batch = judge._client.wait_for_batch(batch_id, poll_interval=poll_interval)
    batch_output = judge._client.fetch_batch_results(batch)

    scores = judge.parse_batch_results(batch_output, questions)
    n_skipped = len(questions) - len(scores)

    agg = aggregate(scores)
    report: dict[str, Any] = {
        "experiment_id": run_results.experiment_id,
        "run_path": str(run_path),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": judge.model,
        "n_reps": judge.n_reps,
        "batch_id": batch_id,
        "n_evaluated": len(scores),
        "n_skipped": n_skipped,
        "aggregated": {k: v.to_dict() for k, v in agg.items()},
        "per_question": [s.to_dict() for s in scores],
    }

    log.info(
        "Batch evaluation complete: %d scored, %d skipped | "
        "faithfulness=%.2f  relevancy=%.2f  recall=%.2f  composite=%.2f",
        len(scores), n_skipped,
        agg["faithfulness"].mean, agg["answer_relevancy"].mean,
        agg["context_recall"].mean, agg["composite"].mean,
    )
    return report


def save_metrics(report: dict[str, Any], output_path: str | Path | None = None) -> Path:
    if output_path is None:
        exp_id = report["experiment_id"]
        output_path = _EVALS_DIR / exp_id / "metrics.json"
    path = Path(output_path)
    write_json(path, report)
    log.info("Metrics saved to %s", path)
    return path
