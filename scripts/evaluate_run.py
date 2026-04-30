#!/usr/bin/env python3
"""CP4: LLM-as-a-Judge evaluation script.

Scores every predicted answer in a run_results.json on three metrics
(faithfulness, answer relevancy, context recall) and writes metrics.json
to data/evals/<experiment_id>/.

Two modes:
  default  — streaming: one call at a time, uses RPD quota.
  --batch  — OpenAI Batch API: submits all prompts as a single job,
             uses separate batch limits (50% cheaper, no RPD impact).

Usage:
    # Streaming (small runs / smoke tests)
    python scripts/evaluate_run.py --run data/runs/exp_001.../run_results.json

    # Batch API (recommended for full 1054-question runs)
    python scripts/evaluate_run.py --run data/runs/exp_001.../run_results.json --batch
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()

from rag_eval_lab.evaluation.judge import LLMJudge
from rag_eval_lab.evaluation.metrics import evaluate_run, evaluate_run_batch, save_metrics
from rag_eval_lab.rag.runner import RunResults
from rag_eval_lab.utils.llm_client import LLMClient
from rag_eval_lab.utils.logging import get_logger, setup_logging

app = typer.Typer(add_completion=False)
log = get_logger(__name__)


@app.command()
def main(
    run: Path = typer.Option(..., help="Path to run_results.json"),
    model: str = typer.Option("gpt-4o-mini", help="Judge model"),
    n_reps: int = typer.Option(3, min=1, max=5, help="Repetitions per metric (median is kept)"),
    out: Optional[Path] = typer.Option(None, help="Output path for metrics.json"),
    max_questions: Optional[int] = typer.Option(None, help="Evaluate only first N questions"),
    batch: bool = typer.Option(False, help="Use OpenAI Batch API (recommended for full runs)"),
    poll_interval: int = typer.Option(30, help="Seconds between batch status polls (--batch only)"),
    force: bool = typer.Option(False, help="Overwrite existing metrics.json"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    setup_logging(log_level)

    if not run.exists():
        typer.echo(f"ERROR: run file not found: {run}", err=True)
        raise typer.Exit(1)

    run_results = RunResults.load(run)
    log.info("Loaded run: %s  (%d questions)", run_results.experiment_id, run_results.n_questions)

    output_path = out or (Path("data/evals") / run_results.experiment_id / "metrics.json")
    if output_path.exists() and not force:
        typer.echo(
            f"ERROR: metrics file already exists: {output_path}. Use --force to overwrite.",
            err=True,
        )
        raise typer.Exit(1)

    client = LLMClient()
    judge = LLMJudge(client, model=model, n_reps=n_reps)

    if batch:
        log.info("Mode: Batch API  (poll_interval=%ds)", poll_interval)
        report = evaluate_run_batch(
            run_results, judge,
            run_path=run,
            max_questions=max_questions,
            poll_interval=poll_interval,
        )
    else:
        log.info("Mode: streaming")
        report = evaluate_run(run_results, judge, run_path=run, max_questions=max_questions)

    path = save_metrics(report, output_path)

    agg = report["aggregated"]
    typer.echo("\n── Evaluation Report " + "─" * 27)
    typer.echo(f"  experiment_id   : {report['experiment_id']}")
    typer.echo(f"  mode            : {'batch (id=' + report.get('batch_id','') + ')' if batch else 'streaming'}")
    typer.echo(f"  judge model     : {report['judge_model']}  (n_reps={report['n_reps']})")
    typer.echo(f"  evaluated       : {report['n_evaluated']}  (skipped: {report['n_skipped']})")
    typer.echo(f"  faithfulness    : {agg['faithfulness']['mean']:.3f}  (median {agg['faithfulness']['median']:.1f})")
    typer.echo(f"  answer_relevancy: {agg['answer_relevancy']['mean']:.3f}  (median {agg['answer_relevancy']['median']:.1f})")
    typer.echo(f"  context_recall  : {agg['context_recall']['mean']:.3f}  (median {agg['context_recall']['median']:.1f})")
    typer.echo(f"  composite       : {agg['composite']['mean']:.3f}  (median {agg['composite']['median']:.1f})")
    typer.echo(f"  saved to        : {path}")
    typer.echo("─" * 49)
    typer.echo("Done.")


if __name__ == "__main__":
    app()
