#!/usr/bin/env python3
"""CP4: LLM-as-a-Judge evaluation script.

Scores every predicted answer in a run_results.json on three metrics
(faithfulness, answer relevancy, context recall) using an LLM judge and
writes a metrics.json to data/evals/<experiment_id>/.

Usage:
    python scripts/evaluate_run.py \\
        --run data/runs/exp_001_chunk256_ada/run_results.json \\
        [--model gpt-4o-mini] [--n-reps 3] [--out data/evals/...] \\
        [--max-questions 50] [--force] [--log-level INFO]
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv

load_dotenv()

from rag_eval_lab.evaluation.judge import LLMJudge
from rag_eval_lab.evaluation.metrics import evaluate_run, save_metrics
from rag_eval_lab.rag.runner import RunResults
from rag_eval_lab.utils.llm_client import LLMClient
from rag_eval_lab.utils.logging import setup_logging, get_logger

app = typer.Typer(add_completion=False)
log = get_logger(__name__)


@app.command()
def main(
    run: Path = typer.Option(..., help="Path to run_results.json"),
    model: str = typer.Option("gpt-4o-mini", help="Judge model"),
    n_reps: int = typer.Option(3, min=1, max=5, help="Repetitions per metric (median is kept)"),
    out: Optional[Path] = typer.Option(None, help="Output path for metrics.json (default: data/evals/<exp_id>/metrics.json)"),
    max_questions: Optional[int] = typer.Option(None, help="Evaluate only first N questions (for cheap test runs)"),
    force: bool = typer.Option(False, help="Overwrite existing metrics.json"),
    log_level: str = typer.Option("INFO", help="Log level"),
) -> None:
    setup_logging(log_level)

    if not run.exists():
        typer.echo(f"ERROR: run file not found: {run}", err=True)
        raise typer.Exit(1)

    run_results = RunResults.load(run)
    log.info(
        "Loaded run: %s  (%d questions)",
        run_results.experiment_id,
        run_results.n_questions,
    )

    output_path = out or (Path("data/evals") / run_results.experiment_id / "metrics.json")
    if output_path.exists() and not force:
        typer.echo(
            f"ERROR: metrics file already exists: {output_path}. Use --force to overwrite.",
            err=True,
        )
        raise typer.Exit(1)

    client = LLMClient()
    judge = LLMJudge(client, model=model, n_reps=n_reps)

    report = evaluate_run(run_results, judge, run_path=run, max_questions=max_questions)
    path = save_metrics(report, output_path)

    agg = report["aggregated"]
    typer.echo("\n── Evaluation Report " + "─" * 27)
    typer.echo(f"  experiment_id   : {report['experiment_id']}")
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
