#!/usr/bin/env python
"""CLI: run one declarative RAG experiment and persist run_results.json."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

from rag_eval_lab.config.loader import load_config
from rag_eval_lab.rag.runner import RunResults, run_experiment
from rag_eval_lab.utils.logging import get_logger, setup_logging

app = typer.Typer(add_completion=False)
log = get_logger(__name__)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML experiment config."),
    benchmark: Path | None = typer.Option(
        None,
        "--benchmark",
        "-b",
        help="Optional benchmark override. Required if the YAML does not define one.",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing run_results.json."),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
) -> None:
    setup_logging(log_level)

    log.info("Loading config: %s", config)
    exp_config = load_config(config)
    output_path = run_experiment(
        exp_config,
        benchmark_path=benchmark,
        force=force,
    )
    report = RunResults.load(output_path)

    typer.echo("")
    typer.echo("── RAG Run Report ───────────────────────────────")
    typer.echo(f"  experiment_id     : {report.experiment_id}")
    typer.echo(f"  benchmark_version : {report.benchmark_version}")
    typer.echo(f"  questions         : {report.n_questions}")
    typer.echo(f"  prompt tokens     : {report.total_prompt_tokens}")
    typer.echo(f"  completion tokens : {report.total_completion_tokens}")
    typer.echo(f"  embedding tokens  : {report.total_embedding_tokens}")
    typer.echo(f"  total cost (est.) : ${report.total_cost_usd:.4f}")
    typer.echo(f"  saved to          : {output_path}")
    typer.echo("─────────────────────────────────────────────────")
    typer.echo("Done.")


if __name__ == "__main__":
    app()
