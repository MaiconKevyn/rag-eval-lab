#!/usr/bin/env python
"""CP5 — Log all completed RAG experiments to MLflow."""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

from rag_eval_lab.tracking.mlflow_logger import log_all_experiments, log_experiment
from rag_eval_lab.utils.logging import get_logger, setup_logging

app = typer.Typer(add_completion=False)
log = get_logger(__name__)


@app.command()
def main(
    experiment_id: str | None = typer.Option(
        None, "--experiment", "-e",
        help="Log a single experiment. Omit to log all.",
    ),
    tracking_uri: str = typer.Option(
        "sqlite:///mlflow.db", "--tracking-uri",
        help="MLflow tracking URI (local path or remote server).",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    setup_logging(log_level)

    if experiment_id:
        run_id = log_experiment(experiment_id, tracking_uri=tracking_uri)
        typer.echo(f"Logged {experiment_id} → run_id={run_id}")
    else:
        results = log_all_experiments(tracking_uri=tracking_uri)
        typer.echo("")
        typer.echo("── MLflow Logging Report ────────────────────────")
        for exp_id, run_id in results.items():
            typer.echo(f"  {exp_id:<35} run_id={run_id}")
        typer.echo(f"  Total: {len(results)} experiment(s) logged")
        typer.echo("─────────────────────────────────────────────────")
        typer.echo(f"  Run: mlflow ui --backend-store-uri {tracking_uri}")
        typer.echo("─────────────────────────────────────────────────")
        typer.echo("Done.")


if __name__ == "__main__":
    app()
