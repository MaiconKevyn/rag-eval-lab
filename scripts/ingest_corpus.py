#!/usr/bin/env python
"""CLI: ingest a corpus into Pinecone using a declarative YAML experiment config.

Usage:
    python scripts/ingest_corpus.py --config configs/exp_001_chunk256_ada.yaml
    python scripts/ingest_corpus.py --config configs/exp_001_chunk256_ada.yaml --rebuild
"""

from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv

load_dotenv()

from rag_eval_lab.config.loader import load_config
from rag_eval_lab.ingestion.ingest import ingest
from rag_eval_lab.utils.logging import get_logger, setup_logging

app = typer.Typer(add_completion=False)
log = get_logger(__name__)


@app.command()
def main(
    config: Path = typer.Option(..., "--config", "-c", help="Path to YAML experiment config."),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Delete existing namespace before ingesting."
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level."),
) -> None:
    setup_logging(log_level)

    log.info("Loading config: %s", config)
    exp_config = load_config(config)
    log.info(
        "Experiment: %s | chunk_size=%d | overlap=%d | model=%s",
        exp_config.experiment_id,
        exp_config.chunking.chunk_size,
        exp_config.chunking.chunk_overlap,
        exp_config.embedding.model,
    )

    report = ingest(exp_config, rebuild=rebuild)

    typer.echo("")
    typer.echo("── Ingestion Report ─────────────────────────────")
    typer.echo(f"  experiment_id : {report.experiment_id}")
    typer.echo(f"  namespace     : {report.namespace}")
    typer.echo(f"  pages read    : {report.n_pages}")
    typer.echo(f"  chunks        : {report.n_chunks}")
    typer.echo(f"  tokens (est.) : {report.total_tokens_estimated:,}")
    typer.echo(f"  cost (est.)   : ${report.estimated_cost_usd:.4f}")
    typer.echo(f"  corpus hash   : {report.corpus_hash[:16]}…")
    typer.echo(f"  from cache    : {report.from_cache}")
    if report.warnings:
        for w in report.warnings:
            typer.echo(f"  ⚠  {w}")
    typer.echo("─────────────────────────────────────────────────")
    typer.echo("Done.")


if __name__ == "__main__":
    app()
