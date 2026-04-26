#!/usr/bin/env python
"""CLI: generate a QA benchmark dataset from a corpus PDF.

Usage:
    python scripts/generate_benchmark.py \\
        --corpus data/corpus/agentic_ai_landscape.pdf \\
        --n-per-chunk 3 \\
        --out data/benchmark/

Optional flags for cheaper test runs:
    --max-chunks 20          # use only first N chunks
    --chunk-size 256
    --chunk-overlap 32
    --model gpt-5.4-mini
    --similarity-threshold 0.92
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

from rag_eval_lab.ingestion.chunker import Chunker
from rag_eval_lab.ingestion.embedder import OpenAIEmbedder
from rag_eval_lab.qa_generation.dataset import BenchmarkDataset
from rag_eval_lab.qa_generation.generator import QAGenerator
from rag_eval_lab.qa_generation.validator import QAValidator
from rag_eval_lab.utils.io import sha256_of_file
from rag_eval_lab.utils.llm_client import LLMClient
from rag_eval_lab.utils.logging import get_logger, setup_logging

app = typer.Typer(add_completion=False)
log = get_logger(__name__)

_FACTUAL_DOMINANCE_THRESHOLD = 0.70


@app.command()
def main(
    corpus: Path = typer.Option(..., "--corpus", help="Path to PDF corpus file."),
    out: Path = typer.Option(Path("data/benchmark"), "--out", help="Output directory."),
    n_per_chunk: int = typer.Option(3, "--n-per-chunk", help="QA pairs to generate per chunk."),
    chunk_size: int = typer.Option(256, "--chunk-size"),
    chunk_overlap: int = typer.Option(32, "--chunk-overlap"),
    model: str = typer.Option("gpt-5.4-mini", "--model", help="LLM model for generation."),
    similarity_threshold: float = typer.Option(0.92, "--similarity-threshold"),
    max_chunks: int = typer.Option(0, "--max-chunks", help="Limit chunks (0 = all). For testing."),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    setup_logging(log_level)

    if not corpus.exists():
        typer.echo(f"Error: corpus file not found: {corpus}", err=True)
        raise typer.Exit(1)

    corpus_hash = sha256_of_file(corpus)
    log.info("Corpus: %s (hash=%s…)", corpus.name, corpus_hash[:16])

    # — Read + chunk PDF —
    reader = PdfReader(corpus)
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))

    chunker = Chunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.split(pages, source=corpus.name)
    log.info("Chunked into %d chunks (size=%d overlap=%d)", len(chunks), chunk_size, chunk_overlap)

    limit = max_chunks if max_chunks > 0 else len(chunks)
    if max_chunks > 0:
        log.info("Limiting to first %d chunks (--max-chunks)", limit)

    # — Generate QA pairs —
    client = LLMClient()
    generator = QAGenerator(client, model=model, n_per_chunk=n_per_chunk)
    raw_pairs = generator.generate_for_corpus(chunks, max_chunks=limit if max_chunks > 0 else None)

    if not raw_pairs:
        typer.echo("No QA pairs generated. Check LLM output or corpus quality.", err=True)
        raise typer.Exit(1)

    # — Validate: filter trivial + deduplicate —
    embedder = OpenAIEmbedder(client, model="text-embedding-3-small")
    validator = QAValidator(embedder, similarity_threshold=similarity_threshold)

    pairs = validator.filter_trivial(raw_pairs)
    pairs, dedup_report = validator.deduplicate(pairs)

    # — Quality checks —
    distribution = _type_distribution(pairs)
    _log_distribution(distribution, len(pairs))

    dominant_type, dominant_pct = _dominant(distribution, len(pairs))
    if dominant_pct > _FACTUAL_DOMINANCE_THRESHOLD:
        log.warning(
            "Question type '%s' dominates at %.0f%% (threshold %.0f%%). "
            "Consider re-generating with different chunks or higher n_per_chunk.",
            dominant_type,
            dominant_pct * 100,
            _FACTUAL_DOMINANCE_THRESHOLD * 100,
        )

    if len(pairs) < 50:
        log.warning(
            "Only %d QA pairs after dedup. Acceptance criterion is ≥50. "
            "Consider using more chunks or increasing n_per_chunk.",
            len(pairs),
        )

    # — Persist —
    dataset = BenchmarkDataset(
        corpus_hash=corpus_hash,
        created_at=datetime.now(timezone.utc).isoformat(),
        generator_model=model,
        n_per_chunk=n_per_chunk,
        n_qa_pairs=len(pairs),
        qa_pairs=pairs,
    )
    out.mkdir(parents=True, exist_ok=True)
    saved_path = dataset.save(out)

    typer.echo("")
    typer.echo("── Benchmark Report ─────────────────────────────")
    typer.echo(f"  corpus        : {corpus.name}")
    typer.echo(f"  chunks used   : {limit}")
    typer.echo(f"  raw pairs     : {len(raw_pairs)}")
    typer.echo(f"  after trivial : {len(pairs) + dedup_report.n_removed}")
    typer.echo(f"  after dedup   : {len(pairs)}")
    for qt, count in distribution.items():
        pct = count / len(pairs) * 100 if pairs else 0
        typer.echo(f"    {qt:<12}: {count:>4} ({pct:.0f}%)")
    typer.echo(f"  cost (est.)   : ${client.usage.estimated_cost_usd:.4f}")
    typer.echo(f"  saved to      : {saved_path}")
    typer.echo("─────────────────────────────────────────────────")
    typer.echo("Done.")


def _type_distribution(pairs: list) -> dict[str, int]:
    d: dict[str, int] = {"factual": 0, "comparative": 0, "why": 0, "how": 0}
    for p in pairs:
        d[p.question_type] = d.get(p.question_type, 0) + 1
    return d


def _dominant(distribution: dict[str, int], total: int) -> tuple[str, float]:
    if not total:
        return "factual", 0.0
    best = max(distribution, key=lambda k: distribution[k])
    return best, distribution[best] / total


def _log_distribution(distribution: dict[str, int], total: int) -> None:
    log.info("Question type distribution:")
    for qt, count in distribution.items():
        pct = count / total * 100 if total else 0
        log.info("  %-12s %3d  (%.0f%%)", qt, count, pct)


if __name__ == "__main__":
    app()
