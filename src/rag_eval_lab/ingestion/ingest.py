"""Orchestrates the full ingestion pipeline for one experiment config.

Flow:
  1. Read PDF(s) → pages
  2. Chunk (parametrised by YAML)
  3. Embed in batches (with parquet cache keyed by corpus hash + model + chunk params)
  4. Upsert into Pinecone namespace = config.experiment_id
  5. Return IngestionReport
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from pypdf import PdfReader
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from rag_eval_lab.config.schema import ExperimentConfig
from rag_eval_lab.ingestion.chunker import Chunk, Chunker
from rag_eval_lab.ingestion.embedder import OpenAIEmbedder
from rag_eval_lab.ingestion.pinecone_store import PineconeStore
from rag_eval_lab.utils.io import sha256_of_file
from rag_eval_lab.utils.llm_client import LLMClient, estimate_tokens
from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

_CACHE_DIR = Path("data/cache/embeddings")


@dataclass
class IngestionReport:
    experiment_id: str
    namespace: str
    n_pages: int
    n_chunks: int
    total_tokens_estimated: int
    estimated_cost_usd: float
    corpus_hash: str
    from_cache: bool
    warnings: list[str] = field(default_factory=list)


def _read_pdf_pages(path: Path) -> list[tuple[int, str]]:
    reader = PdfReader(path)
    pages: list[tuple[int, str]] = []
    skipped = 0
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
        else:
            skipped += 1
    if skipped:
        log.warning("Skipped %d empty/scanned pages in %s", skipped, path.name)
    return pages


def _cache_path(corpus_hash: str, model: str, chunk_size: int, overlap: int) -> Path:
    key = f"{corpus_hash[:16]}_{model.replace('/', '_')}_{chunk_size}_{overlap}"
    return _CACHE_DIR / f"{key}.parquet"


def _load_cache(cache_file: Path) -> list[list[float]] | None:
    if not cache_file.exists():
        return None
    df = pd.read_parquet(cache_file)
    log.info("Cache hit: loaded %d embeddings from %s", len(df), cache_file.name)
    return [np.array(v, dtype=np.float32).tolist() for v in df["embedding"]]


def _save_cache(cache_file: Path, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "chunk_id": [c.chunk_id for c in chunks],
            "embedding": [np.array(e, dtype=np.float32) for e in embeddings],
        }
    )
    df.to_parquet(cache_file, index=False)
    log.debug("Saved embedding cache → %s", cache_file.name)


def ingest(
    config: ExperimentConfig,
    *,
    rebuild: bool = False,
    llm_client: LLMClient | None = None,
) -> IngestionReport:
    corpus_path = config.corpus
    corpus_hash = sha256_of_file(corpus_path)
    namespace = config.experiment_id

    # — Pinecone setup —
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-eval-lab")
    embedder_model = config.embedding.model
    batch_size = config.embedding.batch_size

    client = llm_client or LLMClient()
    embedder = OpenAIEmbedder(client, model=embedder_model, batch_size=batch_size)
    store = PineconeStore(index_name=index_name, dim=embedder.dim)

    # — Namespace collision guard —
    existing = store.list_namespaces()
    if namespace in existing:
        if rebuild:
            log.info("--rebuild: deleting existing namespace '%s'", namespace)
            store.delete_namespace(namespace)
        else:
            raise RuntimeError(
                f"Namespace '{namespace}' already exists in index '{index_name}'. "
                "Pass rebuild=True (CLI: --rebuild) to overwrite."
            )

    # — Read PDF —
    log.info("Reading corpus: %s", corpus_path.name)
    pages = _read_pdf_pages(corpus_path)
    log.info("Loaded %d non-empty pages", len(pages))

    if len(pages) < 3:
        log.warning(
            "Only %d pages extracted — PDF may be scanned (no OCR). "
            "Consider adding pymupdf as fallback.",
            len(pages),
        )

    # — Chunk —
    chunker = Chunker(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        separators=config.chunking.separators,
    )
    chunks = chunker.split(pages, source=corpus_path.name)
    log.info("Created %d chunks (size=%d overlap=%d)", len(chunks), config.chunking.chunk_size, config.chunking.chunk_overlap)

    # — Estimate token cost before API call —
    texts = [c.text for c in chunks]
    total_tokens = sum(estimate_tokens(t, model=embedder_model) for t in texts)
    warnings: list[str] = []

    if config.chunking.chunk_size < 100:
        warnings.append(
            f"chunk_size={config.chunking.chunk_size} is very small. "
            "QA generation quality may degrade."
        )

    # — Embed (with cache) —
    cache_file = _cache_path(
        corpus_hash, embedder_model, config.chunking.chunk_size, config.chunking.chunk_overlap
    )
    cached = _load_cache(cache_file)
    from_cache = cached is not None

    if cached is not None:
        embeddings = cached
        cost = 0.0
        log.info("Using cached embeddings (%d vectors, $0.00)", len(embeddings))
    else:
        log.info(
            "Embedding %d chunks (~%d tokens) via %s…",
            len(chunks), total_tokens, embedder_model,
        )
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ) as progress:
            task = progress.add_task("Embedding", total=len(chunks))
            embeddings = []
            for i in range(0, len(chunks), batch_size):
                batch_texts = texts[i : i + batch_size]
                batch_vecs = client.embed(batch_texts, model=embedder_model)
                embeddings.extend(batch_vecs)
                progress.advance(task, len(batch_texts))

        cost = client.usage.estimated_cost_usd
        log.info(
            "Embedded %d chunks (~%d tokens, ~$%.4f)",
            len(chunks), total_tokens, cost,
        )
        _save_cache(cache_file, chunks, embeddings)

    # — Upsert to Pinecone —
    log.info("Upserting %d vectors to Pinecone namespace '%s'…", len(chunks), namespace)
    store.upsert(namespace=namespace, chunks=chunks, embeddings=embeddings)

    return IngestionReport(
        experiment_id=config.experiment_id,
        namespace=namespace,
        n_pages=len(pages),
        n_chunks=len(chunks),
        total_tokens_estimated=total_tokens,
        estimated_cost_usd=cost,
        corpus_hash=corpus_hash,
        from_cache=from_cache,
        warnings=warnings,
    )
