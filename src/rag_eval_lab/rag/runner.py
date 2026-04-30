from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from pydantic import BaseModel, Field
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from rag_eval_lab.config.schema import ExperimentConfig
from rag_eval_lab.ingestion.embedder import OpenAIEmbedder
from rag_eval_lab.ingestion.ingest import ingest
from rag_eval_lab.ingestion.pinecone_store import PineconeStore
from rag_eval_lab.qa_generation.dataset import BenchmarkDataset
from rag_eval_lab.rag.generator import Answerer
from rag_eval_lab.rag.retriever import Retriever
from rag_eval_lab.utils.io import sha256_of_file, write_json
from rag_eval_lab.utils.llm_client import LLMClient, UsageTracker
from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

_RUNS_DIR = Path("data/runs")


class RetrievedContextRecord(BaseModel):
    chunk_id: str
    text: str
    score: float
    source: str = ""
    page: int | None = None


class TokensUsed(BaseModel):
    prompt: int = Field(ge=0)
    completion: int = Field(ge=0)


class QuestionRunResult(BaseModel):
    qa_id: str
    question: str
    expected_answer: str
    retrieved_context: list[RetrievedContextRecord]
    predicted_answer: str
    tokens_used: TokensUsed
    latency_ms: int = Field(ge=0)
    finish_reason: str | None = None


class RunResults(BaseModel):
    experiment_id: str
    config_snapshot: dict[str, Any]
    benchmark_version: str
    benchmark_path: str
    started_at: str
    finished_at: str
    n_questions: int = Field(ge=0)
    total_cost_usd: float = Field(ge=0.0)
    total_prompt_tokens: int = Field(ge=0)
    total_completion_tokens: int = Field(ge=0)
    total_embedding_tokens: int = Field(ge=0)
    results: list[QuestionRunResult]

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        write_json(destination, self.model_dump(mode="json"))
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "RunResults":
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))


def run_experiment(
    config: ExperimentConfig,
    *,
    benchmark_path: str | Path | None = None,
    force: bool = False,
    max_questions: int | None = None,
    llm_client: LLMClient | None = None,
    store: PineconeStore | None = None,
    ingest_fn: Callable[..., Any] = ingest,
) -> Path:
    benchmark_file = _resolve_benchmark_path(config, benchmark_path)
    benchmark = BenchmarkDataset.load(benchmark_file)
    _validate_benchmark_corpus(config, benchmark)

    output_dir = _RUNS_DIR / config.experiment_id
    output_path = output_dir / "run_results.json"
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Run output already exists: {output_path}. Pass force=True (CLI: --force) to overwrite."
        )

    client = llm_client or LLMClient()
    embedder = OpenAIEmbedder(
        client,
        model=config.embedding.model,
        batch_size=config.embedding.batch_size,
    )
    runtime_store = store or PineconeStore(
        index_name=os.getenv("PINECONE_INDEX_NAME", "rag-eval-lab"),
        dim=embedder.dim,
    )

    _ensure_namespace(
        config,
        store=runtime_store,
        llm_client=client,
        ingest_fn=ingest_fn,
    )

    retriever = Retriever(
        runtime_store,
        embedder,
        namespace=config.experiment_id,
        top_k=config.retrieval.top_k,
        score_threshold=config.retrieval.score_threshold,
    )
    answerer = Answerer(
        client,
        model=config.generation.model,
        system_prompt=config.generation.system_prompt,
        temperature=config.generation.temperature,
        max_tokens=config.generation.max_tokens,
    )

    usage_before = _usage_snapshot(client.usage)
    started_at = _utc_now()
    results: list[QuestionRunResult] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        qa_pairs = benchmark.qa_pairs[:max_questions] if max_questions else benchmark.qa_pairs
        task = progress.add_task("Running RAG", total=len(qa_pairs))
        for qa_pair in qa_pairs:
            started = perf_counter()
            context = retriever.retrieve(qa_pair.question)
            completion = answerer.answer(qa_pair.question, context)
            latency_ms = int((perf_counter() - started) * 1000)

            if completion.finish_reason == "length":
                log.warning("Answer truncated for qa_id=%s", qa_pair.qa_id)

            results.append(
                QuestionRunResult(
                    qa_id=qa_pair.qa_id,
                    question=qa_pair.question,
                    expected_answer=qa_pair.expected_answer,
                    retrieved_context=[
                        RetrievedContextRecord.model_validate(chunk.__dict__)
                        for chunk in context
                    ],
                    predicted_answer=completion.text.strip(),
                    tokens_used=TokensUsed(
                        prompt=completion.prompt_tokens,
                        completion=completion.completion_tokens,
                    ),
                    latency_ms=latency_ms,
                    finish_reason=completion.finish_reason,
                )
            )
            progress.advance(task)

    usage_after = _usage_snapshot(client.usage)
    run_results = RunResults(
        experiment_id=config.experiment_id,
        config_snapshot=config.model_dump(mode="json", exclude_none=True),
        benchmark_version=_benchmark_version(benchmark),
        benchmark_path=str(benchmark_file),
        started_at=started_at,
        finished_at=_utc_now(),
        n_questions=len(results),
        total_cost_usd=max(0.0, usage_after["estimated_cost_usd"] - usage_before["estimated_cost_usd"]),
        total_prompt_tokens=max(0, usage_after["prompt_tokens"] - usage_before["prompt_tokens"]),
        total_completion_tokens=max(
            0,
            usage_after["completion_tokens"] - usage_before["completion_tokens"],
        ),
        total_embedding_tokens=max(
            0,
            usage_after["embedding_tokens"] - usage_before["embedding_tokens"],
        ),
        results=results,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_results.save(output_path)
    log.info(
        "Saved run results for %s (%d questions, ~$%.4f)",
        config.experiment_id,
        run_results.n_questions,
        run_results.total_cost_usd,
    )
    return output_path


def _resolve_benchmark_path(
    config: ExperimentConfig,
    benchmark_path: str | Path | None,
) -> Path:
    resolved = Path(benchmark_path) if benchmark_path is not None else config.benchmark
    if resolved is None:
        raise ValueError(
            "Benchmark path is required for CP3. Set `benchmark` in the YAML or pass --benchmark."
        )
    if not resolved.exists():
        raise FileNotFoundError(f"Benchmark file not found: {resolved}")
    return resolved


def _validate_benchmark_corpus(config: ExperimentConfig, benchmark: BenchmarkDataset) -> None:
    corpus_hash = sha256_of_file(config.corpus)
    if benchmark.corpus_hash != corpus_hash:
        raise ValueError(
            "Benchmark corpus hash does not match the configured corpus: "
            f"{benchmark.corpus_hash} != {corpus_hash}"
        )


def _ensure_namespace(
    config: ExperimentConfig,
    *,
    store: PineconeStore,
    llm_client: LLMClient,
    ingest_fn: Callable[..., Any],
) -> None:
    n_vectors = store.namespace_vector_count(config.experiment_id)
    if n_vectors > 0:
        log.info(
            "Namespace '%s' already populated with %d vectors; skipping ingestion.",
            config.experiment_id,
            n_vectors,
        )
        return

    log.info(
        "Namespace '%s' is empty; running ingestion before the experiment.",
        config.experiment_id,
    )
    ingest_fn(config, llm_client=llm_client)


def _benchmark_version(benchmark: BenchmarkDataset) -> str:
    return f"{benchmark.version}_{benchmark.corpus_hash[:8]}_{benchmark.created_at[:10]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _usage_snapshot(usage: UsageTracker) -> dict[str, float]:
    return {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "embedding_tokens": usage.embedding_tokens,
        "estimated_cost_usd": usage.estimated_cost_usd,
    }
