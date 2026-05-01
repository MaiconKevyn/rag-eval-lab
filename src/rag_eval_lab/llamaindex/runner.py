"""LlamaIndex experiment runner — produces the same RunResults format as the vanilla pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from rag_eval_lab.config.schema import ExperimentConfig
from rag_eval_lab.llamaindex.indexer import build_llamaindex_index
from rag_eval_lab.llamaindex.retriever import LlamaIndexRetriever
from rag_eval_lab.ingestion.embedder import OpenAIEmbedder
from rag_eval_lab.qa_generation.dataset import BenchmarkDataset
from rag_eval_lab.rag.generator import Answerer
from rag_eval_lab.rag.runner import (
    QuestionRunResult,
    RetrievedContextRecord,
    RunResults,
    TokensUsed,
    _benchmark_version,
    _resolve_benchmark_path,
    _usage_snapshot,
    _validate_benchmark_corpus,
)
from rag_eval_lab.utils.llm_client import LLMClient
from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

_RUNS_DIR = Path("data/runs")


def run_llamaindex_experiment(
    config: ExperimentConfig,
    *,
    benchmark_path: str | Path | None = None,
    force: bool = False,
    max_questions: int | None = None,
    llm_client: LLMClient | None = None,
    rebuild: bool = False,
) -> Path:
    benchmark_file = _resolve_benchmark_path(config, benchmark_path)
    benchmark = BenchmarkDataset.load(benchmark_file)
    _validate_benchmark_corpus(config, benchmark)

    output_dir = _RUNS_DIR / config.experiment_id
    output_path = output_dir / "run_results.json"
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Run output already exists: {output_path}. Pass --force to overwrite."
        )

    client = llm_client or LLMClient()
    embedder = OpenAIEmbedder(
        client,
        model=config.embedding.model,
        batch_size=config.embedding.batch_size,
    )

    log.info("Building LlamaIndex index for '%s'...", config.experiment_id)
    resources = build_llamaindex_index(config, llm_client=client, rebuild=rebuild)

    retriever = LlamaIndexRetriever(
        resources=resources,
        embedder=embedder,
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
    started_at = datetime.now(timezone.utc).isoformat()
    results: list[QuestionRunResult] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
    ) as progress:
        qa_pairs = benchmark.qa_pairs[:max_questions] if max_questions else benchmark.qa_pairs
        task = progress.add_task("Running LlamaIndex RAG", total=len(qa_pairs))

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
        finished_at=datetime.now(timezone.utc).isoformat(),
        n_questions=len(results),
        total_cost_usd=max(0.0, usage_after["estimated_cost_usd"] - usage_before["estimated_cost_usd"]),
        total_prompt_tokens=max(0, usage_after["prompt_tokens"] - usage_before["prompt_tokens"]),
        total_completion_tokens=max(0, usage_after["completion_tokens"] - usage_before["completion_tokens"]),
        total_embedding_tokens=max(0, usage_after["embedding_tokens"] - usage_before["embedding_tokens"]),
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
