"""Unit tests for CP3 runner/retriever/answerer integration.

These tests run offline with a fake store and mocked OpenAI client.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_eval_lab.config.schema import ExperimentConfig
from rag_eval_lab.qa_generation.dataset import BenchmarkDataset, QAPair
from rag_eval_lab.rag.runner import RunResults, run_experiment
from rag_eval_lab.utils.io import sha256_of_file, write_json
from rag_eval_lab.utils.llm_client import LLMClient


class FakeStore:
    def __init__(self, matches: list[dict], namespace_count: int = 1) -> None:
        self._matches = matches
        self._namespace_counts = {"exp_runner": namespace_count}
        self.query_calls = 0

    def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[dict]:
        self.query_calls += 1
        matches = [m for m in self._matches if score_threshold is None or m["score"] >= score_threshold]
        return matches[:top_k]

    def namespace_vector_count(self, namespace: str) -> int:
        return self._namespace_counts.get(namespace, 0)

    def set_namespace_count(self, namespace: str, count: int) -> None:
        self._namespace_counts[namespace] = count


def _make_config(tmp_path: Path, benchmark_path: Path | None) -> ExperimentConfig:
    corpus = tmp_path / "corpus.pdf"
    corpus.write_bytes(b"%PDF-1.4 fake corpus")
    return ExperimentConfig(
        experiment_id="exp_runner",
        description="Runner test",
        corpus=corpus,
        benchmark=benchmark_path,
        chunking={"chunk_size": 256, "chunk_overlap": 32},
        retrieval={"top_k": 2, "score_threshold": 0.7},
        generation={
            "model": "gpt-5.4-mini",
            "temperature": 0.0,
            "max_tokens": 128,
            "system_prompt": "Answer only from context.",
        },
    )


def _make_benchmark(tmp_path: Path, corpus: Path) -> Path:
    benchmark = BenchmarkDataset(
        corpus_hash=sha256_of_file(corpus),
        created_at="2026-04-25T10:00:00+00:00",
        generator_model="gpt-5.4-mini",
        n_per_chunk=2,
        n_qa_pairs=2,
        qa_pairs=[
            QAPair(
                question="What is agentic AI?",
                expected_answer="A system with autonomous decision-making.",
                question_type="factual",
                source_chunk_id="c1",
            ),
            QAPair(
                question="Why does retrieval matter?",
                expected_answer="It improves answer grounding.",
                question_type="why",
                source_chunk_id="c2",
            ),
        ],
    )
    out_dir = tmp_path / "benchmark"
    out_dir.mkdir()
    return benchmark.save(out_dir)


def test_run_experiment_writes_valid_results(tmp_path: Path, monkeypatch, mock_openai_client) -> None:
    monkeypatch.chdir(tmp_path)

    config = _make_config(tmp_path, benchmark_path=None)
    benchmark_path = _make_benchmark(tmp_path, config.corpus)
    client = LLMClient(client=mock_openai_client)
    store = FakeStore(
        matches=[
            {
                "chunk_id": "doc_p1_0",
                "text": "Agentic AI can reason and act.",
                "source": "doc.pdf",
                "page": 1,
                "score": 0.91,
            },
            {
                "chunk_id": "doc_p2_1",
                "text": "Retrieval brings supporting evidence.",
                "source": "doc.pdf",
                "page": 2,
                "score": 0.88,
            },
        ],
        namespace_count=4,
    )

    mock_openai_client.set_embedding_response([[0.1, 0.2, 0.3]], total_tokens=6)
    mock_openai_client.set_chat_responses(
        [
            "Agentic AI is a system that can reason and act autonomously.",
            "Retrieval matters because it keeps answers grounded in evidence.",
        ]
    )

    output_path = run_experiment(
        config,
        benchmark_path=benchmark_path,
        llm_client=client,
        store=store,
        ingest_fn=lambda *args, **kwargs: pytest.fail("ingest should not be called"),
    )

    results = RunResults.load(output_path)
    assert results.experiment_id == "exp_runner"
    assert results.n_questions == 2
    assert results.benchmark_version == "v1_" + sha256_of_file(config.corpus)[:8] + "_2026-04-25"
    assert results.total_prompt_tokens == 20
    assert results.total_completion_tokens == 10
    assert results.total_embedding_tokens == 12
    assert len(results.results) == 2
    assert results.results[0].retrieved_context[0].chunk_id == "doc_p1_0"
    assert results.results[1].predicted_answer.startswith("Retrieval matters")
    assert store.query_calls == 2


def test_run_experiment_requires_force_when_output_exists(
    tmp_path: Path,
    monkeypatch,
    mock_openai_client,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = _make_config(tmp_path, benchmark_path=None)
    benchmark_path = _make_benchmark(tmp_path, config.corpus)
    output_path = tmp_path / "data" / "runs" / config.experiment_id / "run_results.json"
    output_path.parent.mkdir(parents=True)
    write_json(output_path, {"existing": True})

    client = LLMClient(client=mock_openai_client)
    store = FakeStore(matches=[], namespace_count=1)

    with pytest.raises(FileExistsError, match="run_results.json"):
        run_experiment(
            config,
            benchmark_path=benchmark_path,
            llm_client=client,
            store=store,
        )


def test_run_experiment_uses_benchmark_override_and_triggers_ingestion(
    tmp_path: Path,
    monkeypatch,
    mock_openai_client,
) -> None:
    monkeypatch.chdir(tmp_path)

    config = _make_config(tmp_path, benchmark_path=None)
    benchmark_path = _make_benchmark(tmp_path, config.corpus)
    client = LLMClient(client=mock_openai_client)
    store = FakeStore(
        matches=[
            {
                "chunk_id": "doc_p1_0",
                "text": "Context after ingestion.",
                "source": "doc.pdf",
                "page": 1,
                "score": 0.95,
            }
        ],
        namespace_count=0,
    )
    mock_openai_client.set_embedding_response([[0.1, 0.2, 0.3]], total_tokens=4)
    mock_openai_client.set_chat_responses(["Grounded answer one.", "Grounded answer two."])

    calls = {"ingest": 0}

    def _fake_ingest(cfg: ExperimentConfig, *, llm_client: LLMClient) -> None:
        calls["ingest"] += 1
        store.set_namespace_count(cfg.experiment_id, 10)

    output_path = run_experiment(
        config,
        benchmark_path=benchmark_path,
        llm_client=client,
        store=store,
        ingest_fn=_fake_ingest,
    )

    assert output_path.exists()
    assert calls["ingest"] == 1
