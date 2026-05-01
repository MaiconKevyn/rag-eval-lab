"""Offline tests for the CP7 LlamaIndex path.

These tests avoid importing the real LlamaIndex package by monkeypatching the
minimal interfaces the project depends on.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_eval_lab.config.schema import ExperimentConfig
from rag_eval_lab.qa_generation.dataset import BenchmarkDataset, QAPair
from rag_eval_lab.utils.io import sha256_of_file
from rag_eval_lab.utils.llm_client import LLMClient


def _make_config(tmp_path: Path) -> ExperimentConfig:
    corpus = tmp_path / "corpus.pdf"
    corpus.write_bytes(b"%PDF-1.4 fake corpus")
    return ExperimentConfig(
        experiment_id="exp_llamaindex",
        description="LlamaIndex test",
        corpus=corpus,
        benchmark=None,
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


class _FakeNode:
    def __init__(self, node_id: str, text: str, metadata: dict) -> None:
        self.node_id = node_id
        self.id_ = node_id
        self._text = text
        self.metadata = metadata

    def get_content(self) -> str:
        return self._text


class _FakeVectorStore:
    def __init__(self, nodes: list[_FakeNode], similarities: list[float | None]) -> None:
        self._nodes = nodes
        self._similarities = similarities
        self.queries = []

    def query(self, query_obj):
        self.queries.append(query_obj)
        return SimpleNamespace(nodes=self._nodes, similarities=self._similarities)


def test_llamaindex_retriever_returns_chunks_and_filters_scores(monkeypatch) -> None:
    from rag_eval_lab.llamaindex.indexer import LlamaIndexResources
    from rag_eval_lab.llamaindex.retriever import LlamaIndexRetriever

    class _FakeVectorStoreQuery:
        def __init__(self, query_embedding, similarity_top_k) -> None:
            self.query_embedding = query_embedding
            self.similarity_top_k = similarity_top_k

    monkeypatch.setitem(__import__("sys").modules, "llama_index.core.vector_stores.types", SimpleNamespace(VectorStoreQuery=_FakeVectorStoreQuery))

    nodes = [
        _FakeNode("c1", "Chunk one", {"source": "doc.pdf", "page": 1}),
        _FakeNode("c2", "Chunk two", {"source": "doc.pdf", "page": 2}),
    ]
    vector_store = _FakeVectorStore(nodes, [0.91, 0.42])
    resources = LlamaIndexResources(index=object(), vector_store=vector_store)
    embedder = SimpleNamespace(embed=lambda texts: [[0.1, 0.2, 0.3]])

    retriever = LlamaIndexRetriever(resources=resources, embedder=embedder, top_k=2, score_threshold=0.7)
    chunks = retriever.retrieve("What is agentic AI?")

    assert len(chunks) == 1
    assert chunks[0].chunk_id == "c1"
    assert chunks[0].text == "Chunk one"


def test_llamaindex_retriever_keeps_nodes_when_similarity_is_missing(monkeypatch) -> None:
    from rag_eval_lab.llamaindex.indexer import LlamaIndexResources
    from rag_eval_lab.llamaindex.retriever import LlamaIndexRetriever

    class _FakeVectorStoreQuery:
        def __init__(self, query_embedding, similarity_top_k) -> None:
            self.query_embedding = query_embedding
            self.similarity_top_k = similarity_top_k

    monkeypatch.setitem(__import__("sys").modules, "llama_index.core.vector_stores.types", SimpleNamespace(VectorStoreQuery=_FakeVectorStoreQuery))

    nodes = [_FakeNode("c1", "Recovered text", {"source": "doc.pdf", "page": 1, "text": "Recovered text"})]
    vector_store = _FakeVectorStore(nodes, [None])
    resources = LlamaIndexResources(index=object(), vector_store=vector_store)
    embedder = SimpleNamespace(embed=lambda texts: [[0.1, 0.2, 0.3]])

    retriever = LlamaIndexRetriever(resources=resources, embedder=embedder, top_k=1, score_threshold=0.7)
    chunks = retriever.retrieve("Question")

    assert len(chunks) == 1
    assert chunks[0].text == "Recovered text"


def test_run_llamaindex_experiment_writes_valid_results(tmp_path: Path, monkeypatch, mock_openai_client) -> None:
    from rag_eval_lab.llamaindex.indexer import LlamaIndexResources
    from rag_eval_lab.llamaindex.runner import RunResults, run_llamaindex_experiment

    monkeypatch.chdir(tmp_path)
    config = _make_config(tmp_path)
    benchmark_path = _make_benchmark(tmp_path, config.corpus)

    fake_nodes = [
        _FakeNode("doc_p1_0", "Agentic AI can reason and act.", {"source": "doc.pdf", "page": 1}),
        _FakeNode("doc_p2_1", "Retrieval brings supporting evidence.", {"source": "doc.pdf", "page": 2}),
    ]
    resources = LlamaIndexResources(
        index=object(),
        vector_store=_FakeVectorStore(fake_nodes, [0.91, 0.88]),
    )

    monkeypatch.setattr(
        "rag_eval_lab.llamaindex.runner.build_llamaindex_index",
        lambda *args, **kwargs: resources,
    )

    class _FakeVectorStoreQuery:
        def __init__(self, query_embedding, similarity_top_k) -> None:
            self.query_embedding = query_embedding
            self.similarity_top_k = similarity_top_k

    monkeypatch.setitem(__import__("sys").modules, "llama_index.core.vector_stores.types", SimpleNamespace(VectorStoreQuery=_FakeVectorStoreQuery))

    client = LLMClient(client=mock_openai_client)
    mock_openai_client.set_embedding_response([[0.1, 0.2, 0.3]], total_tokens=6)
    mock_openai_client.set_chat_responses(
        [
            "Agentic AI is a system that can reason and act autonomously.",
            "Retrieval matters because it keeps answers grounded in evidence.",
        ]
    )

    output_path = run_llamaindex_experiment(
        config,
        benchmark_path=benchmark_path,
        llm_client=client,
    )

    results = RunResults.load(output_path)
    assert results.experiment_id == "exp_llamaindex"
    assert results.n_questions == 2
    assert results.total_embedding_tokens == 12
    assert len(results.results[0].retrieved_context) == 2
    assert results.results[0].retrieved_context[0].chunk_id == "doc_p1_0"


def test_llamaindex_configs_match_vanilla_thresholds() -> None:
    from rag_eval_lab.config.loader import load_config

    pairs = [
        ("configs/exp_001_chunk256_ada.yaml", "configs/exp_001_llamaindex.yaml"),
        ("configs/exp_002_chunk512_ada.yaml", "configs/exp_002_llamaindex.yaml"),
        ("configs/exp_003_chunk128_topk10.yaml", "configs/exp_003_llamaindex.yaml"),
        ("configs/exp_004_chunk256_topk3.yaml", "configs/exp_004_llamaindex.yaml"),
    ]

    for vanilla_path, llamaindex_path in pairs:
        vanilla = load_config(vanilla_path)
        llamaindex = load_config(llamaindex_path)
        assert vanilla.chunking.chunk_size == llamaindex.chunking.chunk_size
        assert vanilla.chunking.chunk_overlap == llamaindex.chunking.chunk_overlap
        assert vanilla.retrieval.top_k == llamaindex.retrieval.top_k
        assert vanilla.retrieval.score_threshold == llamaindex.retrieval.score_threshold
        assert vanilla.embedding.model == llamaindex.embedding.model
