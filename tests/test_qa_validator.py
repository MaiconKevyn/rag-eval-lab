"""Unit tests for QAGenerator (parsing) and QAValidator (dedup + trivial filter).

All tests run offline — LLM and embedder are mocked.
"""

from __future__ import annotations

import json
from typing import Protocol

import numpy as np
import pytest

from rag_eval_lab.qa_generation.dataset import QAPair
from rag_eval_lab.qa_generation.generator import QAGenerator, _normalise_type
from rag_eval_lab.qa_generation.validator import QAValidator


# ── Helpers ─────────────────────────────────────────────────────────────────

def _qa(question: str, qt: str = "factual", chunk_id: str = "c1") -> QAPair:
    return QAPair(question=question, expected_answer="ans", question_type=qt, source_chunk_id=chunk_id)


class _FakeEmbedder:
    """Returns fixed vectors per question for deterministic similarity tests."""

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors
        self._dim = len(next(iter(vectors.values()))) if vectors else 3

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return "fake"

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vectors.get(t, [0.0] * self._dim) for t in texts]


def _unit(v: list[float]) -> list[float]:
    arr = np.array(v, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


# ── QAGenerator — parsing ────────────────────────────────────────────────────

class TestQAGeneratorParsing:
    def _make_generator(self, mock_openai_client):
        from rag_eval_lab.utils.llm_client import LLMClient
        return QAGenerator(LLMClient(client=mock_openai_client), model="gpt-4o-mini", n_per_chunk=2)

    def test_valid_json_returns_pairs(self, mock_openai_client) -> None:
        payload = json.dumps({
            "qa_pairs": [
                {"question": "Q1?", "expected_answer": "A1", "question_type": "factual", "source_chunk_id": "c1"},
                {"question": "Q2?", "expected_answer": "A2", "question_type": "why", "source_chunk_id": "c1"},
            ]
        })
        mock_openai_client.set_chat_response(payload)
        gen = self._make_generator(mock_openai_client)

        from rag_eval_lab.ingestion.chunker import Chunk
        chunk = Chunk(chunk_id="c1", text="A" * 200, source="doc.pdf", page=1, chunk_index=0)
        pairs = gen.generate_for_chunk(chunk)

        assert len(pairs) == 2
        assert pairs[0].question == "Q1?"
        assert pairs[1].question_type == "why"

    def test_malformed_json_returns_empty_not_raises(self, mock_openai_client) -> None:
        mock_openai_client.set_chat_response("not json at all {{{")
        gen = self._make_generator(mock_openai_client)

        from rag_eval_lab.ingestion.chunker import Chunk
        chunk = Chunk(chunk_id="c1", text="A" * 200, source="doc.pdf", page=1, chunk_index=0)
        pairs = gen.generate_for_chunk(chunk)

        assert pairs == []  # must not raise

    def test_short_chunk_is_skipped(self, mock_openai_client) -> None:
        gen = self._make_generator(mock_openai_client)

        from rag_eval_lab.ingestion.chunker import Chunk
        chunk = Chunk(chunk_id="c1", text="short", source="doc.pdf", page=1, chunk_index=0)
        pairs = gen.generate_for_chunk(chunk)

        assert pairs == []
        mock_openai_client.chat.completions.create.assert_not_called()

    def test_invalid_qa_pair_field_skipped(self, mock_openai_client) -> None:
        payload = json.dumps({
            "qa_pairs": [
                {"question": "Q?", "expected_answer": "A", "question_type": "invalid_type", "source_chunk_id": "c1"},
                {"question": "Good Q?", "expected_answer": "A", "question_type": "factual", "source_chunk_id": "c1"},
            ]
        })
        mock_openai_client.set_chat_response(payload)
        gen = self._make_generator(mock_openai_client)

        from rag_eval_lab.ingestion.chunker import Chunk
        chunk = Chunk(chunk_id="c1", text="A" * 200, source="doc.pdf", page=1, chunk_index=0)
        pairs = gen.generate_for_chunk(chunk)
        # "invalid_type" normalises to "factual", so both should be kept
        assert len(pairs) == 2


# ── QAValidator — deduplication ──────────────────────────────────────────────

class TestQAValidatorDedup:
    def test_identical_questions_deduped(self) -> None:
        q = "What is the purpose of AI agents?"
        vec = _unit([1.0, 0.0, 0.0])
        embedder = _FakeEmbedder({q: vec})

        validator = QAValidator(embedder, similarity_threshold=0.92)
        pairs = [_qa(q), _qa(q)]
        kept, report = validator.deduplicate(pairs)

        assert len(kept) == 1
        assert report.n_removed == 1

    def test_different_questions_both_kept(self) -> None:
        q1, q2 = "What is AI?", "How does retrieval work?"
        embedder = _FakeEmbedder({
            q1: _unit([1.0, 0.0, 0.0]),
            q2: _unit([0.0, 1.0, 0.0]),
        })

        validator = QAValidator(embedder, similarity_threshold=0.92)
        pairs = [_qa(q1), _qa(q2)]
        kept, report = validator.deduplicate(pairs)

        assert len(kept) == 2
        assert report.n_removed == 0

    def test_near_duplicate_above_threshold_removed(self) -> None:
        q1 = "What is the role of memory in agents?"
        q2 = "What role does memory play in AI agents?"
        # Both nearly identical vectors → cosine similarity ~ 0.99
        v = _unit([1.0, 0.01, 0.0])
        embedder = _FakeEmbedder({q1: v, q2: v})

        validator = QAValidator(embedder, similarity_threshold=0.92)
        pairs = [_qa(q1), _qa(q2)]
        kept, report = validator.deduplicate(pairs)

        assert len(kept) == 1
        assert report.n_removed == 1

    def test_single_pair_returns_unchanged(self) -> None:
        q = "Only question?"
        embedder = _FakeEmbedder({q: _unit([1.0, 0.0])})
        validator = QAValidator(embedder)
        pairs = [_qa(q)]
        kept, report = validator.deduplicate(pairs)

        assert kept == pairs
        assert report.n_removed == 0

    def test_empty_list_returns_unchanged(self) -> None:
        embedder = _FakeEmbedder({})
        validator = QAValidator(embedder)
        kept, report = validator.deduplicate([])
        assert kept == []
        assert report.n_removed == 0


# ── QAValidator — trivial filter ─────────────────────────────────────────────

class TestQAValidatorTrivialFilter:
    def _validator(self) -> QAValidator:
        return QAValidator(_FakeEmbedder({"x": [1.0]}))

    def test_trivial_questions_removed(self) -> None:
        pairs = [
            _qa("O que diz o documento?"),
            _qa("Do que se trata o trecho?"),
            _qa("What is the definition of an agent?"),
        ]
        kept = self._validator().filter_trivial(pairs)
        assert len(kept) == 1
        assert kept[0].question == "What is the definition of an agent?"

    def test_non_trivial_questions_kept(self) -> None:
        pairs = [_qa("What are the three memory types?"), _qa("Why does planning matter?")]
        kept = self._validator().filter_trivial(pairs)
        assert len(kept) == 2


# ── _normalise_type ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("factual", "factual"),
    ("FACTUAL", "factual"),
    ("why", "why"),
    ("por que", "why"),
    ("porquê", "why"),
    ("how", "how"),
    ("como", "how"),
    ("comparative", "comparative"),
    ("comparativo", "comparative"),
    ("unknown_type", "factual"),  # fallback
])
def test_normalise_type(raw: str, expected: str) -> None:
    assert _normalise_type(raw) == expected
