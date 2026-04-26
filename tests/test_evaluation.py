"""Unit tests for CP4 — LLM-as-a-Judge.

All LLM calls are mocked; no network required.
"""

from __future__ import annotations

import json

import pytest

from rag_eval_lab.evaluation.judge import LLMJudge, QuestionScores
from rag_eval_lab.evaluation.metrics import aggregate
from rag_eval_lab.rag.runner import QuestionRunResult, RetrievedContextRecord, TokensUsed
from rag_eval_lab.utils.llm_client import LLMClient


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _result(
    *,
    question: str = "What is an AI agent?",
    expected: str = "A system that perceives and acts.",
    predicted: str = "An AI agent perceives its environment and takes actions.",
    context_texts: list[str] | None = None,
) -> QuestionRunResult:
    retrieved = [
        RetrievedContextRecord(chunk_id=f"c{i}", text=t, score=0.9, source="doc.pdf", page=1)
        for i, t in enumerate(context_texts or ["An AI agent perceives its environment and takes actions."])
    ]
    return QuestionRunResult(
        qa_id="test-001",
        question=question,
        expected_answer=expected,
        retrieved_context=retrieved,
        predicted_answer=predicted,
        tokens_used=TokensUsed(prompt=100, completion=20),
        latency_ms=500,
        finish_reason="stop",
    )


def _result_no_context(**kwargs) -> QuestionRunResult:
    r = _result(**kwargs)
    return r.model_copy(update={"retrieved_context": []})


def _judge_response(score: int, reasoning: str = "ok") -> str:
    return json.dumps({"score": score, "reasoning": reasoning})


# ── LLMJudge — single score parsing ─────────────────────────────────────────


class TestLLMJudgeScoring:
    def _make_judge(self, mock_openai_client, n_reps: int = 1) -> LLMJudge:
        client = LLMClient(client=mock_openai_client)
        return LLMJudge(client, model="gpt-4o-mini", n_reps=n_reps)

    def test_score_parses_valid_json(self, mock_openai_client) -> None:
        mock_openai_client.set_chat_responses([
            _judge_response(5, "fully grounded"),   # faithfulness
            _judge_response(4, "mostly relevant"),  # answer_relevancy
            _judge_response(3, "partial recall"),   # context_recall
        ])
        judge = self._make_judge(mock_openai_client, n_reps=1)
        scores = judge.score(_result())

        assert scores.faithfulness == 5.0
        assert scores.answer_relevancy == 4.0
        assert scores.context_recall == 3.0
        assert scores.faithfulness_reasoning == "fully grounded"

    def test_composite_is_mean_of_three(self, mock_openai_client) -> None:
        mock_openai_client.set_chat_responses([
            _judge_response(4),
            _judge_response(2),
            _judge_response(3),
        ])
        judge = self._make_judge(mock_openai_client, n_reps=1)
        scores = judge.score(_result())
        assert abs(scores.composite - (4 + 2 + 3) / 3) < 1e-9

    def test_score_clamped_to_1_5(self, mock_openai_client) -> None:
        mock_openai_client.set_chat_responses([
            json.dumps({"score": 99, "reasoning": ""}),  # faithfulness — above max
            _judge_response(5),
            _judge_response(5),
        ])
        judge = self._make_judge(mock_openai_client, n_reps=1)
        scores = judge.score(_result())
        assert scores.faithfulness == 5.0

    def test_malformed_json_falls_back_to_3(self, mock_openai_client) -> None:
        mock_openai_client.set_chat_responses([
            "not json {{{",  # faithfulness
            _judge_response(4),
            _judge_response(5),
        ])
        judge = self._make_judge(mock_openai_client, n_reps=1)
        scores = judge.score(_result())
        assert scores.faithfulness == 3.0  # fallback default

    def test_n_reps_3_returns_median(self, mock_openai_client) -> None:
        # faithfulness called 3 times: scores [3, 5, 4] → median 4
        # answer_relevancy: [5, 5, 5] → 5
        # context_recall: [1, 2, 3] → 2
        mock_openai_client.set_chat_responses([
            _judge_response(3),  # faithfulness rep 1
            _judge_response(5),  # faithfulness rep 2
            _judge_response(4),  # faithfulness rep 3
            _judge_response(5),  # answer_relevancy rep 1
            _judge_response(5),  # answer_relevancy rep 2
            _judge_response(5),  # answer_relevancy rep 3
            _judge_response(1),  # context_recall rep 1
            _judge_response(2),  # context_recall rep 2
            _judge_response(3),  # context_recall rep 3
        ])
        judge = self._make_judge(mock_openai_client, n_reps=3)
        scores = judge.score(_result())

        assert scores.faithfulness == 4.0   # median(3, 5, 4)
        assert scores.answer_relevancy == 5.0
        assert scores.context_recall == 2.0  # median(1, 2, 3)

    def test_no_context_format(self, mock_openai_client) -> None:
        mock_openai_client.set_chat_responses([
            _judge_response(5),
            _judge_response(3),
            _judge_response(1),
        ])
        judge = self._make_judge(mock_openai_client, n_reps=1)
        # Should not raise even with empty retrieved_context
        scores = judge.score(_result_no_context())
        assert isinstance(scores, QuestionScores)

    def test_context_truncated_when_too_long(self, mock_openai_client) -> None:
        long_text = "x" * 10_000
        mock_openai_client.set_chat_responses([
            _judge_response(4),
            _judge_response(4),
            _judge_response(4),
        ])
        judge = self._make_judge(mock_openai_client, n_reps=1)
        # Should not raise; context is truncated internally
        judge.score(_result(context_texts=[long_text]))
        # Verify the prompt sent was within reasonable length
        call_args = mock_openai_client.chat.completions.create.call_args_list[0]
        _, kwargs = call_args
        prompt_content = kwargs["messages"][0]["content"]
        assert len(prompt_content) < 20_000  # well under any model limit


# ── QuestionScores — composite ────────────────────────────────────────────────


class TestQuestionScoresComposite:
    def test_composite_arithmetic(self) -> None:
        s = QuestionScores(qa_id="x", faithfulness=3.0, answer_relevancy=5.0, context_recall=4.0)
        assert abs(s.composite - 4.0) < 1e-9

    def test_to_dict_contains_all_keys(self) -> None:
        s = QuestionScores(qa_id="x", faithfulness=3.0, answer_relevancy=5.0, context_recall=4.0)
        d = s.to_dict()
        assert {"qa_id", "faithfulness", "answer_relevancy", "context_recall", "composite"} <= d.keys()


# ── aggregate ─────────────────────────────────────────────────────────────────


class TestAggregate:
    def _scores(self, fs: list[float], rs: list[float], cs: list[float]) -> list[QuestionScores]:
        return [
            QuestionScores(qa_id=f"q{i}", faithfulness=f, answer_relevancy=r, context_recall=c)
            for i, (f, r, c) in enumerate(zip(fs, rs, cs))
        ]

    def test_mean_is_correct(self) -> None:
        scores = self._scores([2.0, 4.0], [4.0, 4.0], [3.0, 5.0])
        agg = aggregate(scores)
        assert agg["faithfulness"].mean == 3.0
        assert agg["context_recall"].mean == 4.0

    def test_composite_stat(self) -> None:
        scores = self._scores([4.0], [4.0], [4.0])
        agg = aggregate(scores)
        assert agg["composite"].mean == 4.0

    def test_empty_scores_returns_zeros(self) -> None:
        agg = aggregate([])
        for key in ("faithfulness", "answer_relevancy", "context_recall", "composite"):
            assert agg[key].mean == 0.0

    def test_single_score_std_is_zero(self) -> None:
        scores = self._scores([3.0], [3.0], [3.0])
        agg = aggregate(scores)
        assert agg["faithfulness"].std == 0.0

    def test_min_max_correct(self) -> None:
        scores = self._scores([1.0, 5.0], [3.0, 3.0], [2.0, 4.0])
        agg = aggregate(scores)
        assert agg["faithfulness"].min == 1.0
        assert agg["faithfulness"].max == 5.0

    def test_returns_all_four_keys(self) -> None:
        agg = aggregate(self._scores([3.0], [3.0], [3.0]))
        assert set(agg.keys()) == {"faithfulness", "answer_relevancy", "context_recall", "composite"}
