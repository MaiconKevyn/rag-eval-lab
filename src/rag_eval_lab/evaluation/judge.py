"""LLM-as-a-Judge for three RAG quality metrics.

Each metric is scored 1-5 on n_reps independent calls; the median is kept to
smooth out LLM variance. Three metrics:

  faithfulness     — predicted answer is grounded in the retrieved context
  answer_relevancy — predicted answer addresses the question asked
  context_recall   — retrieved context contains the expected answer's information
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import median
from typing import Any

# Short codes used in batch custom_ids to stay under the 64-char limit.
# Format: "{qa_id}:{metric_code}:{rep}"  e.g. "abc-123:f:0"
_METRIC_CODES: dict[str, str] = {
    "f": "faithfulness",
    "r": "answer_relevancy",
    "c": "context_recall",
}

from rag_eval_lab.rag.runner import QuestionRunResult, RetrievedContextRecord
from rag_eval_lab.utils.llm_client import LLMClient
from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

_MAX_CONTEXT_CHARS = 3_000

_FAITHFULNESS_PROMPT = """\
You are an impartial judge evaluating a RAG system.

Score the FAITHFULNESS of the predicted answer relative to the retrieved context.
Faithfulness measures whether the predicted answer is grounded in the context and
does not hallucinate facts.

Question: {question}

Retrieved context:
{context}

Predicted answer: {predicted_answer}

Rubric:
1 – Contradicts context or invents facts not in it.
2 – Significant unsupported claims.
3 – Mostly supported, minor unsupported elements.
4 – Well-supported, only trivial gaps.
5 – Fully and precisely grounded in context.

Special cases:
• No context + answer says "not enough information" → score 5.
• No context + answer fabricates facts → score 1.

Reply with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>"}}\
"""

_ANSWER_RELEVANCY_PROMPT = """\
You are an impartial judge evaluating a RAG system.

Score the ANSWER RELEVANCY of the predicted answer to the question.
Relevancy measures whether the answer actually addresses what was asked.

Question: {question}

Predicted answer: {predicted_answer}

Rubric:
1 – Completely irrelevant or refuses without reason.
2 – Tangentially related, misses the core question.
3 – Partially addresses the question.
4 – Mostly addresses the question, minor omissions.
5 – Directly and completely addresses the question.

Reply with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>"}}\
"""

_CONTEXT_RECALL_PROMPT = """\
You are an impartial judge evaluating a RAG system.

Score the CONTEXT RECALL for this pair.
Context recall measures whether the retrieved context contains the information
needed to produce the expected (ground-truth) answer.

Question: {question}

Expected answer (ground truth): {expected_answer}

Retrieved context:
{context}

Rubric:
1 – Context has none of the needed information.
2 – Context has minimal relevant information.
3 – Context partially supports the expected answer.
4 – Context largely supports, minor gaps.
5 – Context fully contains the needed information.

If there is no retrieved context, score 1.

Reply with JSON only: {{"score": <1-5>, "reasoning": "<one sentence>"}}\
"""


@dataclass
class QuestionScores:
    qa_id: str
    faithfulness: float
    answer_relevancy: float
    context_recall: float
    faithfulness_reasoning: str = ""
    answer_relevancy_reasoning: str = ""
    context_recall_reasoning: str = ""

    @property
    def composite(self) -> float:
        return (self.faithfulness + self.answer_relevancy + self.context_recall) / 3

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_recall": self.context_recall,
            "composite": round(self.composite, 4),
            "faithfulness_reasoning": self.faithfulness_reasoning,
            "answer_relevancy_reasoning": self.answer_relevancy_reasoning,
            "context_recall_reasoning": self.context_recall_reasoning,
        }


class LLMJudge:
    """Scores RAG results on faithfulness, answer relevancy, and context recall."""

    def __init__(
        self,
        llm_client: LLMClient,
        model: str = "gpt-4o-mini",
        n_reps: int = 3,
        max_context_chars: int = _MAX_CONTEXT_CHARS,
    ) -> None:
        self._client = llm_client
        self.model = model
        self.n_reps = n_reps
        self.max_context_chars = max_context_chars

    def _format_context(self, retrieved: list[RetrievedContextRecord]) -> str:
        if not retrieved:
            return "[No context retrieved]"
        parts = [f"[{i + 1}] {r.text}" for i, r in enumerate(retrieved)]
        combined = "\n\n".join(parts)
        if len(combined) > self.max_context_chars:
            combined = combined[: self.max_context_chars] + "…[truncated]"
        return combined

    def _call_once(self, prompt: str) -> tuple[int, str]:
        result = self._client.complete(
            messages=[{"role": "user", "content": prompt}],
            model=self.model,
            json_mode=True,
            temperature=0.0,
        )
        try:
            data = json.loads(result.text)
            score = max(1, min(5, int(data["score"])))
            reasoning = str(data.get("reasoning", ""))
            return score, reasoning
        except Exception:
            log.warning("Judge parse failed for response: %s", result.text[:200])
            return 3, ""

    def _score_metric(self, prompt: str) -> tuple[float, str]:
        """Score n_reps times and return (median score, last non-empty reasoning)."""
        scores: list[int] = []
        last_reasoning = ""
        for _ in range(self.n_reps):
            s, r = self._call_once(prompt)
            scores.append(s)
            if r:
                last_reasoning = r
        return float(median(scores)), last_reasoning

    def score(self, result: QuestionRunResult) -> QuestionScores:
        ctx = self._format_context(result.retrieved_context)

        f_score, f_reason = self._score_metric(
            _FAITHFULNESS_PROMPT.format(
                question=result.question,
                context=ctx,
                predicted_answer=result.predicted_answer,
            )
        )
        r_score, r_reason = self._score_metric(
            _ANSWER_RELEVANCY_PROMPT.format(
                question=result.question,
                predicted_answer=result.predicted_answer,
            )
        )
        c_score, c_reason = self._score_metric(
            _CONTEXT_RECALL_PROMPT.format(
                question=result.question,
                expected_answer=result.expected_answer,
                context=ctx,
            )
        )

        return QuestionScores(
            qa_id=result.qa_id,
            faithfulness=f_score,
            answer_relevancy=r_score,
            context_recall=c_score,
            faithfulness_reasoning=f_reason,
            answer_relevancy_reasoning=r_reason,
            context_recall_reasoning=c_reason,
        )

    # ── Batch API helpers ────────────────────────────────────────────────────

    def build_batch_requests(
        self, results: list[QuestionRunResult]
    ) -> list[dict[str, Any]]:
        """Build the flat list of batch requests for all results × metrics × reps."""
        requests: list[dict[str, Any]] = []
        for qr in results:
            ctx = self._format_context(qr.retrieved_context)
            prompts = {
                "f": _FAITHFULNESS_PROMPT.format(
                    question=qr.question, context=ctx, predicted_answer=qr.predicted_answer
                ),
                "r": _ANSWER_RELEVANCY_PROMPT.format(
                    question=qr.question, predicted_answer=qr.predicted_answer
                ),
                "c": _CONTEXT_RECALL_PROMPT.format(
                    question=qr.question, expected_answer=qr.expected_answer, context=ctx
                ),
            }
            for code, prompt in prompts.items():
                for rep in range(self.n_reps):
                    requests.append({
                        "custom_id": f"{qr.qa_id}:{code}:{rep}",
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": {
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "temperature": 0.0,
                            "response_format": {"type": "json_object"},
                        },
                    })
        return requests

    def parse_batch_results(
        self,
        batch_output: list[dict[str, Any]],
        results: list[QuestionRunResult],
    ) -> list[QuestionScores]:
        """Map batch output back to per-question scores using custom_ids."""
        # Index raw output by custom_id
        by_id: dict[str, dict[str, Any]] = {}
        for item in batch_output:
            cid = item.get("custom_id", "")
            if item.get("error") is not None:
                log.warning("Batch error for %s: %s", cid, item["error"])
                continue
            by_id[cid] = item.get("response", {}).get("body", {})

        scores: list[QuestionScores] = []
        for qr in results:
            metric_scores: dict[str, list[int]] = {"f": [], "r": [], "c": []}
            metric_reasons: dict[str, str] = {}

            for code in ("f", "r", "c"):
                for rep in range(self.n_reps):
                    cid = f"{qr.qa_id}:{code}:{rep}"
                    body = by_id.get(cid)
                    if body is None:
                        metric_scores[code].append(3)
                        continue
                    try:
                        text = body["choices"][0]["message"]["content"]
                        data = json.loads(text)
                        score = max(1, min(5, int(data["score"])))
                        metric_scores[code].append(score)
                        if data.get("reasoning"):
                            metric_reasons[code] = str(data["reasoning"])
                    except Exception:
                        metric_scores[code].append(3)

            scores.append(QuestionScores(
                qa_id=qr.qa_id,
                faithfulness=float(median(metric_scores["f"] or [3])),
                answer_relevancy=float(median(metric_scores["r"] or [3])),
                context_recall=float(median(metric_scores["c"] or [3])),
                faithfulness_reasoning=metric_reasons.get("f", ""),
                answer_relevancy_reasoning=metric_reasons.get("r", ""),
                context_recall_reasoning=metric_reasons.get("c", ""),
            ))

        return scores
