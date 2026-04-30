from __future__ import annotations

from rag_eval_lab.rag.retriever import RetrievedChunk
from rag_eval_lab.utils.llm_client import CompletionResult, LLMClient

ANSWER_PROMPT = """\
Use only the retrieved context below to answer the question.
If the context is insufficient, say that there is not enough information in the context.

Retrieved context:
{context}

Question: {question}

Answer:
"""


class Answerer:
    def __init__(
        self,
        llm_client: LLMClient,
        *,
        model: str = "gpt-4o-mini",
        system_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._client = llm_client
        self._model = model
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    def answer(self, question: str, context: list[RetrievedChunk]) -> CompletionResult:
        formatted_context = _format_context(context)
        return self._client.complete(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {
                    "role": "user",
                    "content": ANSWER_PROMPT.format(
                        context=formatted_context,
                        question=question,
                    ),
                },
            ],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )


def _format_context(context: list[RetrievedChunk]) -> str:
    if not context:
        return "[no retrieved context]"

    return "\n\n".join(
        (
            f"[{idx}] chunk_id={chunk.chunk_id} "
            f"source={chunk.source or 'unknown'} "
            f"page={chunk.page if chunk.page is not None else 'unknown'} "
            f"score={chunk.score:.3f}\n"
            f"{chunk.text}"
        )
        for idx, chunk in enumerate(context, start=1)
    )
