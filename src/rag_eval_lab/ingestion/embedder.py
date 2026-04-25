from __future__ import annotations

from typing import Protocol, runtime_checkable

from rag_eval_lab.utils.llm_client import LLMClient

# Dimensions for the models supported by this project.
_MODEL_DIMS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}


@runtime_checkable
class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    @property
    def dim(self) -> int: ...

    @property
    def model_name(self) -> str: ...


class OpenAIEmbedder:
    """Batching embedder backed by LLMClient (inherits retry + cost tracking)."""

    def __init__(
        self,
        client: LLMClient,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
    ) -> None:
        self._client = client
        self._model = model
        self._batch_size = batch_size
        self._dim = _MODEL_DIMS.get(model, 1536)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._client.embed(batch, model=self._model))
        return results
