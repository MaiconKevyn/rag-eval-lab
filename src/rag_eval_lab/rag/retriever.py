from __future__ import annotations

from dataclasses import dataclass

from rag_eval_lab.ingestion.embedder import Embedder
from rag_eval_lab.ingestion.pinecone_store import PineconeStore


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: str = ""
    page: int | None = None


class Retriever:
    def __init__(
        self,
        store: PineconeStore,
        embedder: Embedder,
        namespace: str,
        top_k: int,
        score_threshold: float | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._namespace = namespace
        self._top_k = top_k
        self._score_threshold = score_threshold

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        vector = self._embedder.embed([question])[0]
        matches = self._store.query(
            namespace=self._namespace,
            vector=vector,
            top_k=self._top_k,
            score_threshold=self._score_threshold,
        )
        return [
            RetrievedChunk(
                chunk_id=match["chunk_id"],
                text=match.get("text", ""),
                score=float(match["score"]),
                source=match.get("source", ""),
                page=match.get("page"),
            )
            for match in matches
        ]

