"""LlamaIndex retriever returning the same RetrievedChunk contract as the vanilla pipeline."""

from __future__ import annotations

from typing import Any

from rag_eval_lab.ingestion.embedder import Embedder
from rag_eval_lab.llamaindex.indexer import LlamaIndexResources
from rag_eval_lab.rag.retriever import RetrievedChunk


class LlamaIndexRetriever:
    def __init__(
        self,
        resources: LlamaIndexResources,
        embedder: Embedder,
        top_k: int,
        score_threshold: float | None = None,
    ) -> None:
        self._resources = resources
        self._embedder = embedder
        self._top_k = top_k
        self._score_threshold = score_threshold or 0.0

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        similarities, nodes = self._query_vector_store(question)
        results: list[RetrievedChunk] = []

        for node, similarity in zip(nodes, similarities, strict=True):
            score = float(similarity if similarity is not None else 0.0)
            if similarity is not None and score < self._score_threshold:
                continue
            results.append(
                RetrievedChunk(
                    chunk_id=getattr(node, "node_id", getattr(node, "id_", "")),
                    text=node.get_content() or node.metadata.get("text", ""),
                    score=score,
                    source=node.metadata.get("source", ""),
                    page=node.metadata.get("page"),
                )
            )
        return results

    def _query_vector_store(self, question: str) -> tuple[list[float | None], list[Any]]:
        from llama_index.core.vector_stores.types import VectorStoreQuery

        query_embedding = self._embedder.embed([question])[0]
        result = self._resources.vector_store.query(
            VectorStoreQuery(
                query_embedding=query_embedding,
                similarity_top_k=self._top_k,
            )
        )
        similarities = list(result.similarities or [])
        nodes = list(result.nodes or [])
        if len(similarities) < len(nodes):
            similarities.extend([None] * (len(nodes) - len(similarities)))
        return similarities, nodes
