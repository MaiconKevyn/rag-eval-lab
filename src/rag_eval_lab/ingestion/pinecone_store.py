from __future__ import annotations

import os

from pinecone import Pinecone

from rag_eval_lab.ingestion.chunker import Chunk
from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

_UPSERT_BATCH = 100  # Pinecone rate-limit safe


class PineconeStore:
    def __init__(self, index_name: str, dim: int, api_key: str | None = None) -> None:
        key = api_key or os.getenv("PINECONE_API_KEY")
        if not key:
            raise RuntimeError("PINECONE_API_KEY not set.")
        self._pc = Pinecone(api_key=key)
        self._index_name = index_name
        self._dim = dim
        self._index = self._pc.Index(index_name)
        self._validate_dimension()

    def _validate_dimension(self) -> None:
        """Guard against silent mismatches when swapping embedding models."""
        desc = self._pc.describe_index(self._index_name)
        remote_dim = desc.dimension
        if remote_dim != self._dim:
            raise ValueError(
                f"Dimension mismatch: index '{self._index_name}' has dim={remote_dim}, "
                f"but embedder expects dim={self._dim}. "
                "Re-create the index or switch the embedding model."
            )

    def upsert(
        self,
        namespace: str,
        chunks: list[Chunk],
        embeddings: list[list[float]],
    ) -> None:
        vectors = [
            {
                "id": chunk.chunk_id,
                "values": emb,
                "metadata": {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_index": chunk.chunk_index,
                    "text": chunk.text,
                },
            }
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]

        for i in range(0, len(vectors), _UPSERT_BATCH):
            batch = vectors[i : i + _UPSERT_BATCH]
            self._index.upsert(vectors=batch, namespace=namespace)
            log.debug("upserted %d/%d vectors to namespace=%s", i + len(batch), len(vectors), namespace)

        log.info("Upsert complete: %d vectors → namespace '%s'", len(vectors), namespace)

    def query(
        self,
        namespace: str,
        vector: list[float],
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[dict]:
        resp = self._index.query(
            namespace=namespace,
            vector=vector,
            top_k=top_k,
            include_metadata=True,
        )
        matches = [
            {
                "chunk_id": m.metadata.get("chunk_id", m.id),
                "text": m.metadata.get("text", ""),
                "source": m.metadata.get("source", ""),
                "page": m.metadata.get("page"),
                "score": m.score,
            }
            for m in resp.matches
        ]
        if score_threshold is not None:
            matches = [m for m in matches if m["score"] >= score_threshold]
        return matches

    def delete_namespace(self, namespace: str) -> None:
        self._index.delete(delete_all=True, namespace=namespace)
        log.info("Deleted all vectors in namespace '%s'", namespace)

    def list_namespaces(self) -> list[str]:
        stats = self._index.describe_index_stats()
        return list(stats.namespaces.keys())

    def namespace_vector_count(self, namespace: str) -> int:
        stats = self._index.describe_index_stats()
        ns = stats.namespaces.get(namespace)
        return ns.vector_count if ns else 0
