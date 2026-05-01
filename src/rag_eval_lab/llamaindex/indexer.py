"""LlamaIndex ingestion: chunk -> TextNode (pre-embedded) -> PineconeVectorStore."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pinecone import Pinecone

from rag_eval_lab.config.schema import ExperimentConfig
from rag_eval_lab.ingestion.chunker import Chunk, Chunker
from rag_eval_lab.ingestion.ingest import (
    _read_pdf_pages,
    _cache_path,
    _load_cache,
    _save_cache,
)
from rag_eval_lab.utils.io import sha256_of_file
from rag_eval_lab.utils.llm_client import LLMClient
from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

if TYPE_CHECKING:
    from llama_index.core import StorageContext, VectorStoreIndex
    from llama_index.core.schema import TextNode
    from llama_index.vector_stores.pinecone import PineconeVectorStore
    from llama_index.embeddings.openai import OpenAIEmbedding


@dataclass
class LlamaIndexResources:
    index: Any
    vector_store: Any


def _load_llamaindex_classes() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    try:
        from llama_index.core import StorageContext, VectorStoreIndex
        from llama_index.core.schema import TextNode
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.vector_stores.pinecone import PineconeVectorStore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "LlamaIndex dependencies are not installed. Install the project dependencies "
            "from pyproject.toml before running CP7."
        ) from exc
    return StorageContext, VectorStoreIndex, TextNode, PineconeVectorStore


def _make_embed_model(model_name: str) -> Any:
    _, _, _, _ = _load_llamaindex_classes()
    from llama_index.embeddings.openai import OpenAIEmbedding

    return OpenAIEmbedding(model=model_name)


def _build_text_nodes(chunks: list[Chunk], embeddings: list[list[float]]) -> list[Any]:
    _, _, TextNode, _ = _load_llamaindex_classes()
    return [
        TextNode(
            id_=chunk.chunk_id,
            text=chunk.text,
            metadata={
                "chunk_id": chunk.chunk_id,
                "source": chunk.source,
                "page": chunk.page or 0,
                "chunk_index": chunk.chunk_index,
                # Keep text accessible when loading the namespace later from the vector store.
                "text": chunk.text,
            },
            embedding=emb,
        )
        for chunk, emb in zip(chunks, embeddings, strict=True)
    ]


def build_llamaindex_index(
    config: ExperimentConfig,
    *,
    llm_client: LLMClient | None = None,
    rebuild: bool = False,
) -> LlamaIndexResources:
    """Return LlamaIndex resources backed by Pinecone.

    If the namespace is already populated, loads the existing index.
    Otherwise chunks the PDF, reuses the embedding cache, and ingests.
    """
    StorageContext, VectorStoreIndex, _, PineconeVectorStore = _load_llamaindex_classes()

    api_key = os.getenv("PINECONE_API_KEY", "")
    index_name = os.getenv("PINECONE_INDEX_NAME", "rag-eval-lab")
    namespace = config.experiment_id

    pc = Pinecone(api_key=api_key)
    pinecone_index = pc.Index(index_name)
    vector_store = PineconeVectorStore(
        pinecone_index=pinecone_index,
        namespace=namespace,
        text_key="text",
        remove_text_from_metadata=False,
    )
    embed_model = _make_embed_model(config.embedding.model)

    stats = pinecone_index.describe_index_stats()
    ns_info = stats.namespaces.get(namespace)
    n_vectors = ns_info.vector_count if ns_info else 0

    if n_vectors > 0 and not rebuild:
        log.info(
            "Namespace '%s' already has %d vectors — loading existing LlamaIndex index.",
            namespace,
            n_vectors,
        )
        return LlamaIndexResources(
            index=VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                embed_model=embed_model,
            ),
            vector_store=vector_store,
        )

    if n_vectors > 0 and rebuild:
        log.info("--rebuild: deleting namespace '%s'", namespace)
        pinecone_index.delete(delete_all=True, namespace=namespace)

    # Read → chunk
    pages = _read_pdf_pages(config.corpus)
    log.info("Loaded %d pages from %s", len(pages), config.corpus.name)

    chunker = Chunker(
        chunk_size=config.chunking.chunk_size,
        chunk_overlap=config.chunking.chunk_overlap,
        separators=config.chunking.separators,
    )
    chunks = chunker.split(pages, source=config.corpus.name)
    log.info("Created %d chunks (size=%d overlap=%d)", len(chunks), config.chunking.chunk_size, config.chunking.chunk_overlap)

    # Embed — reuse cache from vanilla pipeline when params match
    corpus_hash = sha256_of_file(config.corpus)
    cache_file = _cache_path(
        corpus_hash,
        config.embedding.model,
        config.chunking.chunk_size,
        config.chunking.chunk_overlap,
    )
    cached = _load_cache(cache_file)

    if cached is not None:
        embeddings = cached
        log.info("Using cached embeddings (%d vectors, $0.00)", len(embeddings))
    else:
        client = llm_client or LLMClient()
        texts = [c.text for c in chunks]
        log.info("Embedding %d chunks via %s...", len(chunks), config.embedding.model)
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), config.embedding.batch_size):
            batch = texts[i : i + config.embedding.batch_size]
            embeddings.extend(client.embed(batch, model=config.embedding.model))
        _save_cache(cache_file, chunks, embeddings)

    # Insert pre-embedded nodes — LlamaIndex skips re-embedding when node.embedding is set
    nodes = _build_text_nodes(chunks, embeddings)
    log.info("Ingesting %d nodes into namespace '%s'...", len(nodes), namespace)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
    )
    log.info("Ingestion complete: namespace '%s'", namespace)
    return LlamaIndexResources(index=index, vector_store=vector_store)
