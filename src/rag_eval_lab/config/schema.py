from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(gt=0, le=8192)
    chunk_overlap: int = Field(ge=0)
    separators: list[str] | None = None

    @model_validator(mode="after")
    def overlap_lt_size(self) -> "ChunkingConfig":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be < chunk_size ({self.chunk_size})"
            )
        return self


class EmbeddingConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    model: str = "text-embedding-3-small"
    batch_size: int = Field(default=100, gt=0, le=2048)


class RetrievalConfig(BaseModel):
    top_k: int = Field(default=5, gt=0, le=50)
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class GenerationConfig(BaseModel):
    provider: Literal["openai"] = "openai"
    model: str = "gpt-5.4-mini"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, gt=0, le=4096)
    system_prompt: str = (
        "You are an assistant that answers questions using ONLY the provided context. "
        "If the context is insufficient, say 'There is not enough information in the context.'"
    )


class ExperimentConfig(BaseModel):
    experiment_id: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    corpus: Path
    benchmark: Path | None = None
    chunking: ChunkingConfig
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    @field_validator("corpus", mode="after")
    @classmethod
    def corpus_must_exist(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"corpus path does not exist: {v}")
        return v

    @field_validator("benchmark", mode="after")
    @classmethod
    def benchmark_must_exist_if_set(cls, v: Path | None) -> Path | None:
        if v is not None and not v.exists():
            raise ValueError(f"benchmark path does not exist: {v}")
        return v
