from __future__ import annotations

import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from rag_eval_lab.utils.io import read_json, write_json


class QAPair(BaseModel):
    qa_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str
    expected_answer: str
    question_type: Literal["factual", "comparative", "why", "how"]
    source_chunk_id: str


class BenchmarkDataset(BaseModel):
    version: str = "v1"
    corpus_hash: str
    created_at: str          # ISO 8601
    generator_model: str
    n_per_chunk: int
    n_qa_pairs: int
    qa_pairs: list[QAPair]

    def save(self, directory: str | Path) -> Path:
        date = self.created_at[:10]
        hash_prefix = self.corpus_hash[:8]
        filename = f"benchmark_{self.version}_{hash_prefix}_{date}.json"
        path = Path(directory) / filename
        write_json(path, self.model_dump())
        return path

    @classmethod
    def load(cls, path: str | Path) -> "BenchmarkDataset":
        return cls.model_validate(read_json(path))

    def question_type_distribution(self) -> dict[str, int]:
        counts: dict[str, int] = {"factual": 0, "comparative": 0, "why": 0, "how": 0}
        for qa in self.qa_pairs:
            counts[qa.question_type] = counts.get(qa.question_type, 0) + 1
        return counts
