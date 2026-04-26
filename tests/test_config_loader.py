"""Unit tests for config schema and loader.

Runs offline — no API calls, no filesystem side effects beyond tmp_path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from rag_eval_lab.config.loader import load_config
from rag_eval_lab.config.schema import ChunkingConfig, ExperimentConfig


# ── ChunkingConfig validation ───────────────────────────────────────────────

class TestChunkingConfig:
    def test_valid_config(self) -> None:
        cfg = ChunkingConfig(chunk_size=256, chunk_overlap=32)
        assert cfg.chunk_size == 256
        assert cfg.chunk_overlap == 32

    def test_overlap_equal_to_size_raises(self) -> None:
        with pytest.raises(Exception, match="chunk_overlap"):
            ChunkingConfig(chunk_size=100, chunk_overlap=100)

    def test_overlap_greater_than_size_raises(self) -> None:
        with pytest.raises(Exception, match="chunk_overlap"):
            ChunkingConfig(chunk_size=100, chunk_overlap=150)

    def test_zero_overlap_is_valid(self) -> None:
        cfg = ChunkingConfig(chunk_size=256, chunk_overlap=0)
        assert cfg.chunk_overlap == 0

    def test_chunk_size_zero_raises(self) -> None:
        with pytest.raises(Exception):
            ChunkingConfig(chunk_size=0, chunk_overlap=0)


# ── ExperimentConfig ID pattern ─────────────────────────────────────────────

class TestExperimentConfigId:
    def _base_kwargs(self, corpus: Path) -> dict:
        return {
            "experiment_id": "exp_001",
            "description": "test",
            "corpus": corpus,
            "chunking": {"chunk_size": 256, "chunk_overlap": 32},
        }

    def test_valid_id(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"")
        cfg = ExperimentConfig(**self._base_kwargs(pdf))
        assert cfg.experiment_id == "exp_001"

    def test_id_with_uppercase_raises(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"")
        with pytest.raises(Exception):
            ExperimentConfig(**{**self._base_kwargs(pdf), "experiment_id": "Exp_001"})

    def test_id_with_space_raises(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"")
        with pytest.raises(Exception):
            ExperimentConfig(**{**self._base_kwargs(pdf), "experiment_id": "exp 001"})

    def test_corpus_nonexistent_raises(self, tmp_path: Path) -> None:
        with pytest.raises(Exception):
            ExperimentConfig(**{**self._base_kwargs(tmp_path / "missing.pdf")})


# ── load_config from YAML ────────────────────────────────────────────────────

class TestLoadConfig:
    def _write_yaml(self, tmp_path: Path, data: dict) -> Path:
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(data), encoding="utf-8")
        return p

    def _minimal_data(self, corpus: Path) -> dict:
        return {
            "experiment_id": "exp_001",
            "description": "test experiment",
            "corpus": str(corpus),
            "chunking": {"chunk_size": 256, "chunk_overlap": 32},
        }

    def test_load_minimal_config(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"")
        p = self._write_yaml(tmp_path, self._minimal_data(pdf))
        cfg = load_config(p)
        assert cfg.experiment_id == "exp_001"
        assert cfg.chunking.chunk_size == 256

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.yaml"
        p.write_text("experiment_id: [unclosed", encoding="utf-8")
        with pytest.raises(Exception):
            load_config(p)

    def test_invalid_overlap_in_yaml_raises(self, tmp_path: Path) -> None:
        """Pydantic must catch overlap >= chunk_size before any API call."""
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"")
        data = self._minimal_data(pdf)
        data["chunking"] = {"chunk_size": 100, "chunk_overlap": 100}
        p = self._write_yaml(tmp_path, data)
        with pytest.raises(Exception, match="chunk_overlap"):
            load_config(p)

    def test_defaults_are_applied(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"")
        p = self._write_yaml(tmp_path, self._minimal_data(pdf))
        cfg = load_config(p)
        assert cfg.embedding.model == "text-embedding-3-small"
        assert cfg.retrieval.top_k == 5
        assert cfg.generation.temperature == 0.0

    def test_missing_benchmark_when_declared_raises(self, tmp_path: Path) -> None:
        pdf = tmp_path / "doc.pdf"
        pdf.write_bytes(b"")
        data = self._minimal_data(pdf)
        data["benchmark"] = str(tmp_path / "missing.json")
        p = self._write_yaml(tmp_path, data)
        with pytest.raises(Exception, match="benchmark path does not exist"):
            load_config(p)
