"""Smoke tests for Checkpoint 0 — proves utils + mock fixture wire up correctly."""

from __future__ import annotations

import json
from pathlib import Path

from rag_eval_lab.utils.io import read_json, sha256_of_file, write_json
from rag_eval_lab.utils.llm_client import LLMClient, estimate_tokens


def test_sha256_of_file_is_stable(tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("hello world", encoding="utf-8")
    h1 = sha256_of_file(p)
    h2 = sha256_of_file(p)
    assert h1 == h2
    assert len(h1) == 64


def test_json_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "nested" / "data.json"
    payload = {"a": 1, "b": [1, 2, 3], "c": "snowman ☃"}
    write_json(p, payload)
    assert read_json(p) == payload
    # Non-ascii preserved (ensure_ascii=False)
    assert "☃" in p.read_text(encoding="utf-8")


def test_estimate_tokens_returns_positive() -> None:
    assert estimate_tokens("hello world") > 0


def test_llm_client_complete_with_mock(mock_openai_client) -> None:
    mock_openai_client.set_chat_response('{"score": 4}', prompt_tokens=12, completion_tokens=4)
    client = LLMClient(client=mock_openai_client)

    result = client.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-4o-mini",
        json_mode=True,
    )

    assert json.loads(result.text) == {"score": 4}
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 4
    assert client.usage.n_chat_calls == 1
    assert client.usage.estimated_cost_usd > 0

    # json_mode must propagate as response_format=json_object
    _, kwargs = mock_openai_client.chat.completions.create.call_args
    assert kwargs["response_format"] == {"type": "json_object"}
    assert kwargs["temperature"] == 0.0


def test_llm_client_uses_max_completion_tokens_for_gpt5_family(mock_openai_client) -> None:
    mock_openai_client.set_chat_response("ok")
    client = LLMClient(client=mock_openai_client)

    client.complete(
        messages=[{"role": "user", "content": "hi"}],
        model="gpt-5.4-mini",
        max_tokens=123,
    )

    _, kwargs = mock_openai_client.chat.completions.create.call_args
    assert "max_completion_tokens" in kwargs
    assert kwargs["max_completion_tokens"] == 123
    assert "max_tokens" not in kwargs


def test_llm_client_embed_with_mock(mock_openai_client) -> None:
    mock_openai_client.set_embedding_response([[0.1, 0.2], [0.3, 0.4]], total_tokens=8)
    client = LLMClient(client=mock_openai_client)

    vectors = client.embed(["a", "b"], model="text-embedding-3-small")

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]
    assert client.usage.embedding_tokens == 8
    assert client.usage.n_embed_calls == 1
