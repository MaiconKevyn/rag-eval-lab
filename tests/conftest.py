"""Shared fixtures for the test suite.

The `mock_openai_client` fixture returns an object that quacks like
`openai.OpenAI`. Tests pass it directly into `LLMClient(client=...)` and
program canned responses via `mock.set_chat_response(...)` /
`mock.set_embedding_response(...)`.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _build_chat_response(
    content: str,
    *,
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
    finish_reason: str = "stop",
) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _build_embedding_response(
    vectors: list[list[float]],
    *,
    total_tokens: int | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        data=[SimpleNamespace(embedding=v, index=i) for i, v in enumerate(vectors)],
        usage=SimpleNamespace(total_tokens=total_tokens or sum(len(v) for v in vectors)),
    )


class FakeOpenAIClient:
    """Quacks like openai.OpenAI for tests."""

    def __init__(self) -> None:
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = MagicMock(
            return_value=_build_chat_response('{"ok": true}')
        )
        self.embeddings = MagicMock()
        self.embeddings.create = MagicMock(
            return_value=_build_embedding_response([[0.1, 0.2, 0.3]])
        )

    def set_chat_response(
        self,
        content: str,
        *,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
        finish_reason: str = "stop",
    ) -> None:
        self.chat.completions.create.return_value = _build_chat_response(
            content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )

    def set_chat_responses(self, contents: list[str]) -> None:
        """Round-robin a list of responses across successive calls."""
        self.chat.completions.create.side_effect = [
            _build_chat_response(c) for c in contents
        ]

    def set_embedding_response(
        self, vectors: list[list[float]], *, total_tokens: int | None = None
    ) -> None:
        self.embeddings.create.return_value = _build_embedding_response(
            vectors, total_tokens=total_tokens
        )


@pytest.fixture
def mock_openai_client() -> FakeOpenAIClient:
    return FakeOpenAIClient()
