"""Single chokepoint for all OpenAI calls.

All Chat Completions and Embeddings go through this module so we get one
place to: (a) retry on transient errors, (b) count tokens, (c) accumulate
cost, (d) swap providers later, (e) mock in tests.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any

import tiktoken
from openai import APIConnectionError, APIError, BadRequestError, OpenAI, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from rag_eval_lab.utils.logging import get_logger

log = get_logger(__name__)

# Approximate USD per 1M tokens. Update as prices change. Used only for
# logging an estimate — the source of truth is the OpenAI dashboard.
_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    # chat models: (prompt, completion)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    # embeddings: (input, 0)
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


@dataclass
class CompletionResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None
    raw: Any = field(repr=False, default=None)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass
class UsageTracker:
    """Accumulates token + cost totals for the lifetime of the client."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    embedding_tokens: int = 0
    estimated_cost_usd: float = 0.0
    n_chat_calls: int = 0
    n_embed_calls: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add_chat(self, model: str, prompt: int, completion: int) -> float:
        cost = _estimate_cost(model, prompt, completion)
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.estimated_cost_usd += cost
            self.n_chat_calls += 1
        return cost

    def add_embed(self, model: str, tokens: int) -> float:
        cost = _estimate_cost(model, tokens, 0)
        with self._lock:
            self.embedding_tokens += tokens
            self.estimated_cost_usd += cost
            self.n_embed_calls += 1
        return cost


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    price = _PRICING_USD_PER_1M.get(model)
    if price is None:
        return 0.0
    p_in, p_out = price
    return (prompt_tokens * p_in + completion_tokens * p_out) / 1_000_000


def _encoder_for(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    return len(_encoder_for(model).encode(text))


_RETRYABLE = (RateLimitError, APIConnectionError)


def _uses_max_completion_tokens(model: str) -> bool:
    return model.startswith("gpt-5")


class LLMClient:
    """Thin wrapper over openai.OpenAI with retry, usage tracking, and json_mode."""

    def __init__(
        self,
        client: OpenAI | None = None,
        api_key: str | None = None,
        max_retries: int = 3,
    ) -> None:
        if client is None:
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise RuntimeError(
                    "OPENAI_API_KEY not set. Copy .env.example to .env and fill it in."
                )
            client = OpenAI(api_key=key)
        self._client = client
        self._max_retries = max_retries
        self.usage = UsageTracker()

    @property
    def raw(self) -> OpenAI:
        return self._client

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4o-mini",
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        seed: int | None = None,
        **extra: Any,
    ) -> CompletionResult:
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(_RETRYABLE),
            before_sleep=before_sleep_log(log, 30),  # WARNING
            reraise=True,
        )
        def _call() -> Any:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens is not None:
                token_key = "max_completion_tokens" if _uses_max_completion_tokens(model) else "max_tokens"
                kwargs[token_key] = max_tokens
            if seed is not None:
                kwargs["seed"] = seed
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            kwargs.update(extra)
            return self._client.chat.completions.create(**kwargs)

        try:
            resp = _call()
        except BadRequestError:
            raise
        choice = resp.choices[0]
        usage = resp.usage
        prompt_t = getattr(usage, "prompt_tokens", 0) or 0
        comp_t = getattr(usage, "completion_tokens", 0) or 0
        cost = self.usage.add_chat(model, prompt_t, comp_t)
        log.debug(
            "chat model=%s prompt=%d completion=%d ~$%.5f finish=%s",
            model, prompt_t, comp_t, cost, choice.finish_reason,
        )
        return CompletionResult(
            text=choice.message.content or "",
            model=model,
            prompt_tokens=prompt_t,
            completion_tokens=comp_t,
            finish_reason=choice.finish_reason,
            raw=resp,
        )

    def embed(
        self,
        texts: list[str],
        model: str = "text-embedding-3-small",
    ) -> list[list[float]]:
        @retry(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=20),
            retry=retry_if_exception_type(_RETRYABLE),
            before_sleep=before_sleep_log(log, 30),
            reraise=True,
        )
        def _call() -> Any:
            return self._client.embeddings.create(model=model, input=texts)

        resp = _call()
        tokens = getattr(resp.usage, "total_tokens", 0) or 0
        cost = self.usage.add_embed(model, tokens)
        log.debug("embed model=%s n=%d tokens=%d ~$%.5f", model, len(texts), tokens, cost)
        return [d.embedding for d in resp.data]

    # ── Batch API ────────────────────────────────────────────────────────────

    def submit_batch(
        self,
        requests: list[dict[str, Any]],
        description: str = "",
    ) -> str:
        """Upload requests as a JSONL file and create a batch job. Returns batch_id.

        Each request must follow the OpenAI batch request schema:
          {"custom_id": "<id>", "method": "POST", "url": "/v1/chat/completions", "body": {...}}
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            tmp_path = f.name
            for req in requests:
                f.write(json.dumps(req) + "\n")

        try:
            with open(tmp_path, "rb") as f:
                uploaded = self._client.files.create(file=f, purpose="batch")
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        log.info("Uploaded batch file: %s  (%d requests)", uploaded.id, len(requests))

        meta = {"description": description} if description else {}
        batch = self._client.batches.create(
            input_file_id=uploaded.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata=meta,
        )
        log.info("Batch created: %s  (status=%s)", batch.id, batch.status)
        return batch.id

    def wait_for_batch(self, batch_id: str, poll_interval: int = 30) -> Any:
        """Poll until the batch reaches a terminal state. Returns the final batch object."""
        terminal = {"completed", "failed", "expired", "cancelled"}
        while True:
            batch = self._client.batches.retrieve(batch_id)
            counts = batch.request_counts
            total = getattr(counts, "total", "?")
            completed = getattr(counts, "completed", 0)
            failed = getattr(counts, "failed", 0)
            log.info(
                "Batch %s  status=%-12s  %s/%s completed  %s failed",
                batch_id, batch.status, completed, total, failed,
            )
            if batch.status in terminal:
                return batch
            time.sleep(poll_interval)

    def fetch_batch_results(self, batch: Any) -> list[dict[str, Any]]:
        """Download and parse the JSONL output of a completed batch."""
        if batch.status != "completed":
            raise RuntimeError(
                f"Batch {batch.id} ended with status={batch.status!r}. "
                "Check the OpenAI dashboard for details."
            )
        raw = self._client.files.content(batch.output_file_id).text
        return [json.loads(line) for line in raw.splitlines() if line.strip()]
