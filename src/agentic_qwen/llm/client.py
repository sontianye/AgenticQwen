"""
Unified LLM client supporting any OpenAI-compatible API.

Designed for high-throughput data synthesis:
- Async-first with configurable concurrency
- Automatic retry with exponential back-off (tenacity)
- JSON-mode helper
- Configuration via ``configs/llm.yaml`` + environment variable substitution
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()
logger = logging.getLogger(__name__)

_RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _load_config(path: str = "configs/llm.yaml") -> dict[str, Any]:
    """Load YAML config with ``${ENV_VAR}`` placeholder resolution."""
    with open(path) as fh:
        raw = fh.read()
    raw = re.sub(r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), ""), raw)
    return yaml.safe_load(raw)


# ---------------------------------------------------------------------------
# LLMClient
# ---------------------------------------------------------------------------

class LLMClient:
    """Async wrapper around the OpenAI SDK.

    Supports any OpenAI-compatible endpoint (Volcano ARK, DeepSeek, DashScope …).

    Usage::

        client = LLMClient.from_config("teacher")
        async with client:
            text = await client.chat([{"role": "user", "content": "Hello!"}])
            data = await client.chat_json(messages)  # forces JSON output
            texts = await client.chat_batch(list_of_messages)
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout: float = 120.0,
        max_retries: int = 3,
        concurrency: int = 20,
    ) -> None:
        self.model = model
        self._max_retries = max_retries
        self._sem = asyncio.Semaphore(concurrency)
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,  # retries managed by tenacity
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, role: str = "teacher", config_path: str = "configs/llm.yaml") -> "LLMClient":
        """Instantiate from ``configs/llm.yaml`` (role = ``teacher`` | ``student``)."""
        cfg = _load_config(config_path)[role]
        return cls(
            base_url=cfg["base_url"],
            api_key=cfg["api_key"],
            model=cfg["model"],
            timeout=cfg.get("timeout", 120.0),
            max_retries=cfg.get("max_retries", 3),
            concurrency=cfg.get("concurrency", 20),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        response_format: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> str:
        """Send a single chat completion and return the assistant content."""
        async with self._sem:
            return await self._chat_with_retry(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=response_format,
                **kwargs,
            )

    async def chat_json(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> str:
        """Like :meth:`chat` but enforces ``response_format=json_object``.

        The caller is responsible for ``json.loads``-ing the result.
        """
        return await self.chat(
            messages,
            response_format={"type": "json_object"},
            **kwargs,
        )

    async def chat_batch(
        self,
        batch: list[list[dict[str, str]]],
        **kwargs: Any,
    ) -> list[str]:
        """Concurrently process a batch of message lists, preserving order."""
        return list(await asyncio.gather(*[self.chat(msgs, **kwargs) for msgs in batch]))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    async def _chat_with_retry(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        response_format: dict[str, str] | None,
        **kwargs: Any,
    ) -> str:
        extra: dict[str, Any] = {}
        if response_format:
            extra["response_format"] = response_format
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
            **kwargs,
        )
        return resp.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        """Release underlying HTTP connections."""
        await self._client.close()
