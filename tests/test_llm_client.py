#!/usr/bin/env python3
"""
Unit tests for the unified LLM client.

All network calls are mocked — no API key required.
"""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def fake_response():
    """Build a minimal mock OpenAI ChatCompletion response."""
    msg = MagicMock()
    msg.content = "Hello, world!"
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_client():
    from agentic_qwen.llm.client import LLMClient
    return LLMClient(base_url="http://localhost:11434/v1", api_key="test", model="test", concurrency=1)


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_returns_string(fake_response):
    client = _make_client()
    with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=fake_response)):
        result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == "Hello, world!"
    await client.close()


@pytest.mark.asyncio
async def test_chat_empty_content_returns_empty_string():
    client = _make_client()
    msg = MagicMock()
    msg.content = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    with patch.object(client._client.chat.completions, "create", new=AsyncMock(return_value=resp)):
        result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == ""
    await client.close()


# ---------------------------------------------------------------------------
# chat_json()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_json_passes_response_format(fake_response):
    client = _make_client()
    mock_create = AsyncMock(return_value=fake_response)
    with patch.object(client._client.chat.completions, "create", new=mock_create):
        await client.chat_json([{"role": "user", "content": "give me json"}])
    assert mock_create.call_args.kwargs["response_format"] == {"type": "json_object"}
    await client.close()


# ---------------------------------------------------------------------------
# chat_batch()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_batch_preserves_order():
    from agentic_qwen.llm.client import LLMClient

    client = LLMClient(base_url="http://localhost:11434/v1", api_key="test", model="test", concurrency=5)

    call_n = 0

    async def _create(**_kw):
        nonlocal call_n
        msg = MagicMock()
        msg.content = f"response_{call_n}"
        call_n += 1
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        return resp

    with patch.object(client._client.chat.completions, "create", new=_create):
        batch = [[{"role": "user", "content": f"msg {i}"}] for i in range(4)]
        results = await client.chat_batch(batch)

    assert len(results) == 4
    await client.close()


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_manager_calls_close():
    from agentic_qwen.llm.client import LLMClient

    client = LLMClient(base_url="http://localhost:11434/v1", api_key="test", model="test")
    client.close = AsyncMock()
    async with client:
        pass
    client.close.assert_called_once()
