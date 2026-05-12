#!/usr/bin/env python3
"""
Unit tests for the consistency filter and persona injector.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Consistency filter
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_consistency_passes_when_all_agree():
    from agentic_qwen.data_synthesis.reasoning.consistency import check_consistency

    client = MagicMock()
    # All 3 samples return the same answer
    client.chat = AsyncMock(return_value="Step 1: ...\n<answer>Paris</answer>")

    result = await check_consistency("What is the capital of France?", "Paris", client, n=3)

    assert result["pass"] is True
    assert result["all_same"] is True
    assert result["correct"] is True
    assert len(result["answers"]) == 3


@pytest.mark.asyncio
async def test_consistency_fails_when_answers_diverge():
    from agentic_qwen.data_synthesis.reasoning.consistency import check_consistency

    client = MagicMock()
    answers = ["<answer>Paris</answer>", "<answer>London</answer>", "<answer>Berlin</answer>"]
    call_n = 0

    async def _chat(messages, **_kw):
        nonlocal call_n
        ans = answers[call_n % len(answers)]
        call_n += 1
        return ans

    client.chat = _chat

    result = await check_consistency("What is the capital?", "Paris", client, n=3)

    assert result["pass"] is False
    assert result["all_same"] is False


@pytest.mark.asyncio
async def test_filter_batch_keeps_only_passing():
    from agentic_qwen.data_synthesis.reasoning.consistency import filter_batch

    client = MagicMock()
    call_count = 0

    # First sample: all agree ("42"); second sample: diverge
    async def _chat(messages, **_kw):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return "<answer>42</answer>"
        return f"<answer>{call_count}</answer>"  # always different

    client.chat = _chat

    samples = [
        {"question": "6×7?", "answer": "42"},
        {"question": "capital of Neverland?", "answer": "Neverland City"},
    ]
    filtered = await filter_batch(samples, client, n=3)

    # Only the first sample should pass
    assert len(filtered) == 1
    assert filtered[0]["question"] == "6×7?"


# ---------------------------------------------------------------------------
# Persona injector
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_persona_injector_returns_question_and_answer():
    from agentic_qwen.data_synthesis.reasoning.persona import PersonaInjector
    from unittest.mock import patch

    injector = MagicMock(spec=PersonaInjector)
    injector._pool = ["A software engineer at a fintech startup."]
    injector.sample = MagicMock(return_value="A software engineer at a fintech startup.")

    client = MagicMock()
    client.chat_json = AsyncMock(return_value=json.dumps({
        "question": "A fintech engineer needs to calculate 6×7 for a budget model. What is it?",
        "answer": "42",
    }))

    with patch.object(PersonaInjector, "__init__", lambda self, _: None):
        inj = PersonaInjector.__new__(PersonaInjector)
        inj._pool = ["A software engineer at a fintech startup."]
        inj.sample = lambda: "A software engineer at a fintech startup."

        result = await inj.inject("6×7?", "42", client)

    assert "question" in result
    assert result["answer"] == "42"
    assert "persona" in result


@pytest.mark.asyncio
async def test_persona_injector_falls_back_on_error():
    from agentic_qwen.data_synthesis.reasoning.persona import PersonaInjector

    with patch.object(PersonaInjector, "__init__", lambda self, _: None):
        inj = PersonaInjector.__new__(PersonaInjector)
        inj._pool = ["Test persona"]
        inj.sample = lambda: "Test persona"

    client = MagicMock()
    client.chat_json = AsyncMock(side_effect=Exception("API error"))

    result = await inj.inject("What is 2+2?", "4", client)

    # Falls back to original question/answer
    assert result["question"] == "What is 2+2?"
    assert result["answer"] == "4"
