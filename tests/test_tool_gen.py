#!/usr/bin/env python3
"""Unit tests for agentic tool-set generation."""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock


def _airline_tool_json() -> str:
    return json.dumps({
        "domain": "airline",
        "tools": [
            {
                "name": "search_flights",
                "description": "Search for available flights between two airports.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "origin": {"type": "string", "description": "IATA origin code"},
                        "destination": {"type": "string", "description": "IATA destination code"},
                    },
                    "required": ["origin", "destination"],
                },
                "mock_behavior": "Returns a list of flights with prices and seat availability.",
            }
        ],
    })


@pytest.mark.asyncio
async def test_generate_tool_set_returns_toolset():
    from agentic_qwen.data_synthesis.agentic.tool_gen import generate_tool_set, ToolSet

    client = MagicMock()
    client.chat_json = AsyncMock(return_value=_airline_tool_json())

    ts = await generate_tool_set("I am a frequent business traveller.", client)

    assert isinstance(ts, ToolSet)
    assert ts.domain == "airline"
    assert len(ts.tools) == 1
    assert ts.tools[0].name == "search_flights"
    assert ts.persona == "I am a frequent business traveller."


@pytest.mark.asyncio
async def test_generate_tool_set_retries_on_bad_json():
    from agentic_qwen.data_synthesis.agentic.tool_gen import generate_tool_set

    client = MagicMock()
    client.chat_json = AsyncMock(side_effect=[
        "not valid json",          # attempt 1
        _airline_tool_json(),      # attempt 2 succeeds
    ])

    ts = await generate_tool_set("Test persona", client)
    assert ts.domain == "airline"
    assert client.chat_json.call_count == 2


def test_tool_to_openai_schema():
    from agentic_qwen.data_synthesis.agentic.tool_gen import Tool

    tool = Tool({
        "name": "search_flights",
        "description": "Search for flights.",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"type": "string", "description": "Origin"}},
            "required": ["origin"],
        },
        "mock_behavior": "Returns flights.",
    })

    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search_flights"
    assert "parameters" in schema["function"]


def test_toolset_round_trip():
    from agentic_qwen.data_synthesis.agentic.tool_gen import Tool, ToolSet

    tools = [Tool(json.loads(_airline_tool_json())["tools"][0])]
    ts = ToolSet(domain="airline", tools=tools, persona="Test")
    restored = ToolSet.from_dict(ts.to_dict())

    assert restored.domain == ts.domain
    assert restored.persona == ts.persona
    assert len(restored.tools) == len(ts.tools)
    assert restored.tools[0].name == ts.tools[0].name
