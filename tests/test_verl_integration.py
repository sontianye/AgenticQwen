#!/usr/bin/env python3
"""
Unit tests for the verl integration module.
"""

from __future__ import annotations

import pytest
from agentic_qwen.training.verl_integration import (
    compute_score,
    extract_tool_calls,
    format_tool_result,
    is_terminal,
)


# ---------------------------------------------------------------------------
# compute_score dispatch
# ---------------------------------------------------------------------------

def test_compute_score_dispatches_to_math_for_reasoning():
    score = compute_score(
        "hotpotqa",
        "<answer>Paris</answer>",
        "Paris",
        {"data_type": "reasoning"},
    )
    assert score == 1.0


def test_compute_score_dispatches_to_agentic():
    score = compute_score(
        "tau2",
        "Task complete.\n###STOP",
        "",
        {"data_type": "agentic", "rubrics": []},
    )
    assert 0.0 <= score <= 1.0


def test_compute_score_defaults_to_reasoning():
    """When data_type is absent, should default to reasoning scoring."""
    score = compute_score("unknown", "<answer>42</answer>", "42")
    assert score == 1.0


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------

def test_extract_tool_calls_single():
    text = '<tool_call>{"name": "search_flights", "arguments": {"origin": "SFO"}}</tool_call>'
    calls = extract_tool_calls(text)
    assert len(calls) == 1
    assert calls[0]["name"] == "search_flights"
    assert calls[0]["arguments"]["origin"] == "SFO"


def test_extract_tool_calls_multiple():
    text = (
        '<tool_call>{"name": "tool_a", "arguments": {}}</tool_call>'
        "Some text."
        '<tool_call>{"name": "tool_b", "arguments": {"x": 1}}</tool_call>'
    )
    calls = extract_tool_calls(text)
    assert len(calls) == 2
    assert calls[0]["name"] == "tool_a"
    assert calls[1]["name"] == "tool_b"


def test_extract_tool_calls_malformed_skipped():
    text = "<tool_call>not valid json</tool_call>"
    calls = extract_tool_calls(text)
    assert calls == []


def test_extract_tool_calls_empty():
    assert extract_tool_calls("No tool calls here.") == []


# ---------------------------------------------------------------------------
# Tool result formatting
# ---------------------------------------------------------------------------

def test_format_tool_result_contains_name_and_result():
    result = {"flights": [{"id": "UA123", "price": 450}]}
    formatted = format_tool_result("search_flights", result)
    assert "search_flights" in formatted
    assert "UA123" in formatted
    assert "<tool_response>" in formatted
    assert "</tool_response>" in formatted


# ---------------------------------------------------------------------------
# Terminal detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("I have completed the task.\n###STOP", True),
    ("Escalating.\n###TRANSFER_TO_HUMAN", True),
    ("Still working...", False),
    ("", False),
])
def test_is_terminal(text: str, expected: bool):
    assert is_terminal(text) is expected
