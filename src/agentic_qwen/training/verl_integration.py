"""
verl integration layer.

This module bridges AgenticQwen's reward functions with verl's trainer
infrastructure.  It also defines the agent-loop configuration for
multi-turn tool-use rollouts during RL training.

verl calls ``compute_score`` once per (prompt, response) pair.  For agentic
tasks, the full conversation (including tool calls) is embedded in the
response string; for reasoning tasks, only the final answer matters.

Environment variables used at runtime (set in .env or shell):
  TEACHER_BASE_URL, TEACHER_API_KEY, TEACHER_MODEL  — for optional LLM eval
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from agentic_qwen.training.rewards.agentic_reward import compute_agentic_score
from agentic_qwen.training.rewards.math_reward import compute_math_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified compute_score entry point
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Dispatch to the appropriate reward function based on ``data_type``.

    This is the function referenced in ``training_grpo.yaml`` under
    ``custom_reward_function``.  verl calls it after every rollout.

    Args:
        data_source:  Dataset/source identifier (passed through from Parquet).
        solution_str: Full model response (may include tool-call JSON blocks).
        ground_truth: Expected answer (empty for agentic tasks).
        extra_info:   Row-level metadata dict, expected keys:
                        ``data_type``   – ``"agentic"`` | ``"reasoning"``
                        ``rubrics``     – list[str], for agentic tasks
                        ``use_llm_eval``– bool, default False

    Returns:
        float in [0, 1].
    """
    extra_info = extra_info or {}
    data_type: str = extra_info.get("data_type", "reasoning")

    if data_type == "agentic":
        return compute_agentic_score(data_source, solution_str, ground_truth, extra_info)
    else:
        return compute_math_score(data_source, solution_str, ground_truth, extra_info)


# ---------------------------------------------------------------------------
# Agent-loop helpers (used by the verl multi-turn rollout worker)
# ---------------------------------------------------------------------------

_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END   = "</tool_call>"
_TOOL_RESULT_START = "<tool_response>"
_TOOL_RESULT_END   = "</tool_response>"


def extract_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse ``<tool_call>…</tool_call>`` blocks from model output.

    Returns a list of ``{"name": str, "arguments": dict}`` dicts.
    Unknown / malformed blocks are silently skipped.
    """
    import re
    calls: list[dict[str, Any]] = []
    for match in re.finditer(
        rf"{re.escape(_TOOL_CALL_START)}(.*?){re.escape(_TOOL_CALL_END)}",
        text,
        re.DOTALL,
    ):
        raw = match.group(1).strip()
        try:
            obj = json.loads(raw)
            calls.append({"name": obj.get("name", ""), "arguments": obj.get("arguments", {})})
        except json.JSONDecodeError:
            logger.debug("Skipping malformed tool call: %s", raw[:80])
    return calls


def format_tool_result(name: str, result: dict[str, Any]) -> str:
    """Wrap a tool execution result in the canonical string format.

    The model is trained to read tool results delimited by
    ``<tool_response>`` / ``</tool_response>`` tags.
    """
    return (
        f"{_TOOL_RESULT_START}\n"
        f"Tool: {name}\n"
        f"Result: {json.dumps(result, ensure_ascii=False)}\n"
        f"{_TOOL_RESULT_END}"
    )


def is_terminal(text: str) -> bool:
    """Return *True* if *text* contains a conversation-end marker."""
    return "###STOP" in text or "###TRANSFER_TO_HUMAN" in text
