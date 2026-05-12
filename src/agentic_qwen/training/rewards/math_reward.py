"""
Reward function for reasoning / mathematics tasks.

Binary scoring:
  1.0 — correct answer, no tool calls used
  0.9 — correct answer but used tool calls (slightly penalised to encourage direct reasoning)
  0.0 — wrong answer or empty response

Compatible with the verl ``compute_score`` interface.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def _answers_match(pred: str, gold: str) -> bool:
    def normalise(s: str) -> str:
        return re.sub(r"[\s,]+", " ", s.lower().strip())

    p, g = normalise(pred), normalise(gold)
    return p == g or g in p or p in g


def _has_tool_calls(text: str) -> bool:
    return "<tool_call>" in text or '"function_call"' in text


def compute_math_score(
    data_source: str,  # noqa: ARG001 — kept for verl interface compatibility
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,  # noqa: ARG001
) -> float:
    """Score a reasoning response against *ground_truth*.

    Args:
        data_source:  Dataset identifier (unused, kept for verl compatibility).
        solution_str: Full model response text.
        ground_truth: Expected answer string.
        extra_info:   Optional metadata dict (unused here).

    Returns:
        ``1.0``, ``0.9``, or ``0.0``.
    """
    if not solution_str:
        return 0.0
    pred = _extract_answer(solution_str)
    if not _answers_match(pred, ground_truth):
        return 0.0
    return 0.9 if _has_tool_calls(solution_str) else 1.0
