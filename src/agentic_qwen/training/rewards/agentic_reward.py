"""
Reward function for agentic (tool-use) tasks.

Two scoring strategies, selectable via ``extra_info["use_llm_eval"]``:

Rule-based (default, fast):
    Lightweight heuristics based on completion markers, tool-call density,
    and response length.  Suitable for high-throughput RL rollouts.

LLM-based (optional, accurate):
    Calls the teacher model to score against the task rubric.  Use sparingly
    (e.g., periodic evaluation or offline scoring).

Compatible with the verl ``compute_score`` interface.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule-based scorer (fast path)
# ---------------------------------------------------------------------------

def _rule_based_score(solution_str: str, rubrics: list[str]) -> float:
    """Estimate task completion via lightweight heuristics.

    Weights:
      40% — presence of a completion marker (###STOP / ###TRANSFER_TO_HUMAN)
      30% — ratio of tool calls to expected steps
      30% — response is non-trivially long (≥ 50 chars)
    """
    score = 0.0

    if "###STOP" in solution_str or "###TRANSFER_TO_HUMAN" in solution_str:
        score += 0.4

    tool_count = solution_str.count("<tool_call>") + solution_str.count('"function_call"')
    expected = max(1, len(rubrics))
    score += min(1.0, tool_count / expected) * 0.3

    if len(solution_str) >= 50:
        score += 0.3
    elif solution_str:
        score += 0.1

    return round(min(1.0, score), 4)


# ---------------------------------------------------------------------------
# LLM-based scorer (slow path)
# ---------------------------------------------------------------------------

_EVAL_SYSTEM = """\
Score each rubric criterion for the AI agent's conversation:
  1   = fully met
  0.5 = partially met
  0   = not met

Return JSON: {"scores": [<float>, ...], "overall": <float 0–1>}
"""


@retry(stop=stop_after_attempt(2), wait=wait_exponential(min=1, max=10))
async def _llm_score(conversation: str, rubrics: list[str], client: AsyncOpenAI, model: str) -> float:
    rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubrics))
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _EVAL_SYSTEM},
            {"role": "user", "content": f"Rubrics:\n{rubric_text}\n\nConversation:\n{conversation[:3000]}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=256,
    )
    data = json.loads(resp.choices[0].message.content)
    return float(data.get("overall", 0.0))


# ---------------------------------------------------------------------------
# verl interface
# ---------------------------------------------------------------------------

def compute_agentic_score(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,  # noqa: ARG001
    extra_info: dict[str, Any] | None = None,
) -> float:
    """Score an agent trajectory against its task rubric.

    Args:
        data_source:  Dataset identifier (unused).
        solution_str: Full model response (tool calls + text).
        ground_truth: Unused; rubric lives in *extra_info*.
        extra_info:   Dict with optional keys:
                        ``rubrics``      – list of rubric criterion strings
                        ``use_llm_eval`` – bool, default False

    Returns:
        float in [0, 1].
    """
    extra_info = extra_info or {}
    rubrics: list[str] = extra_info.get("rubrics", [])

    if extra_info.get("use_llm_eval", False):
        try:
            client = AsyncOpenAI(
                base_url=os.environ["TEACHER_BASE_URL"],
                api_key=os.environ["TEACHER_API_KEY"],
            )
            model = os.environ["TEACHER_MODEL"]
            loop = asyncio.new_event_loop()
            score = loop.run_until_complete(_llm_score(solution_str, rubrics, client, model))
            loop.close()
            return float(max(0.0, min(1.0, score)))
        except Exception as exc:
            logger.warning("LLM eval failed, falling back to rule-based: %s", exc)

    return _rule_based_score(solution_str, rubrics)
