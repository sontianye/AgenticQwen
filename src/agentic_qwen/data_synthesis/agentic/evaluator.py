"""
Agentic flywheel — rubric-based trajectory evaluator.

Decomposes the task rubric into verifiable sub-goals and asks the teacher model
to score each one (0 / 0.5 / 1).  The overall score in [0, 1] serves as the
reward signal during RL training.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentic_qwen.llm.client import LLMClient
from agentic_qwen.data_synthesis.agentic.solver import Trajectory

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an expert evaluator assessing how well an AI agent completed a task.

Score each rubric criterion:
  1   → fully met
  0.5 → partially met
  0   → not met

Return valid JSON:
{
  "scores": [
    {"criterion": "<text>", "score": <0|0.5|1>, "reason": "<brief justification>"}
  ],
  "overall_score": <float 0–1>,
  "summary": "<one-sentence assessment>"
}
"""

_USER = """\
User Goal: {goal}

Rubric:
{rubrics}

Agent Conversation:
{conversation}
"""


def _render_conversation(traj: Trajectory, max_chars: int = 4000) -> str:
    lines: list[str] = []
    for turn in traj.turns:
        if turn.role == "tool":
            lines.append(f"[Tool:{turn.tool_name}] {turn.content[:150]}")
        elif turn.role == "assistant":
            lines.append(f"Agent: {turn.content}")
        else:
            lines.append(f"User:  {turn.content}")
    return "\n".join(lines)[:max_chars]


async def evaluate_trajectory(traj: Trajectory, client: LLMClient) -> dict[str, Any]:
    """Score *traj* against its task rubric.

    Returns a dict with keys ``overall_score`` (float), ``scores`` (list),
    and ``summary`` (str).  Falls back to a completion-based heuristic if the
    rubric is empty or the LLM call fails.
    """
    rubrics = traj.task.rubrics
    if not rubrics:
        return {
            "overall_score": 1.0 if traj.completed else 0.2,
            "scores": [],
            "summary": "No rubric; scored by completion status.",
        }

    rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubrics))
    conversation = _render_conversation(traj)

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(
            goal=traj.task.user_goal,
            rubrics=rubric_text,
            conversation=conversation,
        )},
    ]

    for attempt in range(3):
        try:
            raw = await client.chat_json(messages, temperature=0.2, max_tokens=1024)
            result: dict[str, Any] = json.loads(raw)
            if "overall_score" not in result and "scores" in result:
                raw_scores = [s.get("score", 0) for s in result["scores"]]
                result["overall_score"] = sum(raw_scores) / len(raw_scores) if raw_scores else 0.0
            result["overall_score"] = float(max(0.0, min(1.0, result.get("overall_score", 0.0))))
            return result
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            logger.warning("Evaluator attempt %d failed: %s", attempt + 1, exc)

    return {"overall_score": 0.0, "scores": [], "summary": "Evaluation failed."}
