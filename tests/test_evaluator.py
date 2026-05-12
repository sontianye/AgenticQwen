#!/usr/bin/env python3
"""
Unit tests for the rubric evaluator.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from agentic_qwen.data_synthesis.agentic.solver import Trajectory, Turn
from agentic_qwen.data_synthesis.agentic.task_gen import Task


def _make_traj(completed: bool = True, rubrics: list[str] | None = None) -> Trajectory:
    task = Task(
        user_goal="Book a flight",
        rubrics=["Searched for flights", "Confirmed booking"] if rubrics is None else rubrics,
    )
    traj = Trajectory(
        task=task,
        turns=[
            Turn(role="user", content="I need to book a flight."),
            Turn(role="tool", content='{"flights": []}', tool_name="search_flights"),
            Turn(role="assistant", content="I have booked your flight. ###STOP"),
        ],
        completed=completed,
    )
    return traj


@pytest.mark.asyncio
async def test_evaluator_returns_valid_score():
    from agentic_qwen.data_synthesis.agentic.evaluator import evaluate_trajectory

    client = MagicMock()
    client.chat_json = AsyncMock(return_value=json.dumps({
        "scores": [
            {"criterion": "Searched for flights", "score": 1.0, "reason": "Used search_flights"},
            {"criterion": "Confirmed booking", "score": 0.5, "reason": "Partial"},
        ],
        "overall_score": 0.75,
        "summary": "Good overall.",
    }))

    traj = _make_traj()
    result = await evaluate_trajectory(traj, client)

    assert 0.0 <= result["overall_score"] <= 1.0
    assert result["overall_score"] == 0.75
    assert len(result["scores"]) == 2


@pytest.mark.asyncio
async def test_evaluator_clamps_score_to_unit_interval():
    from agentic_qwen.data_synthesis.agentic.evaluator import evaluate_trajectory

    client = MagicMock()
    client.chat_json = AsyncMock(return_value=json.dumps({
        "overall_score": 1.5,  # out of range
        "scores": [],
        "summary": "Perfect",
    }))

    traj = _make_traj()
    result = await evaluate_trajectory(traj, client)
    assert result["overall_score"] == 1.0


@pytest.mark.asyncio
async def test_evaluator_no_rubrics_uses_completion():
    from agentic_qwen.data_synthesis.agentic.evaluator import evaluate_trajectory

    # client is never called when rubrics is empty — no need to mock chat_json
    client = MagicMock()
    traj_done = _make_traj(completed=True, rubrics=[])
    traj_fail = _make_traj(completed=False, rubrics=[])

    result_done = await evaluate_trajectory(traj_done, client)
    result_fail = await evaluate_trajectory(traj_fail, client)

    assert result_done["overall_score"] == 1.0
    assert result_fail["overall_score"] < 0.5
    client.chat_json.assert_not_called()  # no LLM call needed


@pytest.mark.asyncio
async def test_evaluator_retries_on_bad_json():
    from agentic_qwen.data_synthesis.agentic.evaluator import evaluate_trajectory

    client = MagicMock()
    good = json.dumps({"overall_score": 0.8, "scores": [], "summary": "ok"})
    client.chat_json = AsyncMock(side_effect=["not json", good])

    traj = _make_traj()
    result = await evaluate_trajectory(traj, client)
    assert result["overall_score"] == 0.8
    assert client.chat_json.call_count == 2
