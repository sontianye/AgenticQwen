#!/usr/bin/env python3
"""
Unit tests for the agentic flywheel solver.
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from agentic_qwen.data_synthesis.agentic.task_gen import Task
from agentic_qwen.data_synthesis.agentic.tool_gen import Tool, ToolSet


def _make_tool_set() -> ToolSet:
    tool = Tool({
        "name": "search_flights",
        "description": "Search for available flights.",
        "parameters": {
            "type": "object",
            "properties": {"origin": {"type": "string"}, "destination": {"type": "string"}},
            "required": ["origin", "destination"],
        },
        "mock_behavior": "Returns a list of flights.",
    })
    return ToolSet(domain="airline", tools=[tool], persona="Business traveller")


def _make_task() -> Task:
    return Task(
        persona="Business traveller",
        user_goal="Book a flight from SFO to JFK.",
        expected_steps=["search_flights(origin=SFO, destination=JFK)", "book_flight(flight_id=...)"],
        rubrics=["Searched for flights", "Completed booking"],
        tool_set_domain="airline",
    )


@pytest.mark.asyncio
async def test_solver_completes_on_stop_marker():
    """Solver should mark trajectory as completed when model emits ###STOP."""
    from agentic_qwen.data_synthesis.agentic.solver import solve_task

    task = _make_task()
    tool_set = _make_tool_set()
    client = MagicMock()

    # Mock: first call returns a tool call, second returns ###STOP
    tool_call_msg = MagicMock()
    tool_call_msg.tool_calls = [MagicMock()]
    tool_call_msg.tool_calls[0].id = "tc_1"
    tool_call_msg.tool_calls[0].function.name = "search_flights"
    tool_call_msg.tool_calls[0].function.arguments = json.dumps({"origin": "SFO", "destination": "JFK"})
    tool_call_msg.content = None
    tool_call_msg.model_dump = MagicMock(return_value={"role": "assistant", "tool_calls": []})

    stop_msg = MagicMock()
    stop_msg.tool_calls = None
    stop_msg.content = "I have booked your flight. ###STOP"

    resp1 = MagicMock(); resp1.choices = [MagicMock(message=tool_call_msg)]
    resp2 = MagicMock(); resp2.choices = [MagicMock(message=stop_msg)]

    call_count = 0
    async def _mock_create(**_kw):
        nonlocal call_count
        call_count += 1
        return resp1 if call_count == 1 else resp2

    client._client = MagicMock()
    client._client.chat.completions.create = _mock_create
    client.model = "test-model"

    # Mock mock_tools executor and mock_user
    with (
        patch("agentic_qwen.data_synthesis.agentic.solver.MockToolExecutor") as MockExec,
        patch("agentic_qwen.data_synthesis.agentic.solver.MockUser") as MockUserCls,
    ):
        exec_inst = MagicMock()
        exec_inst.to_openai_tools.return_value = []
        exec_inst.call = AsyncMock(return_value={"success": True, "flights": []})
        MockExec.return_value = exec_inst

        user_inst = MagicMock()
        user_inst.initial_message.return_value = "Book a flight SFO→JFK"
        user_inst.respond = AsyncMock(return_value = "Yes please.")
        MockUserCls.return_value = user_inst

        traj = await solve_task(task, tool_set, client)

    assert traj.completed is True
    assert traj.error == ""
    assert len(traj.turns) >= 2


@pytest.mark.asyncio
async def test_solver_trajectory_to_dict_schema():
    """Trajectory.to_dict() should include all required keys."""
    from agentic_qwen.data_synthesis.agentic.solver import Trajectory, Turn

    task = _make_task()
    traj = Trajectory(
        task=task,
        turns=[Turn(role="user", content="hello"), Turn(role="assistant", content="###STOP")],
        completed=True,
    )
    d = traj.to_dict()

    assert "task" in d
    assert "turns" in d
    assert "completed" in d
    assert d["completed"] is True
    assert len(d["turns"]) == 2


def test_behavior_tree_leaf_collection():
    """BehaviorTree.leaf_tasks() should collect all leaf-node tasks."""
    from agentic_qwen.data_synthesis.agentic.task_gen import BehaviorTree, Task

    root = BehaviorTree()
    root.children = [
        BehaviorTree(condition="flight available", task=Task(user_goal="Book flight")),
        BehaviorTree(condition="flight full", task=Task(user_goal="Join waitlist")),
    ]

    leaves = root.leaf_tasks()
    assert len(leaves) == 2
    assert {t.user_goal for t in leaves} == {"Book flight", "Join waitlist"}
