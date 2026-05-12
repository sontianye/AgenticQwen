"""
Agentic flywheel — task solver.

Drives the teacher model as an agent through a simulated environment composed of
:class:`~agentic_qwen.data_synthesis.agentic.mock_tools.MockToolExecutor` and
:class:`~agentic_qwen.data_synthesis.agentic.mock_user.MockUser`.

The result is a full :class:`Trajectory` that records every turn, tool call,
and tool response — ready for rubric evaluation and downstream RL training.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from agentic_qwen.llm.client import LLMClient
from agentic_qwen.data_synthesis.agentic.mock_tools import MockToolExecutor
from agentic_qwen.data_synthesis.agentic.mock_user import MockUser
from agentic_qwen.data_synthesis.agentic.task_gen import Task
from agentic_qwen.data_synthesis.agentic.tool_gen import ToolSet

logger = logging.getLogger(__name__)

_AGENT_SYSTEM = """\
You are a helpful AI agent with access to a set of tools.

Domain: {domain}

Instructions:
1. Understand the user's request.
2. Call tools as needed to gather information or perform actions.
3. Communicate progress and results clearly.
4. When the task is complete, end your response with ###STOP.
5. If you cannot help and must escalate, end with ###TRANSFER_TO_HUMAN.

Think step by step before each action.
"""

MAX_TURNS = 15


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    role: str
    content: str
    tool_name: str = ""
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """Complete record of one agent episode."""

    task: Task
    turns: list[Turn] = field(default_factory=list)
    completed: bool = False
    transferred: bool = False
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.to_dict(),
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "tool_name": t.tool_name,
                    "tool_args": t.tool_args,
                    "tool_result": t.tool_result,
                }
                for t in self.turns
            ],
            "completed": self.completed,
            "transferred": self.transferred,
            "error": self.error,
        }

    def to_messages(self) -> list[dict[str, Any]]:
        """Render the trajectory as an OpenAI messages list (for RL training)."""
        out: list[dict[str, Any]] = []
        for turn in self.turns:
            if turn.role == "tool":
                out.append({"role": "tool", "name": turn.tool_name, "content": json.dumps(turn.tool_result)})
            else:
                out.append({"role": turn.role, "content": turn.content})
        return out


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

async def solve_task(
    task: Task,
    tool_set: ToolSet,
    client: LLMClient,
    *,
    adversarial: bool = False,
) -> Trajectory:
    """Run one agent episode and return the full :class:`Trajectory`.

    The loop follows a standard ReAct pattern:
    ``[user] → [agent] ↔ [tools] → [agent] → … → ###STOP``
    """
    executor = MockToolExecutor(tools=tool_set.tools, client=client)
    user = MockUser(task.user_goal, task.persona, client, adversarial=adversarial)
    traj = Trajectory(task=task)

    init_msg = user.initial_message()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _AGENT_SYSTEM.format(domain=tool_set.domain)},
        {"role": "user", "content": init_msg},
    ]
    traj.turns.append(Turn(role="user", content=init_msg))

    openai_tools = executor.to_openai_tools()

    for _ in range(MAX_TURNS):
        try:
            resp = await client._client.chat.completions.create(
                model=client.model,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
                temperature=0.3,
                max_tokens=1024,
            )
        except Exception as exc:
            traj.error = str(exc)
            logger.error("Agent call failed: %s", exc)
            break

        msg = resp.choices[0].message

        if msg.tool_calls:
            # Append assistant turn with tool_calls intact
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                context = " | ".join(t.content[:80] for t in traj.turns[-3:] if t.content)
                result = await executor.call(tc.function.name, args, context=context)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(result),
                })
                traj.turns.append(Turn(
                    role="tool",
                    content=json.dumps(result),
                    tool_name=tc.function.name,
                    tool_args=args,
                    tool_result=result,
                ))
            continue  # let the agent digest the tool results

        content = msg.content or ""
        messages.append({"role": "assistant", "content": content})
        traj.turns.append(Turn(role="assistant", content=content))

        if "###STOP" in content:
            traj.completed = True
            break
        if "###TRANSFER_TO_HUMAN" in content:
            traj.transferred = True
            break

        # Simulate next user turn
        user_reply = await user.respond(content)
        messages.append({"role": "user", "content": user_reply})
        traj.turns.append(Turn(role="user", content=user_reply))
    else:
        traj.error = "Max turns exceeded"

    return traj
