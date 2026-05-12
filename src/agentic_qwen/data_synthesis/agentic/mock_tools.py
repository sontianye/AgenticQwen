"""
Agentic flywheel — mock tool executor.

Uses the teacher model to simulate realistic tool responses based on each
tool's ``mock_behavior`` hint, so no live API keys are needed during
synthetic trajectory collection.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentic_qwen.llm.client import LLMClient
from agentic_qwen.data_synthesis.agentic.tool_gen import Tool

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a tool-execution simulator. Given a tool description and the arguments an AI
agent has provided, return a realistic JSON response.

Rules:
- Match the domain context; be specific (include IDs, dates, prices, statuses …)
- Occasionally return realistic edge cases (no results, policy violations) when
  contextually appropriate
- Keep the response concise but complete
"""


async def _simulate(
    tool: Tool,
    arguments: dict[str, Any],
    client: LLMClient,
    context: str,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {
            "role": "user",
            "content": (
                f"Tool: {tool.name}\n"
                f"Description: {tool.description}\n"
                f"Mock behavior: {tool.mock_behavior}\n\n"
                f"Agent arguments:\n{json.dumps(arguments, indent=2)}\n\n"
                f"Recent context:\n{context}"
            ),
        },
    ]
    try:
        raw = await client.chat_json(messages, temperature=0.6, max_tokens=1024)
        return json.loads(raw)
    except Exception as exc:
        logger.warning("Mock execution failed for %s: %s", tool.name, exc)
        return {"error": str(exc), "success": False}


class MockToolExecutor:
    """Registry-based tool executor that simulates every call via the teacher model.

    Used inside :func:`~agentic_qwen.data_synthesis.agentic.solver.solve_task`.
    """

    def __init__(self, tools: list[Tool], client: LLMClient) -> None:
        self._registry: dict[str, Tool] = {t.name: t for t in tools}
        self._client = client
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------

    def has(self, name: str) -> bool:
        return name in self._registry

    async def call(self, name: str, arguments: dict[str, Any], *, context: str = "") -> dict[str, Any]:
        """Simulate a tool call and append it to the internal history."""
        if name not in self._registry:
            result: dict[str, Any] = {"error": f"Unknown tool: {name!r}", "success": False}
        else:
            result = await _simulate(self._registry[name], arguments, self._client, context)
        self._history.append({"tool_name": name, "arguments": arguments, "result": result})
        return result

    def to_openai_tools(self) -> list[dict[str, Any]]:
        return [t.to_openai_schema() for t in self._registry.values()]

    @property
    def history(self) -> list[dict[str, Any]]:
        return list(self._history)
