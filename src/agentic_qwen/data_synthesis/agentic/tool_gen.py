"""
Agentic flywheel — virtual tool-set generation.

Given a persona background, prompts the teacher model to produce a realistic
set of API tools (OpenAI function-calling schema) along with mock-execution
hints that describe how each tool should behave when simulated.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from agentic_qwen.llm.client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SYSTEM = """\
You are an expert API designer. Given a user persona, design a realistic set of tools
(APIs) that an AI agent would use to assist this persona in their daily work.

Respond with valid JSON matching this schema exactly:
{
  "domain": "<concise domain label>",
  "tools": [
    {
      "name": "<snake_case_name>",
      "description": "<one clear sentence>",
      "parameters": {
        "type": "object",
        "properties": {
          "<param>": {"type": "<type>", "description": "<desc>"}
        },
        "required": ["<required_params>"]
      },
      "mock_behavior": "<describe realistic return values for simulation>"
    }
  ]
}

Constraints:
- 4–8 tools per set
- Parameters must be realistic (strings, integers, enums, arrays)
- mock_behavior drives the simulation; be concrete and varied
"""

_USER = "Persona background:\n{persona}\n\nGenerate a tool set for an AI agent assisting this persona."


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Tool(dict):
    """A single tool definition, compatible with the OpenAI function-calling schema."""

    @property
    def name(self) -> str:
        return self["name"]

    @property
    def description(self) -> str:
        return self["description"]

    @property
    def parameters(self) -> dict[str, Any]:
        return self["parameters"]

    @property
    def mock_behavior(self) -> str:
        return self.get("mock_behavior", "")

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the OpenAI tool-use representation."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolSet:
    """A named collection of :class:`Tool` objects for a given domain."""

    def __init__(self, domain: str, tools: list[Tool], persona: str = "") -> None:
        self.domain = domain
        self.tools = tools
        self.persona = persona

    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"domain": self.domain, "persona": self.persona, "tools": [dict(t) for t in self.tools]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolSet":
        return cls(domain=d["domain"], tools=[Tool(t) for t in d["tools"]], persona=d.get("persona", ""))


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

async def generate_tool_set(persona: str, client: LLMClient) -> ToolSet:
    """Generate a :class:`ToolSet` conditioned on *persona*.

    Retries up to three times on malformed JSON before raising.
    """
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(persona=persona)},
    ]
    for attempt in range(3):
        try:
            raw = await client.chat_json(messages, temperature=0.8, max_tokens=2048)
            data = json.loads(raw)
            return ToolSet(domain=data["domain"], tools=[Tool(t) for t in data["tools"]], persona=persona)
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Tool-set generation attempt %d failed: %s", attempt + 1, exc)
            if attempt == 2:
                raise RuntimeError(f"Failed to generate tool set after 3 attempts") from exc
    raise RuntimeError("Unreachable")
