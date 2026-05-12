"""
Agentic flywheel — task generation.

Implements the three-phase expansion described in the paper:

Phase 1 — Linear task initialisation
    A simple, single-path workflow where the agent calls a sequence of tools.

Phase 2 — Behaviour-tree expansion
    Branches are added that represent different environment states (e.g. "flight
    available" vs. "flight sold out"), turning one task into a decision tree.

Phase 3 — Branch-to-task inversion
    Every leaf of the behaviour tree becomes an independent training task,
    dramatically expanding dataset size without additional LLM calls.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

from agentic_qwen.llm.client import LLMClient
from agentic_qwen.data_synthesis.agentic.tool_gen import ToolSet

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single agent training task."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    persona: str = ""
    user_goal: str = ""
    expected_steps: list[str] = field(default_factory=list)
    rubrics: list[str] = field(default_factory=list)
    tool_set_domain: str = ""
    branch_condition: str = ""
    adversarial: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "persona": self.persona,
            "user_goal": self.user_goal,
            "expected_steps": self.expected_steps,
            "rubrics": self.rubrics,
            "tool_set_domain": self.tool_set_domain,
            "branch_condition": self.branch_condition,
            "adversarial": self.adversarial,
        }


@dataclass
class BehaviorTree:
    """Recursive behaviour-tree node."""

    condition: str = ""
    task: Task | None = None
    children: list["BehaviorTree"] = field(default_factory=list)

    def is_leaf(self) -> bool:
        return not self.children

    def leaf_tasks(self) -> list[Task]:
        """DFS collection of all leaf-node tasks."""
        if self.is_leaf():
            return [self.task] if self.task else []
        return [t for child in self.children for t in child.leaf_tasks()]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_LINEAR_SYSTEM = """\
You are an expert at designing realistic customer-service scenarios.

Given a user persona and a set of available tools, create one realistic multi-step
user request that requires calling several tools in sequence.

Respond with valid JSON:
{
  "user_goal": "<natural user request>",
  "expected_steps": ["<step 1: call tool_name with ...>", "..."],
  "rubrics": ["<verifiable criterion 1>", "..."]
}

Constraints:
- user_goal must sound natural, not robotic
- 3–6 expected_steps referencing specific tool names
- 3–5 rubric items that are objectively verifiable
"""

_TREE_SYSTEM = """\
You are an expert at designing robust agent evaluation scenarios.

Given a linear task, expand it into a behaviour tree with 2–3 conditional branches.
Each branch represents a meaningfully different environment state.

Respond with valid JSON:
{
  "branches": [
    {
      "condition": "<what triggers this branch>",
      "user_goal": "<adapted request for this condition>",
      "expected_steps": ["<step 1>", "..."],
      "rubrics": ["<criterion 1>", "..."]
    }
  ]
}

Constraints:
- Branches must be independently solvable
- Cover realistic edge cases (e.g. resource not found, policy constraints)
"""


# ---------------------------------------------------------------------------
# Generation functions
# ---------------------------------------------------------------------------

async def generate_linear_task(tool_set: ToolSet, client: LLMClient) -> Task:
    """Phase 1: generate one linear task for *tool_set*."""
    tool_summary = "\n".join(f"- {t.name}: {t.description}" for t in tool_set.tools)
    messages = [
        {"role": "system", "content": _LINEAR_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Persona: {tool_set.persona}\n\n"
                f"Available tools:\n{tool_summary}\n\n"
                "Create a realistic multi-step task."
            ),
        },
    ]
    raw = await client.chat_json(messages, temperature=0.9, max_tokens=1024)
    data = json.loads(raw)
    return Task(
        persona=tool_set.persona,
        user_goal=data["user_goal"],
        expected_steps=data["expected_steps"],
        rubrics=data["rubrics"],
        tool_set_domain=tool_set.domain,
    )


async def expand_to_behavior_tree(
    linear_task: Task,
    tool_set: ToolSet,
    client: LLMClient,
) -> BehaviorTree:
    """Phase 2: expand *linear_task* into a branching behaviour tree."""
    tool_summary = "\n".join(f"- {t.name}: {t.description}" for t in tool_set.tools)
    messages = [
        {"role": "system", "content": _TREE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Original task:\n{json.dumps(linear_task.to_dict(), indent=2)}\n\n"
                f"Available tools:\n{tool_summary}\n\n"
                "Expand into a behaviour tree with conditional branches."
            ),
        },
    ]
    raw = await client.chat_json(messages, temperature=0.85, max_tokens=2048)
    data = json.loads(raw)

    root = BehaviorTree()
    for branch in data["branches"]:
        root.children.append(BehaviorTree(
            condition=branch["condition"],
            task=Task(
                persona=linear_task.persona,
                user_goal=branch["user_goal"],
                expected_steps=branch["expected_steps"],
                rubrics=branch["rubrics"],
                tool_set_domain=tool_set.domain,
                branch_condition=branch["condition"],
            ),
        ))
    return root


def invert_tree_to_tasks(tree: BehaviorTree) -> list[Task]:
    """Phase 3: collect all leaf-node tasks as independent training samples."""
    return tree.leaf_tasks()
