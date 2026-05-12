"""
Agentic flywheel — mock user simulator.

Plays the role of the human in a multi-turn agent conversation.

Normal mode:
    Responds naturally, provides requested information in a realistic way.

Adversarial mode (paper Phase 4):
    Introduces deliberate friction — ambiguous phrasing, minor factual errors,
    mid-conversation goal changes — to train robustness.
"""

from __future__ import annotations

from agentic_qwen.llm.client import LLMClient

_NORMAL = """\
You are roleplaying as a user talking to an AI assistant.

Your goal: {goal}
Your background: {persona}

Respond naturally to the assistant's messages. Provide information when asked.
You may be vague at first but should ultimately cooperate. Keep replies to 1–3 sentences.
Do NOT reveal your full goal in the opening message.
"""

_ADVERSARIAL = """\
You are roleplaying as a somewhat difficult user talking to an AI assistant.

Your goal: {goal}
Your background: {persona}

You tend to:
- Give incomplete or slightly ambiguous information initially
- Use the wrong term occasionally (e.g. wrong date, wrong name)
- Change a minor detail once during the conversation
- Ask the assistant to "just figure it out" before clarifying

You still want the assistant to succeed. If pressed, provide the correct details.
Keep replies to 1–3 sentences.
"""


class MockUser:
    """LLM-backed user simulator for synthetic trajectory generation.

    Args:
        user_goal:   The task the simulated user wants to accomplish.
        persona:     Background description of the simulated user.
        client:      :class:`~agentic_qwen.llm.client.LLMClient` instance.
        adversarial: When *True*, enable the adversarial interaction mode.
    """

    def __init__(
        self,
        user_goal: str,
        persona: str,
        client: LLMClient,
        *,
        adversarial: bool = False,
    ) -> None:
        self._goal = user_goal
        self._client = client
        template = _ADVERSARIAL if adversarial else _NORMAL
        self._system = template.format(goal=user_goal, persona=persona)
        self._history: list[dict[str, str]] = []

    # ------------------------------------------------------------------

    def initial_message(self) -> str:
        """Return the opening user utterance (the raw goal statement)."""
        return self._goal

    async def respond(self, agent_message: str) -> str:
        """Generate the user's reply to *agent_message*."""
        self._history.append({"role": "assistant", "content": agent_message})
        messages = [{"role": "system", "content": self._system}, *self._history]
        reply = await self._client.chat(messages, temperature=0.8, max_tokens=256)
        self._history.append({"role": "user", "content": reply})
        return reply
