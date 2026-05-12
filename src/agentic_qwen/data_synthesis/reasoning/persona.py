"""
Reasoning flywheel — persona injector.

Embeds reasoning problems in a persona's professional context to increase
surface-form diversity while preserving the underlying reasoning structure
and ground-truth answer.
"""

from __future__ import annotations

import json
import logging
import random
from typing import Any

from agentic_qwen.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = """\
Rewrite the following reasoning problem so that it is set in the professional
context of the given persona.  The core logic and ground-truth answer must
remain identical.

Return valid JSON:
{
  "question": "<rewritten question>",
  "answer":   "<same answer as original>"
}
"""

_USER = """\
Original question: {question}
Original answer:   {answer}

Persona: {persona}

Rewrite the question to fit this persona's context.
"""


class PersonaInjector:
    """Load a persona bank and embed questions inside persona contexts.

    Args:
        personas_file: Path to a JSONL file where each line is either a bare
            string or an object with a ``"persona"`` / ``"text"`` key.
    """

    def __init__(self, personas_file: str) -> None:
        self._pool = self._load(personas_file)
        logger.info("PersonaInjector: loaded %d personas from %s", len(self._pool), personas_file)

    # ------------------------------------------------------------------

    def _load(self, path: str) -> list[str]:
        pool: list[str] = []
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, str):
                    pool.append(obj)
                else:
                    pool.append(obj.get("persona") or obj.get("text") or json.dumps(obj))
        return pool

    def sample(self) -> str:
        """Return a random persona string."""
        return random.choice(self._pool)

    async def inject(
        self,
        question: str,
        answer: str,
        client: LLMClient,
        persona: str | None = None,
    ) -> dict[str, Any]:
        """Rewrite *question* / *answer* inside a persona context.

        Returns a dict with keys ``question``, ``answer``, and ``persona``.
        Falls back to the original on failure.
        """
        persona = persona or self.sample()
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": _USER.format(question=question, answer=answer, persona=persona)},
        ]
        try:
            raw = await client.chat_json(messages, temperature=0.8, max_tokens=512)
            data = json.loads(raw)
            return {"question": data["question"], "answer": data["answer"], "persona": persona}
        except Exception as exc:
            logger.warning("Persona injection failed: %s", exc)
            return {"question": question, "answer": answer, "persona": persona}
