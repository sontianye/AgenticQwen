"""
Reasoning flywheel — self-instruct problem expander.

Takes failure samples from the previous training round and generates
structurally diverse, harder variants via the teacher model.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from agentic_qwen.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are an expert at creating challenging reasoning problems.

Given a problem that an AI model answered incorrectly, generate {n} harder or
structurally different variants.  The core reasoning type must be preserved, but
the domain, framing, or number of steps may differ.

Return valid JSON:
{{
  "variants": [
    {{
      "question": "<rewritten question>",
      "answer":   "<ground-truth answer>",
      "reasoning_steps": ["<step 1>", "..."],
      "variant_type": "<harder|more_steps|reverse|different_domain|...>"
    }}
  ]
}}
"""

_USER = """\
Original problem (answered incorrectly):
Question: {question}
Answer:   {answer}

Generate {n} variants.
"""


async def expand_sample(
    question: str,
    answer: str,
    client: LLMClient,
    *,
    n: int = 3,
) -> list[dict[str, Any]]:
    """Generate *n* variants from a single failure sample."""
    messages = [
        {"role": "system", "content": _SYSTEM.format(n=n)},
        {"role": "user", "content": _USER.format(question=question, answer=answer, n=n)},
    ]
    for attempt in range(3):
        try:
            raw = await client.chat_json(messages, temperature=0.9, max_tokens=2048)
            return json.loads(raw).get("variants", [])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Expansion attempt %d failed: %s", attempt + 1, exc)
    return []


async def expand_batch(
    samples: list[dict[str, Any]],
    client: LLMClient,
    *,
    n_per_sample: int = 3,
) -> list[dict[str, Any]]:
    """Expand a batch of failure samples concurrently."""
    tasks = [
        expand_sample(
            s.get("question") or s.get("input", ""),
            s.get("answer") or s.get("output", ""),
            client,
            n=n_per_sample,
        )
        for s in samples
    ]
    nested = await asyncio.gather(*tasks)
    return [item for sublist in nested for item in sublist]
