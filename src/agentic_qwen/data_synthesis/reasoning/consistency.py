"""
Reasoning flywheel — multi-sample consistency filter.

Samples the teacher model *n* times for the same question and keeps only those
where all responses agree on the same answer, ensuring that retained samples
have unambiguous ground truth.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import Counter
from typing import Any

from agentic_qwen.llm.client import LLMClient

logger = logging.getLogger(__name__)

_SYSTEM = "Solve the problem step by step. Put your final answer in <answer>…</answer> tags."


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().lower()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1].lower() if lines else ""


async def _solve_once(question: str, client: LLMClient, temperature: float) -> str:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": question},
    ]
    raw = await client.chat(messages, temperature=temperature, max_tokens=1024)
    return _extract(raw)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def check_consistency(
    question: str,
    expected: str,
    client: LLMClient,
    *,
    n: int = 3,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """Sample the model *n* times and report whether all answers agree.

    Returns a dict with:
      ``pass``    – bool, True only when all *n* answers match *expected*
      ``answers`` – list of *n* raw answers
      ``majority``– the most common answer
    """
    answers = list(await asyncio.gather(*[_solve_once(question, client, temperature) for _ in range(n)]))
    majority, count = Counter(answers).most_common(1)[0]
    expected_norm = expected.strip().lower()
    all_same = count == n
    correct = majority == expected_norm or expected_norm in majority or majority in expected_norm
    return {
        "pass": all_same and correct,
        "answers": answers,
        "majority": majority,
        "all_same": all_same,
        "correct": correct,
    }


async def filter_batch(
    samples: list[dict[str, Any]],
    client: LLMClient,
    *,
    n: int = 3,
) -> list[dict[str, Any]]:
    """Keep only samples where the consistency check passes."""

    async def _check(s: dict[str, Any]) -> dict[str, Any] | None:
        result = await check_consistency(
            question=s.get("question") or s.get("input", ""),
            expected=s.get("answer") or s.get("output", ""),
            client=client,
            n=n,
        )
        return {**s, "_consistency": result} if result["pass"] else None

    results = await asyncio.gather(*[_check(s) for s in samples])
    return [r for r in results if r is not None]
