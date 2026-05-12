#!/usr/bin/env python3
"""
BFCL-V4 (Berkeley Function Calling Leaderboard v4) evaluation.

Evaluates the trained model on BFCL-V4 — a benchmark for function-calling
accuracy covering simple, multiple, parallel, and nested tool-use scenarios.

Usage::

    # Start SGLang server first
    python -m sglang.launch_server --model-path checkpoints/grpo_xxx/actor --port 30000

    # Run evaluation
    uv run python scripts/eval/eval_bfcl.py \\
        --model-url http://localhost:30000/v1 \\
        --model-name Qwen3-1.7B-AgenticQwen \\
        --output results/bfcl_v4.json

Reference: https://gorilla.cs.berkeley.edu/leaderboard.html
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# BFCL task categories
# ---------------------------------------------------------------------------

BFCL_CATEGORIES = [
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
]

_SYSTEM_PROMPT = """\
You are a helpful assistant that calls functions when needed.
Given a user request and a list of available functions, call the appropriate function(s).

Format each function call as:
<tool_call>{"name": "<function_name>", "arguments": {<args>}}</tool_call>

If no function call is needed, respond normally.
"""


async def _evaluate_sample(
    sample: dict[str, Any],
    client: Any,
    model: str,
) -> dict[str, Any]:
    """Evaluate one BFCL sample and return the result dict."""
    tools = sample.get("function", [])
    # Convert BFCL schema to OpenAI tool schema
    openai_tools = [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("parameters", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]

    prompt = sample.get("question", [])
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + prompt

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools if openai_tools else None,
            tool_choice="auto" if openai_tools else None,
            temperature=0.0,
            max_tokens=1024,
        )
        response_text = resp.choices[0].message.content or ""
        tool_calls = resp.choices[0].message.tool_calls or []

        # Simple accuracy: check if called functions match expected
        predicted_fns = {tc.function.name for tc in tool_calls}
        expected = sample.get("ground_truth", {})
        expected_fns: set[str] = set()
        if isinstance(expected, list):
            for e in expected:
                if isinstance(e, dict):
                    expected_fns.add(e.get("name", ""))
        elif isinstance(expected, dict):
            expected_fns.add(expected.get("name", ""))

        correct = predicted_fns == expected_fns

        return {
            "id": sample.get("id", ""),
            "category": sample.get("category", ""),
            "correct": correct,
            "predicted_functions": list(predicted_fns),
            "expected_functions": list(expected_fns),
            "response": response_text,
        }

    except Exception as exc:
        return {
            "id": sample.get("id", ""),
            "category": sample.get("category", ""),
            "correct": False,
            "error": str(exc),
        }


async def run_bfcl_eval(
    model_url: str,
    model_name: str,
    output_path: str,
    limit: int | None,
    concurrency: int,
) -> dict[str, Any]:
    from openai import AsyncOpenAI

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install datasets: uv add datasets")

    logger.info("Loading BFCL-V4 dataset …")
    try:
        ds = load_dataset("gorilla-llm/Berkeley-Function-Calling-Leaderboard", split="train")
        samples = list(ds)
    except Exception as exc:
        logger.error("Could not load BFCL dataset: %s", exc)
        logger.info("Using synthetic BFCL-style samples for demo …")
        samples = _synthetic_bfcl_samples()

    if limit:
        samples = samples[:limit]

    client = AsyncOpenAI(base_url=model_url, api_key="EMPTY")
    sem = asyncio.Semaphore(concurrency)

    async def _bounded(s: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _evaluate_sample(s, client, model_name)

    from tqdm.asyncio import tqdm as atqdm
    results = list(await atqdm.gather(*[_bounded(s) for s in samples], desc="BFCL-V4"))

    # Aggregate metrics
    total = len(results)
    correct = sum(1 for r in results if r.get("correct", False))
    by_category: dict[str, dict[str, int]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        by_category.setdefault(cat, {"total": 0, "correct": 0})
        by_category[cat]["total"] += 1
        if r.get("correct"):
            by_category[cat]["correct"] += 1

    summary = {
        "model": model_name,
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "by_category": {
            cat: {
                "accuracy": round(v["correct"] / v["total"], 4) if v["total"] else 0.0,
                **v,
            }
            for cat, v in by_category.items()
        },
        "results": results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    logger.info("BFCL-V4 accuracy: %.1f%%  (%d/%d)  → %s", correct / total * 100, correct, total, out)
    for cat, v in summary["by_category"].items():
        logger.info("  %-25s  %.1f%%", cat, v["accuracy"] * 100)

    return summary


def _synthetic_bfcl_samples() -> list[dict[str, Any]]:
    """Fallback synthetic samples for offline testing."""
    return [
        {
            "id": "simple_0",
            "category": "simple",
            "question": [{"role": "user", "content": "What is the weather in San Francisco?"}],
            "function": [{"name": "get_weather", "description": "Get weather", "parameters": {
                "type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]
            }}],
            "ground_truth": {"name": "get_weather"},
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on BFCL-V4.")
    parser.add_argument("--model-url", default="http://localhost:30000/v1")
    parser.add_argument("--model-name", default="AgenticQwen")
    parser.add_argument("--output", default="results/bfcl_v4.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=16)
    args = parser.parse_args()

    asyncio.run(run_bfcl_eval(
        args.model_url, args.model_name, args.output, args.limit, args.concurrency
    ))


if __name__ == "__main__":
    main()
