#!/usr/bin/env python3
"""
TAU-2 benchmark evaluation.

TAU-2 tests multi-turn agent behaviour across three domains mirroring
real contact-centre workflows:
  - airline   (flight booking, cancellation, upgrades)
  - retail    (order tracking, returns, product queries)
  - telecom   (plan changes, billing, troubleshooting)

Each scenario provides a task description, a set of tools, and a
simulated user.  The agent must complete the task within a turn limit.
Success is measured by rubric-based scoring (same rubric format as
training data).

Usage::

    # Start SGLang / vLLM server first
    python -m sglang.launch_server --model-path checkpoints/... --port 30000

    uv run python scripts/eval/eval_tau2.py \\
        --model-url http://localhost:30000/v1 \\
        --model-name AgenticQwen-1.7B \\
        --output results/tau2.json

Reference: https://tau-bench.github.io
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

TAU2_DOMAINS = ["airline", "retail", "telecom"]

_AGENT_SYSTEM = """\
You are a helpful AI customer service agent. You have access to tools to assist customers.

Domain: {domain}

When you have resolved the customer's request, end with ###STOP.
If you must escalate to a human agent, end with ###TRANSFER_TO_HUMAN.
"""


# ---------------------------------------------------------------------------
# TAU-2 scenario runner
# ---------------------------------------------------------------------------

async def _run_scenario(
    scenario: dict[str, Any],
    client: Any,
    model: str,
    teacher_client: Any,
    teacher_model: str,
    max_turns: int = 15,
) -> dict[str, Any]:
    """Run one TAU-2 scenario and return the scored result."""
    from agentic_qwen.data_synthesis.agentic.mock_user import MockUser
    from agentic_qwen.llm.client import LLMClient

    domain = scenario.get("domain", "unknown")
    tools = scenario.get("tools", [])
    task = scenario.get("task", "")
    rubrics: list[str] = scenario.get("rubrics", [])
    user_persona = scenario.get("user_persona", "")

    # Wrap teacher in LLMClient for mock user
    teacher_lc = LLMClient(
        base_url=teacher_client.base_url,
        api_key=teacher_client.api_key,
        model=teacher_model,
    )
    user_sim = MockUser(task, user_persona, teacher_lc, adversarial=False)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _AGENT_SYSTEM.format(domain=domain)},
        {"role": "user", "content": user_sim.initial_message()},
    ]
    conversation_log: list[str] = [f"User: {user_sim.initial_message()}"]

    for _ in range(max_turns):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
                temperature=0.0,
                max_tokens=1024,
            )
        except Exception as exc:
            logger.warning("Model call failed: %s", exc)
            break

        msg = resp.choices[0].message
        content = msg.content or ""

        if msg.tool_calls:
            messages.append(msg.model_dump(exclude_none=True))
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}
                # Simple deterministic mock: return empty success for eval
                result = {"success": True, "data": f"Mock result for {tc.function.name}"}
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "name": tc.function.name,
                    "content": json.dumps(result),
                })
                conversation_log.append(f"[Tool:{tc.function.name}] → {result}")
            continue

        messages.append({"role": "assistant", "content": content})
        conversation_log.append(f"Agent: {content}")

        if "###STOP" in content or "###TRANSFER_TO_HUMAN" in content:
            break

        user_reply = await user_sim.respond(content)
        messages.append({"role": "user", "content": user_reply})
        conversation_log.append(f"User: {user_reply}")

    await teacher_lc.close()

    # Score against rubrics using the teacher model
    conversation_str = "\n".join(conversation_log)
    score = await _rubric_score(conversation_str, task, rubrics, teacher_client, teacher_model)

    return {
        "scenario_id": scenario.get("id", ""),
        "domain": domain,
        "score": score,
        "conversation": conversation_log,
    }


async def _rubric_score(
    conversation: str,
    task: str,
    rubrics: list[str],
    client: Any,
    model: str,
) -> float:
    if not rubrics:
        return 1.0 if "###STOP" in conversation else 0.0

    rubric_text = "\n".join(f"{i+1}. {r}" for i, r in enumerate(rubrics))
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Score each rubric criterion: 1=met, 0.5=partial, 0=not met. Return JSON: {\"overall\": <float>}"},
            {"role": "user", "content": f"Task: {task}\n\nRubrics:\n{rubric_text}\n\nConversation:\n{conversation[:3000]}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=128,
    )
    try:
        return float(json.loads(resp.choices[0].message.content).get("overall", 0.0))
    except Exception:
        return 0.0


async def run_tau2_eval(
    model_url: str,
    model_name: str,
    teacher_url: str,
    teacher_model: str,
    output_path: str,
    limit: int | None,
    concurrency: int,
) -> dict[str, Any]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI(base_url=model_url, api_key="EMPTY")
    teacher = AsyncOpenAI(base_url=teacher_url, api_key="EMPTY")

    # Load TAU-2 scenarios
    scenarios = _load_tau2_scenarios(limit)
    logger.info("Loaded %d TAU-2 scenarios", len(scenarios))

    sem = asyncio.Semaphore(concurrency)

    async def _bounded(s: dict[str, Any]) -> dict[str, Any]:
        async with sem:
            return await _run_scenario(s, client, model_name, teacher, teacher_model)

    from tqdm.asyncio import tqdm as atqdm
    results = list(await atqdm.gather(*[_bounded(s) for s in scenarios], desc="TAU-2"))

    # Aggregate
    total = len(results)
    avg_score = sum(r["score"] for r in results) / total if total else 0.0
    by_domain: dict[str, list[float]] = {}
    for r in results:
        by_domain.setdefault(r["domain"], []).append(r["score"])

    summary = {
        "model": model_name,
        "total_scenarios": total,
        "average_score": round(avg_score, 4),
        "by_domain": {
            d: {"average_score": round(sum(scores) / len(scores), 4), "n": len(scores)}
            for d, scores in by_domain.items()
        },
        "results": results,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    logger.info("TAU-2 average score: %.3f  → %s", avg_score, out)
    for d, v in summary["by_domain"].items():
        logger.info("  %-10s  %.3f  (n=%d)", d, v["average_score"], v["n"])

    return summary


def _load_tau2_scenarios(limit: int | None) -> list[dict[str, Any]]:
    """Load TAU-2 scenarios; fall back to synthetic samples if unavailable."""
    try:
        from datasets import load_dataset
        ds = load_dataset("tau-bench/tau-bench", split="test")
        samples = [dict(row) for row in ds]
        if limit:
            samples = samples[:limit]
        return samples
    except Exception as exc:
        logger.warning("Could not load TAU-2 from HuggingFace (%s); using synthetic samples.", exc)
        return _synthetic_tau2_samples(limit or 6)


def _synthetic_tau2_samples(n: int) -> list[dict[str, Any]]:
    domains = ["airline", "retail", "telecom"]
    tasks = [
        ("Book a round-trip flight from SFO to JFK for next Monday, economy class.", [
            "Searched for available flights",
            "Selected appropriate flight",
            "Completed booking with confirmation",
        ]),
        ("I want to return my order #12345 and get a refund.", [
            "Located the order",
            "Initiated return process",
            "Confirmed refund timeline",
        ]),
        ("Upgrade my current mobile plan to unlimited data.", [
            "Retrieved current plan details",
            "Presented available upgrade options",
            "Completed plan change",
        ]),
    ] * 2

    return [
        {
            "id": f"synthetic_{i}",
            "domain": domains[i % 3],
            "task": tasks[i % len(tasks)][0],
            "rubrics": tasks[i % len(tasks)][1],
            "user_persona": f"A typical {domains[i % 3]} customer needing assistance.",
            "tools": [],
        }
        for i in range(n)
    ]


def main() -> None:
    import os
    parser = argparse.ArgumentParser(description="Evaluate on TAU-2 benchmark.")
    parser.add_argument("--model-url", default="http://localhost:30000/v1")
    parser.add_argument("--model-name", default="AgenticQwen")
    parser.add_argument("--teacher-url", default=os.environ.get("TEACHER_BASE_URL", "http://localhost:30000/v1"))
    parser.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", "AgenticQwen"))
    parser.add_argument("--output", default="results/tau2.json")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()

    asyncio.run(run_tau2_eval(
        args.model_url, args.model_name,
        args.teacher_url, args.teacher_model,
        args.output, args.limit, args.concurrency,
    ))


if __name__ == "__main__":
    main()
