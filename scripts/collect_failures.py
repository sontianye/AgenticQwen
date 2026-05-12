#!/usr/bin/env python3
"""
Collect failure samples from a completed RL training run.

After each training round, the model is evaluated on a held-out set.
Samples where the model score < threshold are "failures" and are fed
back into the next flywheel iteration to improve coverage.

This script:
  1. Loads a verl-format Parquet val file
  2. Runs the trained checkpoint on each prompt (via SGLang or local vLLM)
  3. Scores each response with the appropriate reward function
  4. Writes failures to ``data/synthesized/reasoning/failures.jsonl``

Usage::

    uv run python scripts/collect_failures.py \\
        --checkpoint checkpoints/grpo_20260512/actor \\
        --val-file data/synthesized/verl_format/val.parquet \\
        --output data/synthesized/reasoning/failures.jsonl \\
        --threshold 0.5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


async def _score_sample(
    prompt: list[dict[str, str]],
    ground_truth: str,
    extra_info: dict[str, Any],
    client: Any,
    model: str,
    reward_fn: Any,
) -> tuple[str, float]:
    """Generate one response and score it."""
    resp = await client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=0.0,
        max_tokens=2048,
    )
    solution = resp.choices[0].message.content or ""
    score = reward_fn("eval", solution, ground_truth, extra_info)
    return solution, score


async def collect_failures(
    checkpoint: str,
    val_file: str,
    output_path: str,
    threshold: float,
    concurrency: int,
    base_url: str,
) -> None:
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("Install pandas: uv add pandas pyarrow")

    from openai import AsyncOpenAI
    from agentic_qwen.training.rewards.math_reward import compute_math_score
    from agentic_qwen.training.rewards.agentic_reward import compute_agentic_score

    df = pd.read_parquet(val_file)
    logger.info("Loaded %d val samples from %s", len(df), val_file)

    client = AsyncOpenAI(base_url=base_url, api_key="EMPTY")
    # Derive model name from checkpoint dir
    model = str(Path(checkpoint).name)

    sem = asyncio.Semaphore(concurrency)
    failures: list[dict[str, Any]] = []

    async def _process(row: dict[str, Any]) -> None:
        extra = row.get("extra_info", {})
        if isinstance(extra, str):
            extra = json.loads(extra)

        data_type = extra.get("data_type", "reasoning")
        reward_fn = compute_agentic_score if data_type == "agentic" else compute_math_score
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        ground_truth = ""
        rm = row.get("reward_model", {})
        if isinstance(rm, str):
            rm = json.loads(rm)
        ground_truth = rm.get("ground_truth", "")

        async with sem:
            solution, score = await _score_sample(prompt, ground_truth, extra, client, model, reward_fn)

        if score < threshold:
            # Extract the user question for the failure record
            user_content = next(
                (m["content"] for m in reversed(prompt) if m["role"] == "user"), ""
            )
            failures.append({
                "question": user_content,
                "answer": ground_truth,
                "score": score,
                "data_type": data_type,
                "solution": solution,
            })

    from tqdm.asyncio import tqdm as atqdm
    rows = df.to_dict(orient="records")
    await atqdm.gather(*[_process(r) for r in rows], desc="Scoring val set")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for f in failures:
            fh.write(json.dumps(f, ensure_ascii=False) + "\n")

    logger.info(
        "Failures: %d / %d (threshold=%.2f) → %s",
        len(failures), len(rows), threshold, out,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect failure samples for the next flywheel round.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--val-file", default="data/synthesized/verl_format/val.parquet")
    parser.add_argument("--output", default="data/synthesized/reasoning/failures.jsonl")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score below which a sample is a failure")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--base-url", default="http://localhost:30000/v1", help="SGLang / vLLM endpoint")
    args = parser.parse_args()

    asyncio.run(collect_failures(
        args.checkpoint,
        args.val_file,
        args.output,
        args.threshold,
        args.concurrency,
        args.base_url,
    ))


if __name__ == "__main__":
    main()
