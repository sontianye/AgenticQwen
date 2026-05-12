#!/usr/bin/env python3
"""
Run all evaluations and print a consolidated results table.

Usage::

    uv run python scripts/eval/run_all_evals.py \\
        --model-url http://localhost:30000/v1 \\
        --model-name AgenticQwen-1.7B \\
        --output-dir results/

Runs BFCL-V4 and TAU-2 sequentially and writes:
  results/bfcl_v4.json
  results/tau2.json
  results/summary.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run all evaluations.")
    parser.add_argument("--model-url", default="http://localhost:30000/v1")
    parser.add_argument("--model-name", default="AgenticQwen")
    parser.add_argument("--teacher-url", default=os.environ.get("TEACHER_BASE_URL", ""))
    parser.add_argument("--teacher-model", default=os.environ.get("TEACHER_MODEL", ""))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples per benchmark")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from eval_bfcl import run_bfcl_eval
    from eval_tau2 import run_tau2_eval

    logger.info("=" * 60)
    logger.info("  Model: %s", args.model_name)
    logger.info("=" * 60)

    # BFCL-V4
    logger.info("\n[1/2] BFCL-V4 …")
    bfcl = await run_bfcl_eval(
        args.model_url, args.model_name,
        str(out_dir / "bfcl_v4.json"),
        args.limit, 16,
    )

    # TAU-2
    logger.info("\n[2/2] TAU-2 …")
    teacher_url = args.teacher_url or args.model_url
    teacher_model = args.teacher_model or args.model_name
    tau2 = await run_tau2_eval(
        args.model_url, args.model_name,
        teacher_url, teacher_model,
        str(out_dir / "tau2.json"),
        args.limit, 4,
    )

    # Summary
    summary = {
        "model": args.model_name,
        "bfcl_v4_accuracy": bfcl.get("accuracy", 0.0),
        "tau2_average_score": tau2.get("average_score", 0.0),
        "average": round((bfcl.get("accuracy", 0.0) + tau2.get("average_score", 0.0)) / 2, 4),
    }

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    logger.info("\n%s", "=" * 60)
    logger.info("  Results Summary")
    logger.info("  %-20s  %.1f%%", "BFCL-V4 accuracy", summary["bfcl_v4_accuracy"] * 100)
    logger.info("  %-20s  %.3f", "TAU-2 avg score", summary["tau2_average_score"])
    logger.info("  %-20s  %.3f", "Overall average", summary["average"])
    logger.info("=" * 60)
    logger.info("Full results written to %s/", out_dir)


if __name__ == "__main__":
    asyncio.run(main())
