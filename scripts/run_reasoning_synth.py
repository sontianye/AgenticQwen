#!/usr/bin/env python3
"""
Reasoning data-synthesis flywheel entry point.

Pipeline:
  1. Load seed datasets (HotpotQA, 2WikiMultiHopQA, …)
  2. Self-instruct expansion (including failure samples from the previous round)
  3. Persona injection (configurable ratio)
  4. Multi-sample consistency filtering
  5. Write filtered samples to JSONL

Run ``make prepare-data`` and ``make gen-personas`` first.

Examples::

    make synth-reasoning                  # full run
    make dry-run                          # smoke test (20 seed samples)
    python scripts/run_reasoning_synth.py --config configs/synthesis_reasoning.yaml
    python scripts/run_reasoning_synth.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_qwen.data_synthesis.reasoning.pipeline import run_reasoning_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reasoning data-synthesis flywheel.")
    parser.add_argument("--config", default="configs/synthesis_reasoning.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Process only 20 seed samples.")
    args = parser.parse_args()

    asyncio.run(run_reasoning_pipeline(args.config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
