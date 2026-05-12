#!/usr/bin/env python3
"""
Merge synthesised data into the verl training format (Parquet).

Reads agentic and reasoning JSONL files, converts each sample to the
``{"prompt": [...], "reward_model": {...}, "extra_info": {...}}`` schema
expected by verl, shuffles, splits into train / val, and writes Parquet.

Usage::

    python scripts/make_train_data.py
    python scripts/make_train_data.py \\
        --agentic-file data/synthesized/agentic/samples.jsonl \\
        --reasoning-file data/synthesized/reasoning/samples.jsonl \\
        --output-dir data/synthesized/verl_format \\
        --val-ratio 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

_AGENT_SYSTEM = (
    "You are a helpful AI agent with access to a set of tools.  "
    "Call tools as needed, then end your response with ###STOP.  "
    "If you must escalate, end with ###TRANSFER_TO_HUMAN."
)

_REASONING_SYSTEM = (
    "You are a careful reasoning assistant.  "
    "Solve problems step by step and put your final answer in <answer>…</answer> tags."
)


# ---------------------------------------------------------------------------
# Converters
# ---------------------------------------------------------------------------

def _agentic_row(s: dict[str, Any]) -> dict[str, Any] | None:
    try:
        task = s["task"]
        return {
            "prompt": [
                {"role": "system", "content": _AGENT_SYSTEM + f"\nDomain: {s.get('domain', '')}"},
                {"role": "user", "content": task["user_goal"]},
            ],
            "reward_model": {"style": "rule", "ground_truth": ""},
            "extra_info": {
                "rubrics": task.get("rubrics", []),
                "data_type": "agentic",
                "task_id": task.get("task_id", ""),
            },
        }
    except (KeyError, TypeError):
        return None


def _reasoning_row(s: dict[str, Any]) -> dict[str, Any] | None:
    question = s.get("question") or s.get("input", "")
    answer = s.get("answer") or s.get("output", "")
    if not question or not answer:
        return None
    return {
        "prompt": [
            {"role": "system", "content": _REASONING_SYSTEM},
            {"role": "user", "content": question},
        ],
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {"data_type": "reasoning"},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agentic-file", default="data/synthesized/agentic/samples.jsonl")
    parser.add_argument("--reasoning-file", default="data/synthesized/reasoning/samples.jsonl")
    parser.add_argument("--output-dir", default="data/synthesized/verl_format")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []

    for path, converter, label in [
        (args.agentic_file, _agentic_row, "agentic"),
        (args.reasoning_file, _reasoning_row, "reasoning"),
    ]:
        if Path(path).exists():
            n_before = len(rows)
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        row = converter(json.loads(line))
                        if row:
                            rows.append(row)
            logger.info("Loaded %d %s samples", len(rows) - n_before, label)

    if not rows:
        logger.error("No samples found.  Run synthesis scripts first.")
        return

    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_ratio))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(out_dir / "train.parquet", index=False)
    pd.DataFrame(val_rows).to_parquet(out_dir / "val.parquet", index=False)

    logger.info("Wrote train=%d  val=%d → %s", len(train_rows), len(val_rows), out_dir)


if __name__ == "__main__":
    main()
