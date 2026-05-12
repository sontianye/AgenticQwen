#!/usr/bin/env python3
"""
Download and normalise seed reasoning datasets.

Downloads:
  - HotpotQA  (distractor setting, train split)
  - 2WikiMultiHopQA (train split)

Normalises each sample to the unified schema::

    {"question": str, "answer": str, "supporting_facts": [...], "source": str}

Output files:
  data/seed_tasks/hotpotqa_train.jsonl
  data/seed_tasks/2wikimultihopqa_train.jsonl

Usage::

    uv run python scripts/prepare_seed_data.py
    uv run python scripts/prepare_seed_data.py --limit 5000   # subset for quick testing
"""

from __future__ import annotations

import argparse
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

OUT_DIR = Path("data/seed_tasks")


# ---------------------------------------------------------------------------
# HotpotQA
# ---------------------------------------------------------------------------

def _load_hotpotqa(limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install 'datasets': uv add datasets")

    logger.info("Downloading HotpotQA (distractor) …")
    ds = load_dataset("hotpot_qa", "distractor", split="train", trust_remote_code=True)

    samples: list[dict[str, Any]] = []
    for row in ds:
        if limit and len(samples) >= limit:
            break
        samples.append({
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": row.get("supporting_facts", {}),
            "source": "hotpotqa",
        })

    return samples


# ---------------------------------------------------------------------------
# 2WikiMultiHopQA
# ---------------------------------------------------------------------------

def _load_2wiki(limit: int | None) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install 'datasets': uv add datasets")

    logger.info("Downloading 2WikiMultiHopQA …")
    ds = load_dataset("xanhho/2WikiMultihopQA", split="train", trust_remote_code=True)

    samples: list[dict[str, Any]] = []
    for row in ds:
        if limit and len(samples) >= limit:
            break
        samples.append({
            "question": row["question"],
            "answer": row["answer"],
            "supporting_facts": row.get("supporting_facts", []),
            "source": "2wikimultihopqa",
        })

    return samples


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def _write_jsonl(samples: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("Written %d samples → %s", len(samples), path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download and normalise seed datasets.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max samples per dataset (default: full split)",
    )
    parser.add_argument("--skip-hotpotqa", action="store_true")
    parser.add_argument("--skip-2wiki", action="store_true")
    args = parser.parse_args()

    if not args.skip_hotpotqa:
        try:
            samples = _load_hotpotqa(args.limit)
            _write_jsonl(samples, OUT_DIR / "hotpotqa_train.jsonl")
        except Exception as exc:
            logger.error("HotpotQA download failed: %s", exc)

    if not args.skip_2wiki:
        try:
            samples = _load_2wiki(args.limit)
            _write_jsonl(samples, OUT_DIR / "2wikimultihopqa_train.jsonl")
        except Exception as exc:
            logger.error("2WikiMultiHopQA download failed: %s", exc)

    logger.info("Seed data preparation complete.")


if __name__ == "__main__":
    main()
