#!/usr/bin/env python3
"""
Reasoning data-synthesis flywheel entry point.

Pipeline:
  1. Load seed datasets (HotpotQA, 2WikiMultiHopQA, …)
  2. Self-instruct expansion (including failure samples from the previous round)
  3. Persona injection (configurable ratio)
  4. Multi-sample consistency filtering
  5. Write filtered samples to JSONL

Examples::

    python scripts/run_reasoning_synth.py
    python scripts/run_reasoning_synth.py --dry-run   # 20 seed samples only
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from tqdm.asyncio import tqdm as atqdm

from agentic_qwen.data_synthesis.reasoning.consistency import filter_batch
from agentic_qwen.data_synthesis.reasoning.expander import expand_batch
from agentic_qwen.data_synthesis.reasoning.persona import PersonaInjector
from agentic_qwen.llm.client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _load_jsonl(path: str, limit: int | None = None) -> list[dict]:
    samples: list[dict] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


async def run(config_path: str, *, dry_run: bool = False) -> None:
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / "samples.jsonl")

    client = LLMClient.from_config("teacher")
    injector = PersonaInjector(cfg["personas_file"])

    # 1 — seed data
    limit = 20 if dry_run else None
    seed: list[dict] = []
    for p in cfg.get("seed_data", []):
        if Path(p).exists():
            seed.extend(_load_jsonl(p, limit=limit))

    failures: list[dict] = []
    failure_path = cfg.get("failure_samples_file", "")
    if failure_path and Path(failure_path).exists():
        failures = _load_jsonl(failure_path, limit=limit)

    all_inputs = seed + failures
    logger.info("Inputs — seed: %d  failures: %d  total: %d", len(seed), len(failures), len(all_inputs))

    if not all_inputs:
        logger.warning("No input samples found.  Check seed_data paths in config.")
        return

    # 2 — expansion
    n_expand = cfg.get("expand_factor", 3)
    logger.info("Expanding %d samples × %d …", len(all_inputs), n_expand)
    expanded = await expand_batch(all_inputs, client, n_per_sample=n_expand)
    logger.info("After expansion: %d", len(expanded))

    # 3 — persona injection
    inject_ratio: float = cfg.get("persona_inject_ratio", 0.5)
    to_inject = [s for s in expanded if random.random() < inject_ratio]
    not_injected = [s for s in expanded if s not in to_inject]

    injected = list(await atqdm.gather(
        *[injector.inject(
            s.get("question") or s.get("input", ""),
            s.get("answer") or s.get("output", ""),
            client,
        ) for s in to_inject],
        desc="Persona injection",
    ))
    all_samples = injected + not_injected
    logger.info("After persona injection: %d", len(all_samples))

    # 4 — consistency filter
    n_consistency = cfg.get("consistency_samples", 3)
    logger.info("Consistency filter (n=%d) …", n_consistency)
    filtered = await filter_batch(all_samples, client, n=n_consistency)
    logger.info("Passed: %d / %d", len(filtered), len(all_samples))

    # 5 — write
    with open(output_path, "w") as fh:
        for s in filtered:
            fh.write(json.dumps(s, ensure_ascii=False) + "\n")
    logger.info("Written to %s", output_path)

    await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the reasoning data-synthesis flywheel.")
    parser.add_argument("--config", default="configs/synthesis_reasoning.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args.config, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
