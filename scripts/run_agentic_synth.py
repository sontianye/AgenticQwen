#!/usr/bin/env python3
"""
Agentic data-synthesis flywheel entry point.

Run ``make prepare-data`` and ``make gen-personas`` first.

Examples::

    make synth-agentic                  # full run
    make dry-run                        # smoke test (5 personas)
    python scripts/run_agentic_synth.py --config configs/synthesis_agentic.yaml
    python scripts/run_agentic_synth.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_qwen.data_synthesis.pipeline import run_agentic_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)


async def _dry_run(config_path: str) -> None:
    import json
    import yaml
    from agentic_qwen.data_synthesis.pipeline import (
        _load_personas,
        _process_persona,
        LLMClient,
    )

    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    personas = _load_personas(cfg["personas_file"])[:5]
    cfg["checkpoint_file"] = "/tmp/agentic_dry_run_checkpoint.jsonl"
    output = "/tmp/agentic_dry_run_samples.jsonl"
    client = LLMClient.from_config("teacher")

    for idx, persona in enumerate(personas):
        n = await _process_persona(idx, persona, cfg, client, output, cfg["checkpoint_file"])
        print(f"  persona {idx}: {n} sample(s) written")

    await client.close()
    print(f"\nDry-run complete — samples at {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agentic data-synthesis flywheel.")
    parser.add_argument("--config", default="configs/synthesis_agentic.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Process only 5 personas.")
    args = parser.parse_args()

    if args.dry_run:
        asyncio.run(_dry_run(args.config))
    else:
        asyncio.run(run_agentic_pipeline(args.config))


if __name__ == "__main__":
    main()
