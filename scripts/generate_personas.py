#!/usr/bin/env python3
"""
Generate a 5 K persona bank via the teacher model.

Personas cover diverse professions, backgrounds, and geographies.
Output: ``data/personas/personas_5k.jsonl`` (one JSON object per line).

Usage::

    uv run python scripts/generate_personas.py
    uv run python scripts/generate_personas.py --count 1000 --output data/personas/personas_1k.jsonl
    uv run python scripts/generate_personas.py --dry-run   # generates 10 personas
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agentic_qwen.llm.client import LLMClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diverse seed domains to maximise coverage
# ---------------------------------------------------------------------------

_DOMAINS = [
    "healthcare", "legal", "finance", "software engineering", "education",
    "retail", "logistics", "hospitality", "manufacturing", "real estate",
    "media", "government", "non-profit", "agriculture", "energy",
    "telecommunications", "insurance", "consulting", "research", "arts",
]

_SYSTEM = """\
Generate realistic and diverse user personas. Each persona should have a distinct
professional background, role, location, and set of daily challenges.

Return valid JSON with this exact structure:
{{
  "personas": [
    {{
      "persona": "<2-4 sentence description of this person>",
      "domain": "<professional domain>",
      "role": "<job title>",
      "location": "<city, country>"
    }}
  ]
}}

Constraints:
- Personas must be varied (different industries, seniority levels, geographies)
- Each persona should feel like a real individual, not a generic archetype
- Include both technical and non-technical roles
"""

_USER = """\
Focus on the following domains: {domains}

Generate {n} diverse personas.
"""


async def _generate_batch(
    n: int,
    domains: list[str],
    client: LLMClient,
) -> list[dict]:
    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _USER.format(domains=", ".join(domains), n=n)},
    ]
    for attempt in range(3):
        try:
            raw = await client.chat_json(messages, temperature=0.95, max_tokens=4096)
            data = json.loads(raw)
            return data.get("personas", [])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("Persona batch attempt %d failed: %s", attempt + 1, exc)
    return []


async def generate_personas(
    total: int,
    output_path: str,
    batch_size: int = 20,
    concurrency: int = 10,
) -> None:
    """Generate *total* personas and write to *output_path*."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Count already written (for resumption)
    existing = 0
    if out.exists():
        with open(out) as fh:
            existing = sum(1 for line in fh if line.strip())
    remaining = max(0, total - existing)

    if remaining == 0:
        logger.info("Already have %d personas; nothing to do.", existing)
        return

    logger.info("Generating %d more personas (already have %d) …", remaining, existing)

    client = LLMClient.from_config("teacher")
    sem = asyncio.Semaphore(concurrency)

    n_batches = (remaining + batch_size - 1) // batch_size
    import itertools
    domain_cycle = itertools.cycle(_DOMAINS)

    async def _bounded_batch(batch_n: int) -> list[dict]:
        domains = [next(domain_cycle) for _ in range(4)]
        size = min(batch_size, remaining - batch_n * batch_size)
        async with sem:
            return await _generate_batch(size, domains, client)

    from tqdm.asyncio import tqdm as atqdm
    results = await atqdm.gather(
        *[_bounded_batch(i) for i in range(n_batches)],
        desc="Generating personas",
    )

    written = 0
    with open(out, "a") as fh:
        for batch in results:
            for persona in batch:
                fh.write(json.dumps(persona, ensure_ascii=False) + "\n")
                written += 1

    logger.info("Done. Wrote %d new personas → %s (total: %d)", written, out, existing + written)
    await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the persona bank via the teacher model.")
    parser.add_argument("--count", type=int, default=5000, help="Total personas to generate")
    parser.add_argument("--output", default="data/personas/personas_5k.jsonl")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Generate only 10 personas")
    args = parser.parse_args()

    if args.dry_run:
        args.count = 10
        args.output = "/tmp/personas_dry_run.jsonl"

    asyncio.run(generate_personas(args.count, args.output, args.batch_size, args.concurrency))


if __name__ == "__main__":
    main()
