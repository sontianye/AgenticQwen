"""
Reasoning flywheel — end-to-end data synthesis pipeline.

Mirrors the structure of ``agentic/pipeline.py`` but drives the three-stage
reasoning flywheel:

  1. Load seed samples (HotpotQA / 2WikiMultiHopQA) + previous failure samples
  2. Self-instruct expansion (``expander``)
  3. Persona injection (``persona``)
  4. Multi-sample consistency filtering (``consistency``)
  5. Write accepted samples to JSONL

All stages are async and concurrency-limited.  A checkpoint file records
successfully processed batch indices so a restart resumes without reprocessing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import yaml
from tqdm.asyncio import tqdm as atqdm

from agentic_qwen.llm.client import LLMClient
from agentic_qwen.data_synthesis.reasoning.consistency import filter_batch
from agentic_qwen.data_synthesis.reasoning.expander import expand_batch
from agentic_qwen.data_synthesis.reasoning.persona import PersonaInjector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_jsonl(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path) as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _done_batch_ids(checkpoint_path: str) -> set[int]:
    if not os.path.exists(checkpoint_path):
        return set()
    done: set[int] = set()
    with open(checkpoint_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    done.add(int(json.loads(line)["batch_id"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Batch processor
# ---------------------------------------------------------------------------

async def _process_batch(
    batch_id: int,
    batch: list[dict[str, Any]],
    cfg: dict[str, Any],
    client: LLMClient,
    injector: PersonaInjector,
    output_path: str,
    checkpoint_path: str,
) -> int:
    """Expand + inject + filter one batch of samples; return number written."""
    n_expand: int = cfg.get("expand_factor", 3)
    inject_ratio: float = cfg.get("persona_inject_ratio", 0.5)
    n_consistency: int = cfg.get("consistency_samples", 3)
    written = 0

    try:
        # Stage 1 — expand
        expanded = await expand_batch(batch, client, n_per_sample=n_expand)

        # Stage 2 — persona injection (probabilistic)
        to_inject = [s for s in expanded if random.random() < inject_ratio]
        not_injected = [s for s in expanded if s not in to_inject]

        inject_coros = [
            injector.inject(
                s.get("question") or s.get("input", ""),
                s.get("answer") or s.get("output", ""),
                client,
            )
            for s in to_inject
        ]
        injected = list(await asyncio.gather(*inject_coros))
        all_samples = injected + not_injected

        # Stage 3 — consistency filter
        filtered = await filter_batch(all_samples, client, n=n_consistency)

        for s in filtered:
            _append_jsonl(output_path, {**s, "_batch_id": batch_id})
            written += 1

        _append_jsonl(checkpoint_path, {"batch_id": batch_id, "n_samples": written})

    except Exception as exc:
        logger.error("Batch %d failed: %s", batch_id, exc, exc_info=True)
        _append_jsonl(checkpoint_path, {"batch_id": batch_id, "n_samples": 0, "error": str(exc)})

    return written


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_reasoning_pipeline(
    config_path: str = "configs/synthesis_reasoning.yaml",
    *,
    dry_run: bool = False,
) -> None:
    """Run the full reasoning data-synthesis flywheel from *config_path*."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / "samples.jsonl")
    checkpoint_path = cfg.get("checkpoint_file", str(out_dir / ".checkpoint.jsonl"))

    # Collect all input samples
    limit = 20 if dry_run else None
    seed: list[dict[str, Any]] = []
    for p in cfg.get("seed_data", []):
        if Path(p).exists():
            seed.extend(_load_jsonl(p, limit=limit))
        else:
            logger.warning("Seed file not found: %s", p)

    failures: list[dict[str, Any]] = []
    failure_path = cfg.get("failure_samples_file", "")
    if failure_path and Path(failure_path).exists():
        failures = _load_jsonl(failure_path, limit=limit)

    all_inputs = seed + failures
    if not all_inputs:
        logger.error(
            "No input samples found.  "
            "Download seed data first:\n  uv run python scripts/prepare_seed_data.py"
        )
        return

    logger.info("Inputs — seed: %d  failures: %d  total: %d", len(seed), len(failures), len(all_inputs))

    # Chunk into batches for checkpoint granularity
    batch_size: int = cfg.get("batch_size", 50)
    batches = [all_inputs[i : i + batch_size] for i in range(0, len(all_inputs), batch_size)]
    done = _done_batch_ids(checkpoint_path)
    remaining = [(bid, b) for bid, b in enumerate(batches) if bid not in done]

    logger.info("Batches total=%d  done=%d  remaining=%d", len(batches), len(done), len(remaining))

    client = LLMClient.from_config("teacher")
    injector = PersonaInjector(cfg["personas_file"])
    sem = asyncio.Semaphore(cfg.get("concurrency", 20))

    async def _bounded(bid: int, batch: list[dict[str, Any]]) -> int:
        async with sem:
            return await _process_batch(bid, batch, cfg, client, injector, output_path, checkpoint_path)

    results = await atqdm.gather(
        *[_bounded(bid, batch) for bid, batch in remaining],
        desc="Reasoning flywheel",
    )
    logger.info("Done. Total samples written: %d", sum(r for r in results if isinstance(r, int)))
    await client.close()
