"""
Agentic flywheel — end-to-end data synthesis pipeline.

Orchestrates the full four-phase flywheel:
  1. Persona → ToolSet
  2. ToolSet → linear Task
  3. Linear Task → BehaviorTree → branch Tasks
  4. (Optional) Adversarial user injection
  5. Task → Trajectory (via solver)
  6. Trajectory → rubric score (via evaluator)
  7. Score ≥ threshold → write sample; else discard

Supports checkpoint-based resumption: already-processed persona indices are
skipped on restart without re-doing any LLM calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

import yaml
from tqdm.asyncio import tqdm as atqdm

from agentic_qwen.llm.client import LLMClient
from agentic_qwen.data_synthesis.agentic.evaluator import evaluate_trajectory
from agentic_qwen.data_synthesis.agentic.solver import solve_task
from agentic_qwen.data_synthesis.agentic.task_gen import (
    Task,
    expand_to_behavior_tree,
    generate_linear_task,
    invert_tree_to_tasks,
)
from agentic_qwen.data_synthesis.agentic.tool_gen import generate_tool_set

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_personas(path: str) -> list[str]:
    """Load persona strings from a JSONL file.

    Accepts ``{"persona": "..."}`` , ``{"text": "..."}`` , or bare strings.
    """
    personas: list[str] = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, str):
                personas.append(obj)
            else:
                personas.append(obj.get("persona") or obj.get("text") or json.dumps(obj))
    return personas


def _done_indices(checkpoint_path: str) -> set[int]:
    if not os.path.exists(checkpoint_path):
        return set()
    done: set[int] = set()
    with open(checkpoint_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    done.add(int(json.loads(line)["persona_idx"]))
                except (json.JSONDecodeError, KeyError):
                    pass
    return done


def _append_jsonl(path: str, record: dict[str, Any]) -> None:
    with open(path, "a") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Per-persona processor
# ---------------------------------------------------------------------------

async def _process_persona(
    idx: int,
    persona: str,
    cfg: dict[str, Any],
    client: LLMClient,
    output_path: str,
    checkpoint_path: str,
) -> int:
    """Run the full flywheel for one persona; return number of samples written."""
    threshold: float = cfg.get("rubric_threshold", 0.6)
    phases: dict[str, bool] = cfg.get("phases", {})
    n_tasks: int = cfg.get("tasks_per_persona", 3)
    written = 0

    try:
        tool_set = await generate_tool_set(persona, client)
        tasks: list[Task] = []

        for _ in range(n_tasks):
            linear = await generate_linear_task(tool_set, client)
            if phases.get("tree_expansion", True):
                tree = await expand_to_behavior_tree(linear, tool_set, client)
                branch_tasks = invert_tree_to_tasks(tree) if phases.get("branch_inversion", True) else [linear]
                tasks.extend(branch_tasks)
            else:
                tasks.append(linear)

        for task in tasks:
            use_adversarial = phases.get("adversarial_user", False) and task.adversarial
            traj = await solve_task(task, tool_set, client, adversarial=use_adversarial)
            eval_result = await evaluate_trajectory(traj, client)
            score: float = eval_result.get("overall_score", 0.0)

            if score < threshold:
                logger.debug("Persona %d: sample filtered (score=%.2f)", idx, score)
                continue

            _append_jsonl(output_path, {
                "persona_idx": idx,
                "domain": tool_set.domain,
                "task": task.to_dict(),
                "trajectory": traj.to_dict(),
                "eval": eval_result,
                "reward": score,
            })
            written += 1

        _append_jsonl(checkpoint_path, {"persona_idx": idx, "n_samples": written})

    except Exception as exc:
        logger.error("Persona %d failed: %s", idx, exc, exc_info=True)
        _append_jsonl(checkpoint_path, {"persona_idx": idx, "n_samples": 0, "error": str(exc)})

    return written


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_agentic_pipeline(config_path: str = "configs/synthesis_agentic.yaml") -> None:
    """Run the full agentic data-synthesis flywheel from *config_path*."""
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)

    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / "samples.jsonl")
    checkpoint_path = cfg.get("checkpoint_file", str(out_dir / ".checkpoint.jsonl"))

    personas = _load_personas(cfg["personas_file"])
    done = _done_indices(checkpoint_path)
    remaining = [(i, p) for i, p in enumerate(personas) if i not in done]

    logger.info(
        "Personas total=%d  done=%d  remaining=%d",
        len(personas), len(done), len(remaining),
    )

    client = LLMClient.from_config("teacher")
    sem = asyncio.Semaphore(cfg.get("concurrency", 10))

    async def _bounded(idx: int, persona: str) -> int:
        async with sem:
            return await _process_persona(idx, persona, cfg, client, output_path, checkpoint_path)

    results = await atqdm.gather(
        *[_bounded(i, p) for i, p in remaining],
        desc="Agentic flywheel",
    )
    logger.info("Done. Total samples written: %d", sum(r for r in results if isinstance(r, int)))
    await client.close()
