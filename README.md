# AgenticQwen

A clean, production-grade reproduction of
**[AgenticQwen (arXiv 2604.21590)](https://arxiv.org/html/2604.21590v1)** —
training small language models (1.7 B → 8 B → 30 B) for industrial-scale tool use
via **dual data flywheels** and **GRPO reinforcement learning**.

> **Author:** [Tianye Song](https://github.com/sontianye) · songtianye1997@gmail.com

---

## Why this repo?

The original AgenticQwen paper describes a powerful dual-flywheel approach but **does not release code**.
This repository is the **first complete open-source implementation**, with several practical improvements:

- 🔌 **Any OpenAI-compatible API** — Volcano ARK, DeepSeek, DashScope, or your own endpoint; no vendor lock-in
- 💸 **Runs on small models** — validated on Qwen3-1.7B / 4B; no need for 70B+ GPUs
- ⚡ **Fast synthesis** — fully async pipeline with configurable concurrency; thinking-mode disabled by default for 100× speedup on reasoning models
- 🔁 **Resumable** — checkpoint-based; kill and restart at any point without re-doing work
- 🧪 **Testable** — 22 unit tests, zero API key required
- 📦 **Clean & config-driven** — all hyperparameters in YAML, zero magic numbers in source

---

## Key ideas

| Component | What it does |
|-----------|--------------|
| **Reasoning flywheel** | Self-instruct expansion of failure samples → persona injection → multi-sample consistency filter |
| **Agentic flywheel** | Persona → virtual tool set → linear task → behaviour-tree expansion → branch inversion → adversarial user |
| **Trajectory solver** | Teacher model runs a ReAct loop against simulated tools and users |
| **Rubric evaluator** | Decomposes each task into verifiable sub-goals; returns a [0, 1] reward |
| **GRPO training** | verl + SGLang; binary reward for reasoning, rubric reward for agent tasks |

---

## Project layout

```
AgenticQwen/
├── configs/                    # All hyperparameters — zero hard-coding in source
│   ├── llm.yaml                # API endpoints and model names
│   ├── synthesis_agentic.yaml
│   ├── synthesis_reasoning.yaml
│   └── training_grpo.yaml
├── src/agentic_qwen/
│   ├── llm/client.py           # Async, retry-aware OpenAI-compatible client
│   ├── data_synthesis/
│   │   ├── agentic/            # Agentic flywheel modules
│   │   │   ├── tool_gen.py     #   Persona → ToolSet
│   │   │   ├── task_gen.py     #   Linear task → BehaviourTree → Tasks
│   │   │   ├── solver.py       #   ReAct loop → Trajectory
│   │   │   ├── mock_tools.py   #   LLM-simulated tool execution
│   │   │   ├── mock_user.py    #   Normal / adversarial user simulator
│   │   │   └── evaluator.py    #   Rubric scoring
│   │   ├── reasoning/          # Reasoning flywheel modules
│   │   │   ├── expander.py     #   Self-instruct expansion
│   │   │   ├── persona.py      #   Persona injection
│   │   │   └── consistency.py  #   Multi-sample filter
│   │   └── pipeline.py         # End-to-end agentic flywheel orchestrator
│   └── training/rewards/
│       ├── math_reward.py      # Binary reward for reasoning tasks
│       └── agentic_reward.py   # Rubric-based reward for agent tasks
├── scripts/
│   ├── run_agentic_synth.py    # Agentic flywheel entry point
│   ├── run_reasoning_synth.py  # Reasoning flywheel entry point
│   ├── make_train_data.py      # Merge → verl Parquet format
│   └── train_grpo.sh           # Launch verl + SGLang training
├── tests/                      # 22 unit tests, no API key required
└── data/
    ├── personas/               # 5 K persona JSONL
    └── seed_tasks/             # HotpotQA / 2WikiMultiHopQA seed data
```

---

## Quick start

### 1 — Install (Mac dev environment, uv)

```bash
git clone https://github.com/sontianye/AgenticQwen
cd AgenticQwen
uv sync --extra dev
```

### 2 — Configure API keys

```bash
cp env.example .env
# Edit .env: fill in TEACHER_BASE_URL / TEACHER_API_KEY / TEACHER_MODEL
```

Supported providers out of the box (any OpenAI-compatible endpoint works):
- **Volcano ARK** — DeepSeek-V3 / Doubao endpoints
- **DeepSeek official** — `api.deepseek.com`
- **Alibaba DashScope** — Qwen3-235B

### 3 — Run unit tests

```bash
uv run pytest tests/ -v
```

### 4 — Validate the full pipeline (dry-run, ~50 API calls)

```bash
uv run python scripts/run_agentic_synth.py --dry-run
```

---

## Full training workflow

```bash
# 1. Synthesise agentic trajectories
uv run python scripts/run_agentic_synth.py

# 2. Synthesise reasoning problems
uv run python scripts/run_reasoning_synth.py

# 3. Convert to verl Parquet format
uv run python scripts/make_train_data.py

# 4. GRPO training (GPU server)
MODEL=Qwen/Qwen3-1.7B N_GPUS=4 bash scripts/train_grpo.sh

# Scale up progressively
MODEL=Qwen/Qwen3-8B  N_GPUS=8  bash scripts/train_grpo.sh
```

---

## Design principles

| Principle | How it is applied |
|-----------|-------------------|
| **Single responsibility** | Each module does exactly one thing; `pipeline.py` orchestrates |
| **Config-driven** | All hyperparameters in `configs/`; source code has no magic numbers |
| **Async-first** | Data synthesis is fully concurrent; throughput scales with `concurrency` |
| **Resumable** | Every stage appends to a checkpoint JSONL; restarts skip completed items |
| **Testable** | All core modules have isolated unit tests with mocked LLM responses |
| **Progressive** | Start with Qwen3-1.7B to validate, then scale to 4 B → 8 B → 30 B |

---

## Citation

```bibtex
@article{agenticqwen2026,
  title   = {AgenticQwen: Scaling Agentic Capabilities of Small Language Models
             via Dual Data Flywheels},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.21590}
}
```
