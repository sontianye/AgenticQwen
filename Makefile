# Convenience targets for the AgenticQwen project.
# Usage:  make <target>

.PHONY: help install test lint format prepare-data gen-personas \
        synth-agentic synth-reasoning make-data train eval clean

# ── Default ──────────────────────────────────────────────────────────────────

help:
	@echo "AgenticQwen — make targets"
	@echo ""
	@echo "  install          Install dependencies with uv"
	@echo "  test             Run all unit tests"
	@echo "  lint             Lint with ruff"
	@echo "  format           Auto-format with ruff"
	@echo ""
	@echo "  prepare-data     Download HotpotQA + 2WikiMultiHopQA seed data"
	@echo "  gen-personas     Generate 5 K persona bank via teacher model"
	@echo ""
	@echo "  synth-agentic    Run the agentic data-synthesis flywheel"
	@echo "  synth-reasoning  Run the reasoning data-synthesis flywheel"
	@echo "  make-data        Merge synthesised data → verl Parquet format"
	@echo ""
	@echo "  train            Launch GRPO training (requires GPU + verl)"
	@echo "  eval             Run BFCL-V4 + TAU-2 evaluations"
	@echo ""
	@echo "  dry-run          Quick end-to-end smoke test (5 personas)"
	@echo "  clean            Remove synthesised data and checkpoints"

# ── Setup ────────────────────────────────────────────────────────────────────

install:
	uv sync --extra dev

# ── Quality ──────────────────────────────────────────────────────────────────

test:
	uv run pytest tests/ -v

lint:
	uv run ruff check src/ tests/ scripts/

format:
	uv run ruff format src/ tests/ scripts/
	uv run ruff check --fix src/ tests/ scripts/

# ── Data ─────────────────────────────────────────────────────────────────────

prepare-data:
	uv run python scripts/prepare_seed_data.py

gen-personas:
	uv run python scripts/generate_personas.py

synth-agentic:
	uv run python scripts/run_agentic_synth.py

synth-reasoning:
	uv run python scripts/run_reasoning_synth.py

make-data:
	uv run python scripts/make_train_data.py

# ── Training ─────────────────────────────────────────────────────────────────

train:
	bash scripts/train_grpo.sh

# ── Evaluation ───────────────────────────────────────────────────────────────

eval:
	uv run python scripts/eval/run_all_evals.py \
		--model-url http://localhost:30000/v1 \
		--output-dir results/

# ── Smoke test ───────────────────────────────────────────────────────────────

dry-run:
	@echo "==> Dry-run: agentic flywheel (5 personas)"
	uv run python scripts/run_agentic_synth.py --dry-run
	@echo "==> Dry-run: reasoning flywheel (20 seed samples)"
	uv run python scripts/run_reasoning_synth.py --dry-run

# ── Cleanup ──────────────────────────────────────────────────────────────────

clean:
	rm -rf data/synthesized/ checkpoints/ results/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
