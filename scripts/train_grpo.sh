#!/bin/bash
# ==============================================================================
# GRPO 训练启动脚本（基于 verl）
#
# 用法:
#   bash scripts/train_grpo.sh                    # 默认 Qwen3-1.7B
#   MODEL=Qwen/Qwen3-4B bash scripts/train_grpo.sh
#
# 前置条件:
#   1. pip install verl sglang
#   2. data/synthesized/verl_format/*.parquet 已生成
#   3. .env 中设置好 TEACHER_BASE_URL / TEACHER_API_KEY / TEACHER_MODEL
# ==============================================================================

set -euo pipefail

# ---------- 可配置参数 ----------
MODEL=${MODEL:-"Qwen/Qwen3-1.7B"}
N_GPUS=${N_GPUS:-4}
TRAIN_BS=${TRAIN_BS:-128}
GRPO_GROUP=${GRPO_GROUP:-8}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-2048}
MAX_RESP_LEN=${MAX_RESP_LEN:-4096}
OUTPUT_DIR=${OUTPUT_DIR:-"checkpoints/grpo_$(date +%Y%m%d_%H%M%S)"}

TRAIN_FILE="data/synthesized/verl_format/agentic_train.parquet"
VAL_FILE="data/synthesized/verl_format/agentic_val.parquet"

echo "========================================"
echo "  AgenticQwen GRPO Training"
echo "  Model:   $MODEL"
echo "  GPUs:    $N_GPUS"
echo "  Output:  $OUTPUT_DIR"
echo "========================================"

mkdir -p "$OUTPUT_DIR"

# ---------- 启动 SGLang 推理服务（后台）----------
echo "[1/3] Starting SGLang inference server..."
python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port 30000 \
  --tp "$N_GPUS" \
  --mem-fraction-static 0.5 \
  &
SGLANG_PID=$!

# 等待服务就绪
echo "Waiting for SGLang server..."
for i in $(seq 1 60); do
  if curl -sf http://localhost:30000/health > /dev/null 2>&1; then
    echo "SGLang server ready."
    break
  fi
  sleep 5
  if [ $i -eq 60 ]; then
    echo "ERROR: SGLang server failed to start."
    kill $SGLANG_PID 2>/dev/null
    exit 1
  fi
done

# ---------- 启动 verl GRPO 训练 ----------
echo "[2/3] Starting verl GRPO training..."

python -m verl.trainer.main_ppo \
  algorithm=grpo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.max_prompt_length=$MAX_PROMPT_LEN \
  data.max_response_length=$MAX_RESP_LEN \
  actor_rollout_ref.model.path="$MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.rollout.n=$GRPO_GROUP \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.sglang.server_addr="http://localhost:30000" \
  critic.optim.lr=1e-5 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.total_epochs=3 \
  trainer.save_freq=100 \
  trainer.project_name=agentic-qwen \
  trainer.experiment_name="grpo_$(basename $MODEL)" \
  trainer.logger=["console","wandb"] \
  trainer.default_hdfs_dir="$OUTPUT_DIR" \
  custom_reward_function.path="src/agentic_qwen/training/rewards/agentic_reward.py" \
  custom_reward_function.name="compute_agentic_score" \
  2>&1 | tee "$OUTPUT_DIR/train.log"

echo "[3/3] Training complete. Stopping SGLang server..."
kill $SGLANG_PID 2>/dev/null || true

echo "Done. Checkpoint saved to: $OUTPUT_DIR"
