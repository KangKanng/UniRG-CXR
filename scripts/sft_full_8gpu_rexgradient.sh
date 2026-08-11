#!/usr/bin/env bash
set -euo pipefail

# Eight-GPU full-parameter SFT on ReXGradient-160K with DeepSpeed ZeRO-3.
# Usage: WANDB_API_KEY=... bash scripts/sft_full_8gpu_rexgradient.sh
#
# Produces the ms-swift training file first:
#   python main.py prepare-rexgradient --output-dir artifacts
# (override --rexgradient-dir if the dataset lives elsewhere).

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
TRAIN_DATA="${TRAIN_DATA:-${ROOT}/artifacts/rexgradient_train.jsonl}"
VAL_DATA="${VAL_DATA:-${ROOT}/artifacts/rexgradient_valid.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/output/rexgradient_qwen3vl8b_full_sft3}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
WANDB_PROJECT="${WANDB_PROJECT:-unirg-cxr-rexgradient-sft}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3vl8b-rexgradient-full-sft3-bs256-lr5e-5}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/sft_full_rexgradient_${TIMESTAMP}.log"

export WANDB_PROJECT
export WANDB_RUN_NAME
export TOKENIZERS_PARALLELISM=false

test -f "${TRAIN_DATA}"
test -f "${VAL_DATA}"
mkdir -p "${LOG_DIR}"
echo "Local log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
NPROC_PER_NODE=8 \
swift sft \
  --model "${MODEL}" \
  --dataset "${TRAIN_DATA}" \
  --val_dataset "${VAL_DATA}" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --num_train_epochs 3 \
  --learning_rate 5e-5 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --max_length 2048 \
  --max_pixels 262144 \
  --gradient_checkpointing true \
  --attn_impl flash_attention_2 \
  --deepspeed zero3 \
  --warmup_ratio 0.03 \
  --weight_decay 0.01 \
  --logging_steps 1 \
  --eval_strategy epoch \
  --save_strategy epoch \
  --save_total_limit 3 \
  --load_best_model_at_end true \
  --metric_for_best_model eval_loss \
  --greater_is_better false \
  --output_dir "${OUTPUT_DIR}" \
  --run_name "${WANDB_RUN_NAME}" \
  --report_to wandb \
  2>&1 | tee "${LOG_FILE}"
