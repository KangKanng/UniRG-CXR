#!/usr/bin/env bash
set -euo pipefail

# Run best full-SFT model inference on the ReXGradient test set (10000 samples)
# on 8x A100 (tensor parallel size 8).
#
# Best full model by eval_loss (0.6296): v0/checkpoint-1094.
# See logging.jsonl in the v0 run dir; v3 resumed from this checkpoint and
# only regressed (eval_loss 0.7716 at checkpoint-1641).
#
# Usage:
#   bash scripts/infer_rexgradient_full_best_8gpu.sh
#   RESULT_PATH=artifacts/my_pred.jsonl bash scripts/infer_rexgradient_full_best_8gpu.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${CKPT_DIR:-${ROOT}/output/rexgradient_qwen3vl8b_full_sft3/best-checkpoint}"
TEST_DATA="${TEST_DATA:-${ROOT}/artifacts/rexgradient_test.jsonl}"
RESULT_PATH="${RESULT_PATH:-${ROOT}/artifacts/rexgradient_test_full_pred.jsonl}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/infer_rexgradient_full_8gpu_${TIMESTAMP}.log"

export TOKENIZERS_PARALLELISM=false

test -f "${TEST_DATA}"
test -f "${CKPT_DIR}/config.json"
mkdir -p "${LOG_DIR}"
echo "Local log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" \
swift infer \
  --model "${CKPT_DIR}" \
  --val_dataset "${TEST_DATA}" \
  --infer_backend vllm \
  --torch_dtype bfloat16 \
  --max_new_tokens 512 \
  --temperature 0.0 \
  --max_batch_size 8 \
  --max_length 2048 \
  --max_pixels 262144 \
  --vllm_tensor_parallel_size "${VLLM_TP:-8}" \
  --result_path "${RESULT_PATH}" \
  2>&1 | tee "${LOG_FILE}"
