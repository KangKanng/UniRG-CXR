#!/usr/bin/env bash
set -euo pipefail

# Run LoRA model inference on the IU-Xray test set (590 samples).
# Usage:
#   bash scripts/infer_lora_test.sh
#   CUDA_VISIBLE_DEVICES=0 bash scripts/infer_lora_test.sh
#   RESULT_PATH=artifacts/my_pred.jsonl bash scripts/infer_lora_test.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
CKPT_DIR="${CKPT_DIR:-${ROOT}/output/iu_qwen3vl8b_lora_sft3/best-checkpoint}"
TEST_DATA="${TEST_DATA:-${ROOT}/artifacts/iu_test.jsonl}"
RESULT_PATH="${RESULT_PATH:-${ROOT}/artifacts/iu_test_lora_pred.jsonl}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/infer_lora_test_${TIMESTAMP}.log"

export TOKENIZERS_PARALLELISM=false

test -f "${TEST_DATA}"
test -f "${CKPT_DIR}/adapter_model.safetensors"
mkdir -p "${LOG_DIR}"
echo "Local log: ${LOG_FILE}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
swift infer \
  --model "${MODEL}" \
  --adapters "${CKPT_DIR}" \
  --val_dataset "${TEST_DATA}" \
  --infer_backend vllm \
  --merge_lora true \
  --torch_dtype bfloat16 \
  --max_new_tokens 512 \
  --temperature 0.0 \
  --max_batch_size 8 \
  --max_length 2048 \
  --max_pixels 262144 \
  --result_path "${RESULT_PATH}" \
  2>&1 | tee "${LOG_FILE}"