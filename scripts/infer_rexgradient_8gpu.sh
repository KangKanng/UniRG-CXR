#!/usr/bin/env bash
set -euo pipefail

# ReXGradient test inference with the best models, 8x A100 (TP size 8).
#
# Best checkpoints (by eval_loss):
#   full: v0/checkpoint-1094 (0.6296)  — v3 resumed from it and only regressed
#   lora: v0/checkpoint-1641 (0.8029)  — monotonic decrease, last checkpoint best
#
# Usage:
#   bash scripts/infer_rexgradient_8gpu.sh            # full + lora
#   bash scripts/infer_rexgradient_8gpu.sh full       # full only
#   bash scripts/infer_rexgradient_8gpu.sh lora       # lora only
#
# Overridable env: MODEL, FULL_CKPT, LORA_CKPT, TEST_DATA,
#                  RESULT_FULL, RESULT_LORA, LOG_DIR,
#                  CUDA_VISIBLE_DEVICES, VLLM_TP

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
FULL_CKPT="${FULL_CKPT:-${ROOT}/output/rexgradient_qwen3vl8b_full_sft3/best-checkpoint}"
LORA_CKPT="${LORA_CKPT:-${ROOT}/output/rexgradient_qwen3vl8b_lora_sft3/best-checkpoint}"
TEST_DATA="${TEST_DATA:-${ROOT}/artifacts/rexgradient_test.jsonl}"
RESULT_FULL="${RESULT_FULL:-${ROOT}/artifacts/rexgradient_test_full_pred.jsonl}"
RESULT_LORA="${RESULT_LORA:-${ROOT}/artifacts/rexgradient_test_lora_pred.jsonl}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs}"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TP="${VLLM_TP:-8}"

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

test -f "${TEST_DATA}"
test -f "${FULL_CKPT}/config.json"
test -f "${LORA_CKPT}/adapter_model.safetensors"
mkdir -p "${LOG_DIR}"

run_full() {
  local log_file="${LOG_DIR}/infer_rexgradient_full_8gpu_${TIMESTAMP}.log"
  echo "== full (${FULL_CKPT}) TP=${TP} ==  log: ${log_file}"
  swift infer \
    --model "${FULL_CKPT}" \
    --val_dataset "${TEST_DATA}" \
    --infer_backend vllm \
    --torch_dtype bfloat16 \
    --max_new_tokens 512 \
    --temperature 0.0 \
    --max_batch_size 8 \
    --max_length 2048 \
    --max_pixels 262144 \
    --vllm_tensor_parallel_size "${TP}" \
    --result_path "${RESULT_FULL}" \
    2>&1 | tee "${log_file}"
}

run_lora() {
  local log_file="${LOG_DIR}/infer_rexgradient_lora_8gpu_${TIMESTAMP}.log"
  echo "== lora (${LORA_CKPT}) TP=${TP} ==  log: ${log_file}"
  swift infer \
    --model "${MODEL}" \
    --adapters "${LORA_CKPT}" \
    --val_dataset "${TEST_DATA}" \
    --infer_backend vllm \
    --merge_lora true \
    --torch_dtype bfloat16 \
    --max_new_tokens 512 \
    --temperature 0.0 \
    --max_batch_size 8 \
    --max_length 2048 \
    --max_pixels 262144 \
    --vllm_tensor_parallel_size "${TP}" \
    --result_path "${RESULT_LORA}" \
    2>&1 | tee "${log_file}"
}

case "${1:-all}" in
  all)   run_full; run_lora ;;
  full)  run_full ;;
  lora)  run_lora ;;
  *) echo "usage: $0 [all|full|lora]" >&2; exit 2 ;;
esac
