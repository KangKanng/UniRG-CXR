#!/usr/bin/env bash
set -euo pipefail

# Run both ReXGradient experiments sequentially on the same eight GPUs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

bash "${ROOT}/scripts/sft_lora_8gpu_rexgradient.sh"
bash "${ROOT}/scripts/sft_full_8gpu_rexgradient.sh"
