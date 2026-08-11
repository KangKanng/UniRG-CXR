#!/usr/bin/env bash
set -euo pipefail

# Run both experiments sequentially on the same eight GPUs.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT}/scripts/sft_lora_8gpu.sh"
bash "${ROOT}/scripts/sft_full_8gpu.sh"

