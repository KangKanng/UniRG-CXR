#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

bash "${ROOT}/scripts/infer_rexgradient_full_best.sh"
bash "${ROOT}/scripts/infer_rexgradient_lora_best.sh"
