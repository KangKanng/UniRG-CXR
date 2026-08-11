#!/usr/bin/env bash
set -euo pipefail

# Sequentially evaluate the full and LoRA ReXGradient prediction files.
# Set INCLUDE_CHEXPROMPT=1 to include the remote LLM metric configured in .env.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
PYTHON="${EVALKIT_PYTHON:-python3}"
ARTIFACTS_DIR="${EVALKIT_ARTIFACTS_DIR:-${ROOT}/artifacts}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

METRICS=(bleu cider rouge chexbert bertscore f1radgraph ratescore)
if [[ "${INCLUDE_CHEXPROMPT:-0}" == "1" ]]; then
    METRICS+=(chexprompt)
fi

if [[ ! -x "$PYTHON" ]]; then
    echo "Error: Python executable not found: $PYTHON" >&2
    exit 1
fi

run_eval() {
    local name="$1"
    local input="$ARTIFACTS_DIR/rexgradient_test_${name}_pred.jsonl"
    local output="$ARTIFACTS_DIR/rexgradient_test_${name}_eval.json"
    local log="$ARTIFACTS_DIR/rexgradient_test_${name}_eval.log"
    local parts_dir="$ARTIFACTS_DIR/rexgradient_test_${name}_eval_parts"

    if [[ ! -f "$input" ]]; then
        echo "Error: prediction file not found: $input" >&2
        exit 1
    fi

    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Starting ${name} evaluation"
    echo "Input:  $input"
    echo "Output: $output"
    echo "Log:    $log"

    mkdir -p "$parts_dir"
    : > "$log"
    local metric
    local part
    local -a parts=()
    for metric in "${METRICS[@]}"; do
        part="$parts_dir/${metric}.json"
        parts+=("$part")
        echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Running ${name}: ${metric}" \
            | tee -a "$log"
        CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" PYTHONUNBUFFERED=1 \
            "$PYTHON" -m evalkit \
            -m "$metric" \
            --pred-file "$input" \
            --mode corpus \
            --out "$part" \
            2>&1 | tee -a "$log"
    done

    "$PYTHON" - "$output" "${parts[@]}" <<'PY'
import json
import sys

output, *parts = sys.argv[1:]
result = {}
for part in parts:
    with open(part, encoding="utf-8") as f:
        result.update(json.load(f))
with open(output, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

    echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] Finished ${name} evaluation"
}

cd "$SCRIPT_DIR"
run_eval full
run_eval lora

echo "All ReXGradient evaluations completed successfully."
