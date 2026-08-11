#!/usr/bin/env bash
# ============================================================================
# eval-kit 独立运行环境一键搭建脚本
#
# 原则：
#   1. 完全不改动 UniRG-CXR 的训练环境。
#   2. 单独建一个独立 venv（默认 eval-kit/.venv-evalkit），torch / transformers
#      版本默认【锁定为基础环境同版本】，保证 eval-kit 的数值环境与训练环境一致，
#      且后续升级基础环境不会波及 eval-kit。
#   3. 基础环境只用于读取版本号，脚本不会向它写入任何包。
#
# 用法：
#   ./setup_env.sh                  # 建环境并安装（幂等，已存在则跳过安装）
#   ./setup_env.sh --recreate       # 删除旧 venv 后重建
#   ./setup_env.sh --test           # 安装后跑 NLG 一致性测试 + 4 个模型 metric 冒烟
#
# 环境变量覆盖（可选）：
#   EVALKIT_BASE_PYTHON      基础环境 python（默认自动探测 sft/.venv）
#   EVALKIT_PYTHON           用来建 venv 的 python（默认 python3，需与基础环境同大版本）
#   EVALKIT_VENV_DIR         venv 位置（默认 <eval-kit>/.venv-evalkit）
#   EVALKIT_TORCH_VERSION    锁定 torch 版本（默认 = 基础环境 torch.__version__）
#   EVALKIT_TRANSFORMERS_VERSION  锁定 transformers 版本（默认 = 基础环境版本）
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../eval-kit
VENV_DIR="${EVALKIT_VENV_DIR:-$SCRIPT_DIR/.venv-evalkit}"

# --- 1. 定位基础环境 python（只读，绝不写入） --------------------------------
detect_base_python() {
  [[ -n "${EVALKIT_BASE_PYTHON:-}" && -x "${EVALKIT_BASE_PYTHON}" ]] && { echo "$EVALKIT_BASE_PYTHON"; return; }
  [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]] && { echo "$VIRTUAL_ENV/bin/python"; return; }
  command -v python3
}
BASE_PY="$(detect_base_python)"
echo "[1/5] base env python : $BASE_PY"

# --- 2. 读取基础环境版本，作为 eval-kit 的锁定版本 ----------------------------
TORCH_VERSION="${EVALKIT_TORCH_VERSION:-$("$BASE_PY" -c 'import torch; print(torch.__version__)' 2>/dev/null || true)}"
TRANSFORMERS_VERSION="${EVALKIT_TRANSFORMERS_VERSION:-$("$BASE_PY" -c 'import transformers; print(transformers.__version__)' 2>/dev/null || true)}"
echo "      torch         : ${TORCH_VERSION:-<base 无 torch，装默认版>}"
echo "      transformers  : ${TRANSFORMERS_VERSION:-<base 无 transformers>}"

# --- 3. 建 venv（独立环境，与基础环境零共享） --------------------------------
if [[ "${1:-}" == "--recreate" && -d "$VENV_DIR" ]]; then
  echo "[3/5] removing old venv: $VENV_DIR"
  rm -rf "$VENV_DIR"
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  PYTHON_BIN="${EVALKIT_PYTHON:-python3}"
  echo "[3/5] creating venv with $PYTHON_BIN -> $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
PIP="$VENV_DIR/bin/pip"
"$PIP" install -U pip -q || true
echo "      venv python   : $("$PY" --version)"

# --- 4. 安装 eval-kit 及其模型依赖 ------------------------------------------
echo "[4/5] installing torch (pinned to base version) ..."
if [[ -n "$TORCH_VERSION" && "$TORCH_VERSION" == *+cu* ]]; then
  # 形如 2.11.0+cu130 -> 从 PyTorch 官方 cu130 源拉取（该源只含 torch 系轮子）
  CU="${TORCH_VERSION##*+}"
  "$PIP" install -q "torch==$TORCH_VERSION" \
      --index-url "https://download.pytorch.org/whl/$CU" \
      --extra-index-url "https://pypi.org/simple"
else
  if [[ -n "$TORCH_VERSION" ]]; then
    "$PIP" install -q "torch==$TORCH_VERSION"
  else
    "$PIP" install -q torch
  fi
fi

echo "[4/5] installing transformers / sklearn / medspacy ..."
if [[ -n "$TRANSFORMERS_VERSION" ]]; then
  "$PIP" install -q "transformers==$TRANSFORMERS_VERSION"
else
  "$PIP" install -q transformers
fi
"$PIP" install -q numpy scikit-learn
"$PIP" install -q medspacy          # RaTEScore 的 NER 需要（基础环境里没有）

echo "[4/5] installing radgraph (F1 RadGraph 后端) + runtime deps ..."
"$PIP" install -q "git+https://github.com/Stanford-AIMI/radgraph.git"
"$PIP" install -q dotmap appdirs jsonpickle filelock h5py nltk

echo "[4/5] installing RaTEScore 后端 ..."
"$PIP" install -q "git+https://github.com/MAGIC-AI4Med/RaTEScore.git"

echo "[4/5] installing eval-kit 本体 ..."
"$PIP" install -q -e "$SCRIPT_DIR[model]"

# --- 5. 验证 -----------------------------------------------------------------
echo "[5/5] verifying imports ..."
"$PY" - <<'EOF'
import torch, transformers, sklearn, numpy
print(f"  torch {torch.__version__} (cuda_available={torch.cuda.is_available()}) | "
      f"transformers {transformers.__version__} | numpy {numpy.__version__}")
import evalkit, radgraph, RaTEScore, medspacy
print("  evalkit / radgraph / RaTEScore / medspacy: import OK")
EOF

echo
echo "环境就绪：$VENV_DIR"
echo "用法示例："
echo "  $PY -m evalkit -m bleu cider rouge --ref \"heart is normal\" --hypo \"heart is normal\""
echo "  $PY -m evalkit -m chexbert bertscore f1radgraph ratescore --ref-dataset artifacts --ref-split test --hypo-file hypos.txt --mode corpus --limit 10"
echo "  # 模型权重需放在 eval-kit/weights/ 下，或通过 EVALKIT_*_PATH 指定"

if [[ "${1:-}" == "--test" || "${1:-}" == "--recreate" ]]; then
  echo
  echo "== 跑 NLG 一致性测试（对拍 R2GenGPT/evalcap）=="
  (cd "$SCRIPT_DIR" && "$PY" tests/test_evalkit.py)
  if [[ "${1:-}" == "--test" ]]; then
    echo "== 4 个模型 metric 单对冒烟（权重走 eval-kit/weights/ 默认路径）=="
    "$PY" -m evalkit -m chexbert bertscore f1radgraph ratescore \
      --ref "No acute cardiopulmonary process." --hypo "No acute cardiopulmonary abnormality."
  fi
fi
