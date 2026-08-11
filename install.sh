#!/usr/bin/env bash
set -euo pipefail

uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130

uv pip install "vllm>=0.5.1"

uv pip install "transformers<5.13" "trl<1.0" "peft<0.20" "datasets<4.8.5"

uv pip install optimum bitsandbytes "gradio<5.33" mcore-bridge

uv pip install "ms-swift[all] @ git+https://github.com/modelscope/ms-swift.git@release/4.4"

uv pip install timm "deepspeed<0.19" ray

uv pip install qwen_vl_utils qwen_omni_utils keye_vl_utils

uv pip install decord librosa icecream soundfile

uv pip install liger_kernel nvitop pre-commit math_verify py-spy wandb swanlab

MAX_JOBS=4 uv pip install "flash-attn==2.8.3" --no-build-isolation --no-binary flash-attn

# uv pip install "opencv-python-headless>=4.13.0"

# megatron
# MAX_JOBS=8 uv pip install \
#     pybind11 \
#     "git+https://v4.gh-proxy.org/https://github.com/NVIDIA/TransformerEngine.git@stable" \
#     --no-build-isolation

# MAX_JOBS=8 uv pip install \
#     "git+https://v4.gh-proxy.org/https://github.com/deepseek-ai/DeepGEMM.git@v2.1.1.post3" \
#     --no-build-isolation

# MAX_JOBS=8 uv pip install \
#     flash-linear-attention \
#     -U \
#     --no-build-isolation

# MAX_JOBS=8 uv pip install \
#     "git+https://v4.gh-proxy.org/https://github.com/Dao-AILab/causal-conv1d" \
#     -U \
#     --no-build-isolation

# MAX_JOBS=8 uv pip install \
#     "git+https://v4.gh-proxy.org/https://github.com/Dao-AILab/fast-hadamard-transform" \
#     --no-build-isolation

# uv pip install \
#     "git+https://v4.gh-proxy.org/https://github.com/NVIDIA-NeMo/Emerging-Optimizers.git@v0.3.0"
