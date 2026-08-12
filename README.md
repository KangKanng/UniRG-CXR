# UniRG-CXR

An unofficial implementation of *UniRG-CXR: Scaling Medical Imaging Report
Generation with Multimodal Reinforcement Learning* (arXiv:2601.17151).

We are currently conducting SFT experiments and reproducing the RL pipeline on public datasets. Results and implementations will be released progressively.

This repository contains the reproducible code used to:

- prepare IU-Xray and ReXGradient-160K as ms-swift JSONL datasets;
- fine-tune Qwen3-VL-8B-Instruct with full-parameter or LoRA SFT;
- run single- or eight-GPU inference; and
- evaluate reports with BLEU, CIDEr, ROUGE-L, CheXbert, BERTScore, RadGraph,
  RaTEScore, and CheXprompt.

It is an independent reproduction, not the official implementation. Model
weights, datasets, checkpoints, generated predictions, logs, and API keys are
not distributed.

## Repository layout

```text
main.py             Dataset preparation CLI
scripts/            SFT and inference launchers
data/rexrank/       IU-Xray R2Gen/ReXrank split manifests
eval-kit/           Standalone report-evaluation package
test_main.py        Data-pipeline tests
```

## Installation

Python 3.12 and CUDA-capable hardware are recommended for training.

```bash
git clone https://github.com/KangKanng/UniRG-CXR.git
cd UniRG-CXR

# Create and activate your own virtual environment first, then:
bash install.sh
```

The training stack is intentionally separate from `eval-kit`. For lightweight
NLG evaluation:

```bash
python -m venv eval-kit/.venv
eval-kit/.venv/bin/pip install -e ./eval-kit
eval-kit/.venv/bin/python -m evalkit -m bleu cider rouge \
  --ref "heart is normal" --hypo "heart is normal"
```

Model-based metrics additionally require their upstream packages and model
weights. See [`eval-kit/README.md`](eval-kit/README.md) and configure weight
paths with the documented `EVALKIT_*` environment variables. Weights are not
downloaded or committed automatically.

## Prepare data

Download IU-Xray separately and place it anywhere on disk. Its directory must
contain `indiana_projections.csv`, `indiana_reports.csv`, and
`images/images_normalized/`.

```bash
python main.py prepare \
  --iu-dir /path/to/iu-xray \
  --split-dir data/rexrank \
  --output-dir artifacts
```

For ReXGradient-160K, first run that dataset's preprocessing so that
`processed/{train,valid,test}.jsonl` exists, then run:

```bash
python main.py prepare-rexgradient \
  --rexgradient-dir /path/to/ReXGradient-160K \
  --output-dir artifacts
```

## Train and infer

Launchers accept environment-variable overrides. `MODEL` can be a Hugging Face
model ID or a local model directory.

```bash
# LoRA or full SFT on 8 GPUs
MODEL=Qwen/Qwen3-VL-8B-Instruct bash scripts/sft_lora_8gpu.sh
MODEL=Qwen/Qwen3-VL-8B-Instruct bash scripts/sft_full_8gpu.sh

# ReXGradient equivalents
bash scripts/sft_lora_8gpu_rexgradient.sh
bash scripts/sft_full_8gpu_rexgradient.sh
```

Inference scripts default to `output/.../best-checkpoint`; override `CKPT_DIR`,
`FULL_CKPT`, or `LORA_CKPT` when your checkpoint is elsewhere. W&B credentials
must be supplied through your environment or `wandb login`; never store them in
scripts.

## Evaluation results

We evaluated the full-parameter and LoRA SFT models on all 590 examples in the
IU-Xray R2Gen test split. The LoRA results below use the second 590-example
inference run (`lora_b` in the evaluation artifacts), reported here as
`lora`.

| Metric | full | lora |
|---|---:|---:|
| BLEU-1 | 0.1829 | 0.1943 |
| BLEU-2 | 0.1171 | 0.1264 |
| BLEU-3 | 0.0807 | 0.0869 |
| BLEU-4 | 0.0548 | 0.0582 |
| CIDEr | 0.2952 | 0.1718 |
| ROUGE-L | 0.2848 | 0.2736 |
| CheXbert micro-F1 (5 classes) | 0.0000 | 0.2034 |
| CheXbert accuracy | 0.8017 | 0.6881 |
| SembScore | 0.5156 | 0.4670 |
| CheXbert micro-F1 (14 classes) | 0.5233 | 0.4764 |
| BERTScore precision | 0.7929 | 0.7722 |
| BERTScore recall | 0.6885 | 0.6788 |
| BERTScore F1 | 0.7359 | 0.7201 |
| F1 RadGraph | 0.2456 | 0.2137 |
| RaTEScore | 0.5762 | 0.5581 |

The LoRA prediction file contained two concatenated inference runs. The run
used here includes 43 reports without an `Impression` section, 37 reports with
a non-period ending, and one unusually long response; its scores should
therefore be interpreted with this generation-quality caveat.

We also evaluated both best checkpoints on the full ReXGradient-160K test
split (10,000 examples, 8× A100 vLLM inference, temperature 0). Full SFT
outperformed LoRA on all 15 metrics; the largest gaps are in clinical/semantic
metrics (CheXbert micro-F1, RadGraph-F1, CIDEr) rather than lexical overlap.
Full details and quality checks: [`eval-kit/EVAL_REPORT_REXGRADIENT.md`](eval-kit/EVAL_REPORT_REXGRADIENT.md).

| Metric | full | lora |
|---|---:|---:|
| BLEU-1 | 0.3251 | 0.2876 |
| BLEU-2 | 0.2572 | 0.2162 |
| BLEU-3 | 0.2202 | 0.1794 |
| BLEU-4 | 0.1962 | 0.1568 |
| CIDEr | 1.5061 | 1.2218 |
| ROUGE-L | 0.3866 | 0.3389 |
| CheXbert micro-F1 (5 classes) | 0.3150 | 0.2464 |
| CheXbert accuracy | 0.6198 | 0.6174 |
| SembScore | 0.5216 | 0.4777 |
| CheXbert micro-F1 (14 classes) | 0.4255 | 0.3854 |
| BERTScore precision | 0.8055 | 0.7853 |
| BERTScore recall | 0.7648 | 0.7442 |
| BERTScore F1 | 0.7833 | 0.7628 |
| F1 RadGraph | 0.3658 | 0.3147 |
| RaTEScore | 0.6145 | 0.5730 |

## Evaluate

```bash
pip install -e ./eval-kit
python -m evalkit -m bleu cider rouge \
  --ref-file refs.txt --hypo-file hypos.txt --mode corpus
```

For the complete API and model-metric setup, see
[`eval-kit/README.md`](eval-kit/README.md).

## Tests

```bash
python -m unittest -v test_main.py
python eval-kit/tests/test_evalkit.py
python eval-kit/tests/test_chexprompt.py
```

## Scope and reproducibility

The included SFT recipe follows the paper's three-epoch, `5e-5`, global batch
size 256 configuration. The paper's full result additionally requires all
training datasets, substantial multi-GPU compute, and two-stage GRPO. This
repository does not claim reproduction of the paper's reported scores. See
[`RESULTS.md`](RESULTS.md) for the local pipeline validation record.

## License and attribution

Project-authored code is released under the Apache License 2.0. Portions of
the evaluation implementation are adapted from third-party projects; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the upstream licenses
before redistribution.
