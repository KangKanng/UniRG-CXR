# eval-kit

Self-contained metrics for chest X-ray radiology report evaluation.

NLG metrics (BLEU/CIDEr/ROUGE-L) are ported from `R2GenGPT/evalcap`
(MS-COCO caption evaluation toolkit). Model-based metrics (CheXbert,
BERTScore, F1 RadGraph, RaTEScore) are adapted from `rrg-metric` and load
from local weights.

| Metric   | Status | Notes |
|---------|--------|-------|
| BLEU 1-4 | ✅ ported | pure Python, no java |
| CIDEr   | ✅ ported | pure Python, no java |
| ROUGE-L | ✅ ported | pure Python, no java |
| F1 CheXbert + SembScore | ✅ adapted | torch + transformers; local `chexbert.pth` + `bert-base-uncased` |
| BERTScore (P/R/F1) | ✅ adapted | torch + transformers; local `bert-base-uncased` (swap to `roberta-large` for rrg-metric parity) |
| F1 RadGraph | ✅ adapted | torch + radgraph (allennlp); local `radgraph.tar.gz` + PubMedBERT |
| RaTEScore | ✅ adapted | torch + transformers + medspacy; local RaTE-NER + BioLORD |
| CheXprompt | ✅ adapted | official 6-error rubric and 5-shot prompt; OpenAI-compatible API |
| METEOR  | ❌ skipped | requires `meteor-1.5.jar` + Java; not adapted here |
| PTBTokenizer | ❌ skipped | requires `stanford-corenlp-3.4.1.jar` + Java |

The NLG metrics use the original `str.split()` tokenization (whitespace),
matching R2GenGPT's behavior when the PTBTokenizer is bypassed. This keeps
that path dependency-free while preserving the scoring math.

Model metrics require local weights (no network at runtime). Weights are not
distributed with this repository. Put them under `eval-kit/weights/` using the
layout below, or configure the corresponding environment variables:

| Path | Used by |
|------|---------|
| `weights/chexbert.pth` | CheXbert head |
| `weights/bert-base-uncased/` | CheXbert tokenizer + BERTScore model |
| `weights/radgraph.tar.gz` (+ extracted `weights/radgraph/`) | F1 RadGraph (default); `-xl`/`echograph` are also supported |
| `weights/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext/` | F1 RadGraph embedder |
| `weights/RaTE-NER-Deberta/` | RaTEScore NER |
| `weights/BioLORD-2023-C/` | RaTEScore synonym embedding |

Env-var overrides (checked first): `EVALKIT_CHEXBERT_WEIGHTS`,
`EVALKIT_BERT_PATH`, `EVALKIT_BERTSCORE_PATH`, `EVALKIT_PUBMEDBERT_PATH`,
`EVALKIT_RATE_NER_PATH`, `EVALKIT_BIOLORD_PATH`. For rrg-metric BERTScore
parity, point `EVALKIT_BERTSCORE_PATH` at a local `roberta-large` snapshot and
pass `layer=17` to `BertScore(...)`.

All model scorers force `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1` so the
slow transformers 5.x loader does not try the network.

## Install (editable)

```bash
cd eval-kit
pip install -e .
```

### 一键搭建独立环境（不碰训练基础环境）

`setup_env.sh` 为 eval-kit 建一个**独立 venv**（默认
`eval-kit/.venv-evalkit`）。若当前环境已安装 torch / transformers，脚本会
沿用其版本；也可用环境变量显式指定版本：

```bash
cd eval-kit
./setup_env.sh            # 建环境 + 安装（幂等）
./setup_env.sh --test     # 安装后跑 NLG 对拍测试 + 模型 metric 冒烟
./setup_env.sh --recreate # 删旧重建

# 运行
.venv-evalkit/bin/python -m evalkit -m bleu cider rouge \
  --ref "heart is normal" --hypo "heart is normal"
```

可选覆盖：`EVALKIT_BASE_PYTHON`（基础环境 python，用于读取锁定版本）、
`EVALKIT_TORCH_VERSION` / `EVALKIT_TRANSFORMERS_VERSION`（锁定版本）、
`EVALKIT_VENV_DIR`（venv 位置）、`EVALKIT_PYTHON`（建 venv 的 python）。
模型权重不随仓库分发；请放到 `eval-kit/weights/` 的约定路径，或设置下文的环境变量。

## Python API

```python
from evalkit import Evaluator

ev = Evaluator(metrics=["bleu", "cider", "rouge"])

# single pair, single reference
ev.score_single(ref="heart is normal", hypo="heart is normal")
# {'Bleu_1': 1.0, 'Bleu_2': 1.0, 'Bleu_3': 1.0, 'Bleu_4': 0.03,
#  'CIDEr': 7.5, 'ROUGE_L': 1.0}

# single pair, multiple references
ev.score_single(ref=["heart is normal", "normal heart"], hypo="heart is normal")

# batch: per-item scores
ev.score_batch(refs=["heart is normal", "lungs are clear"],
              hypos=["heart is normal", "lungs are clear"])

# corpus: aggregate scores (one value per metric)
ev.score_corpus(refs=["heart is normal", "lungs are clear"],
                hypos=["heart is normal", "lungs are clear"])
```

Select a subset of metrics:

```python
Evaluator(metrics="cider")          # one metric
Evaluator(metrics=["bleu", "rouge"]) # several
Evaluator(metrics="chexbert")       # model metric (lazy load)
Evaluator(metrics="bertscore")     # model metric (lazy load)
```

### CheXbert (F1 CheXbert + SembScore)

CheXbert is a model metric; the first call lazily loads `chexbert.pth` +
`bert-base-uncased` (~tens of seconds). It yields the binary F1 over the five
key findings (`f1chexbert`) and the CLS-embedding cosine similarity
(`sembscore`); corpus mode also surfaces accuracy and micro/macro P/R/F1 for
the 5- and 14-label sets, mirroring `rrg_metric.eval`.

```python
ev = Evaluator(metrics="chexbert")
ev.score_single(ref="No acute cardiopulmonary process.",
                hypo="No acute cardiopulmonary abnormality.")
# {'f1chexbert': 1.0, 'sembscore': 0.9898, 'f1chexbert_accuracy': 1.0, ...}
ev.score_corpus(refs=[...], hypos=[...])
# {'f1chexbert': 0.6667, 'sembscore': 0.8139,
#  'f1chexbert_micro_f1_5': 0.6667, 'f1chexbert_macro_f1_14': 0.1429, ...}
```
CheXbert can be mixed with NLG metrics:

```python
Evaluator(metrics=["bleu", "chexbert"])
```

### BERTScore

BERTScore measures precision/recall/F1 over contextual token embeddings. The
first call lazily loads the model (~tens of seconds). Identical text yields
P=R=F1=1.0.

```python
ev = Evaluator(metrics="bertscore")
ev.score_single(ref="heart is normal", hypo="heart is normal")
# {'bertscore_precision': 1.0, 'bertscore_recall': 1.0, 'bertscore_f1': 1.0}
```

For rrg-metric parity (roberta-large, layer 17), instantiate directly:

```python
from evalkit.bertscore import BertScore
sc = BertScore(model_path="path/to/roberta-large", layer=17)
```

### F1 RadGraph

F1 RadGraph scores entity-graph overlap between reference and hypothesis
reports (reward levels: simple/partial/complete/all). The default model is
`radgraph` (rrg-metric default; `-xl` and `echograph` also supported). Uses
a local `weights/radgraph.tar.gz` + PubMedBERT (override:
`EVALKIT_PUBMEDBERT_PATH`).

```python
ev = Evaluator(metrics="f1radgraph")
ev.score_single(ref="No acute cardiopulmonary process.",
                hypo="No acute cardiopulmonary abnormality.")
ev.score_corpus(refs=[...], hypos=[...])  # {'f1radgraph': ...}
```

For `reward_level="all"`, instantiate directly to get precision/recall/f1:

```python
from evalkit.radgraph_scorer import F1RadGraphScorer
sc = F1RadGraphScorer(reward_level="all")  # -> f1radgraph_precision/recall/f1
```

### RaTEScore

RaTEScore is an entity-aware metric (EMNLP 2024). Uses a local RaTE-NER
(`weights/RaTE-NER-Deberta`) and BioLORD (`weights/BioLORD-2023-C`) models
(overrides: `EVALKIT_RATE_NER_PATH` / `EVALKIT_BIOLORD_PATH`).

```python
ev = Evaluator(metrics="ratescore")
ev.score_single(ref="No acute cardiopulmonary process.",
                hypo="No acute cardiopulmonary abnormality.")
# {'ratescore': ...}
ev.score_corpus(refs=[...], hypos=[...])  # {'ratescore': <mean>}
```

### CheXprompt

CheXprompt uses the official Microsoft rubric and five in-context examples. It
returns total, clinically significant, and clinically insignificant error
counts, plus the UniRG reward `1 / (errors + 1)`. Configure any
OpenAI-compatible chat-completions endpoint:

```bash
export EVALKIT_CHEXPROMPT_BASE_URL="https://your-host/v1"
export EVALKIT_CHEXPROMPT_API_KEY="your-key"
export EVALKIT_CHEXPROMPT_MODEL="gpt-4.1"
export EVALKIT_CHEXPROMPT_CACHE="/path/to/chexprompt-cache.sqlite3"
```

The same values can be placed in `eval-kit/.env`; CheXprompt loads that file
automatically without overriding variables already exported by the shell. Copy
`.env.example` to `.env`, then fill in the endpoint, API key, and model name.

```python
from evalkit import CheXpromptScorer

scorer = CheXpromptScorer(
    base_url="https://your-host/v1",  # or full .../chat/completions URL
    api_key="your-key",
    model="gpt-4.1",
    max_workers=8,
    requests_per_minute=60,
    cache_path="chexprompt-cache.sqlite3",
)
score = scorer.score_single(reference_report, candidate_report)
# chexprompt_errors, chexprompt_significant_errors,
# chexprompt_insignificant_errors, chexprompt_reward, and per-category counts
```

It is also available through `Evaluator(metrics="chexprompt")`. API or parse
failures raise an exception instead of assigning a misleading zero-error
reward. For reproducible evaluation and RL, pin the judge model/version and
keep the SQLite cache.

## CLI

```bash
# single literal pair
python -m evalkit -m cider --ref "heart is normal" --hypo "heart is normal"

# batch from two line-aligned files (per-item scores)
python -m evalkit -m bleu cider rouge --ref-file refs.txt --hypo-file hypos.txt

# corpus aggregate
python -m evalkit -m bleu cider rouge --ref-file refs.txt --hypo-file hypos.txt --mode corpus

# CheXbert (single pair)
python -m evalkit -m chexbert --ref "No acute cardiopulmonary process." --hypo "No acute cardiopulmonary abnormality."

# CheXbert batch + corpus aggregate
python -m evalkit -m chexbert --ref-file refs.txt --hypo-file hypos.txt --mode corpus

# BERTScore (single pair / corpus)
python -m evalkit -m bertscore --ref "heart is normal" --hypo "heart is normal"
python -m evalkit -m bertscore --ref-file refs.txt --hypo-file hypos.txt --mode corpus

# F1 RadGraph (single pair / corpus)
python -m evalkit -m f1radgraph --ref "No acute cardiopulmonary process." --hypo "No acute cardiopulmonary abnormality."
python -m evalkit -m f1radgraph --ref-file refs.txt --hypo-file hypos.txt --mode corpus

# RaTEScore (single pair / corpus)
python -m evalkit -m ratescore --ref "No acute cardiopulmonary process." --hypo "No acute cardiopulmonary abnormality."
python -m evalkit -m ratescore --ref-file refs.txt --hypo-file hypos.txt --mode corpus

# CheXprompt (requires the EVALKIT_CHEXPROMPT_* variables above)
python -m evalkit -m chexprompt --ref "No pleural effusion." --hypo "Small pleural effusion."

# multiple references per item: one JSON list per line in the ref file
python -m evalkit --multi-ref --ref-file refs.jsonl --hypo-file hypos.txt
```

## Datasets (IU-Xray / ReXrank)

`evalkit.datasets` reads report-generation manifests and exposes a uniform
`[{"ref": str, "images": [...], ...}]` list so the Evaluator can score them
against a hypothesis source.

| Name | Manifest | Reference field | Default root |
|------|----------|-----------------|--------------|
| `rexrank` | `<root>/<split>.jsonl` (`query`/`response`/`images`) | `response` | `uni-rg-cxr/data/rexrank` |
| `artifacts` / `iu` | `<root>/iu_<split>.jsonl` (`answer`/`images`/`prompt`) | `answer` | `uni-rg-cxr/artifacts` |

Splits: `train` / `valid` / `test` (590 / 296 / 590 for IU-Xray). Roots are
overridable via `--ref-root`/`--hypo-root` or `EVALKIT_REXRANK_ROOT` /
`EVALKIT_ARTIFACTS_ROOT`.

```python
from evalkit.datasets import load_dataset, refs_from
items = load_dataset("artifacts", "test")      # 590 items
refs = refs_from(items)                         # list of reference reports
```

### CLI: score a dataset against a hypothesis file/source

```bash
# refs from dataset, hypos from a generated-reports file (one per line)
python -m evalkit -m bleu cider rouge chexbert \
  --ref-dataset artifacts --ref-split test \
  --hypo-file generated.txt --mode corpus

# self-evaluate (hypo = same dataset's reference column; sanity/walk-through)
python -m evalkit -m bleu cider rouge \
  --ref-dataset artifacts --ref-split test \
  --hypo-dataset artifacts --hypo-split test --limit 10 --mode corpus

# cross-manifest: artifacts refs vs rexrank hypos
python -m evalkit -m bleu cider rouge \
  --ref-dataset artifacts --ref-split test \
  --hypo-dataset rexrank --hypo-split test --mode batch
```

`--limit N` caps items (smoke runs). Model metrics (`chexbert`/`bertscore`/
`f1radgraph`/`ratescore`) work the same way; set their model-path env vars as
above.

Prediction JSONL files containing both the reference and generated report can
be evaluated directly. By default, `labels` is the reference and `response`
is the hypothesis:

```bash
python -m evalkit -m bleu cider rouge \
  --pred-file rexgradient_test_full_pred.jsonl --mode corpus
```

Use `--ref-key` and `--hypo-key` when a file uses different field names.

## Layout

```
eval-kit/
├── pyproject.toml
├── README.md
├── weights/           # user-provided model weights (ignored by Git)
└── evalkit/
    ├── __init__.py
    ├── evaluator.py     # Evaluator + CLI
    ├── bleu.py          # Bleu + BleuScorer
    ├── cider.py         # Cider + CiderScorer
    ├── rouge.py         # Rouge
    ├── chexbert.py      # CheXbert + SembScore (model metric)
    ├── bertscore.py     # BERTScore P/R/F1 (model metric)
    ├── radgraph_scorer.py  # F1 RadGraph (model metric, radgraph backend)
    ├── ratescore.py     # RaTEScore (model metric, RaTEScore backend)
    ├── chexprompt.py   # API-based CheXprompt error metric
    └── datasets.py      # rexrank / artifacts (IU-Xray) loaders
