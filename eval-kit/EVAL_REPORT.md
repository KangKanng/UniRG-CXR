# IU-Xray 报告生成评测报告

> 评测对象：Qwen3-VL-8B-Instruct 在 IU-Xray（R2Gen split）test 集上的两种生成结果。
> 评测工具：`eval-kit`（本仓库 `eval-kit/`），7 个 metric 全量 590 条。

## 1. 评测对象与数据

| 来源 | 文件 | 行数 | 角色 |
|---|---|---|---|
| 参考报告 (ref) | `uni-rg-cxr/artifacts/iu_test.jsonl` (`answer` 字段) | 590 | ground truth |
| 生成结果 A | `uni-rg-cxr/artifacts/iu_test_full_pred.jsonl` (`response` 字段) | 590 | full 模型生成 |
| 生成结果 B | `uni-rg-cxr/artifacts/iu_test_lora_pred.jsonl` (`response` 字段) | 1180 | lora 生成（异常，见 §2） |

`iu_test.jsonl` 与两个预测文件**行序对齐**（`labels` 字段 == `answer`，590/590 全对齐），可直接按行配对评测。

## 2. 截断 / 异常检查

### 2.1 `iu_test_full_pred.jsonl` — ✅ 无截断
- 590 行，与 test 集一致。
- 全部含 `Findings` 与 `Impression`（590/590）。
- 末尾字符全部为句号 `.`（590/590），无非正常结尾。
- `labels` 全部对齐 `iu_test.jsonl.answer`（590/590）。
- response 长度 min/med/max = 122 / 122 / 249，分布正常。

### 2.2 `iu_test_lora_pred.jsonl` — ⚠️ 异常拼接 + 部分截断
- **行数 1180 = 590 × 2**：前半 590 与后半 590 是针对同一 test 集的**两次独立推理**（response 不同，但两段 `labels` 都与 `iu_test.jsonl.answer` 对齐 590/590）。无法直接与 590 条 ref 配对，故按顺序拆为两段分别评测：
  - `lora_a` = 前 590 行 → 提取到 `artifacts/lora_hypos_a.txt`
  - `lora_b` = 后 590 行 → 提取到 `artifacts/lora_hypos_b.txt`
- 截断统计（每段独立）：

| 段 | 缺 `Impression` | 非句号结尾 | 末尾字符分布 |
|---|---|---|---|
| lora_a (前 590) | 34 | 31 | `.` 559, `l` 8, `s` 5, `e` 4, `b` 4 |
| lora_b (后 590) | 43 | 37 | `.` 553, `f` 15, `l` 8, `t` 3, `s` 3 |

- lora_b 含 max 长度 2530 的异常长 response（疑似重复拼接），拖慢 F1RadGraph 推理（见 §4 耗时）。

> 结论：full 干净可直接评测；lora 需分两段，且每段约有 5–7% 截断，结果仅供参考。

## 3. 评测方法

- **工具**：`eval-kit`（`eval-kit/evalkit/`），metric：BLEU 1-4 / CIDEr / ROUGE-L / F1 CheXbert + SembScore / BERTScore / F1 RadGraph / RaTEScore。
- **参考来源**：`--ref-dataset artifacts --ref-split test`（读 `iu_test.jsonl` 的 `answer`）。
- **生成来源**：预测文件 `response` 列提取为行对齐 txt，经 `--hypo-file` 喂入。
- **环境**（torch 中性，未改 `torch 2.13.0+cu130` / `transformers 5.12.1`，离线）：
  - CheXbert：`eval-kit/weights/chexbert.pth` + `eval-kit/weights/bert-base-uncased`
  - BERTScore：`eval-kit/weights/bert-base-uncased`（最后一层，无 baseline rescaling）
  - F1 RadGraph：`eval-kit/weights/radgraph.tar.gz` + `eval-kit/weights/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext`
  - RaTEScore：`eval-kit/weights/RaTE-NER-Deberta` + `eval-kit/weights/BioLORD-2023-C`

  （权重均已收拢至 `eval-kit/weights/` 并作为默认值自动解析；`EVALKIT_*_PATH` 仅为可选覆盖。）
- **规模**：所有 metric 均**全量 590 条**评测。
- 复现（示例）：
```bash
cd eval-kit
# NLG 全量
python -m evalkit -m bleu cider rouge \
  --ref-dataset artifacts --ref-split test \
  --hypo-file ../uni-rg-cxr/artifacts/full_hypos.txt --mode corpus

# 模型 metric 全量（权重在 eval-kit/weights/ 下自动解析）
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m evalkit -m f1radgraph \
  --ref-dataset artifacts --ref-split test \
  --hypo-file ../uni-rg-cxr/artifacts/full_hypos.txt --mode corpus
```

## 4. 评测结果（全量 590，3 套生成结果）

### 4.1 NLG + 模型语义 metric

| metric | full | lora_a | lora_b |
|---|---|---|---|
| Bleu_1 | 0.1829 | 0.2090 | 0.1943 |
| Bleu_2 | 0.1171 | 0.1382 | 0.1264 |
| Bleu_3 | 0.0807 | 0.0962 | 0.0869 |
| Bleu_4 | 0.0548 | 0.0655 | 0.0582 |
| CIDEr | 0.2952 | 0.1856 | 0.1718 |
| ROUGE_L | 0.2848 | 0.2909 | 0.2736 |
| f1chexbert (5 类 micro-F1) | 0.0000 | 0.2566 | 0.2034 |
| f1chexbert_accuracy | 0.8017 | 0.7780 | 0.6881 |
| sembscore | 0.5156 | 0.4865 | 0.4670 |
| f1chexbert_micro_f1_14 | 0.5233 | 0.5460 | 0.4764 |
| bertscore_precision | 0.7929 | 0.7986 | 0.7722 |
| bertscore_recall | 0.6885 | 0.6798 | 0.6788 |
| bertscore_f1 | 0.7359 | 0.7326 | 0.7201 |
| f1radgraph | 0.2456 | 0.2359 | 0.2137 |
| ratescore | 0.5762 | 0.5837 | 0.5581 |

### 4.2 耗时（模型已缓存，加载为瓶颈）

| metric | full（首套，含加载） | 复用后 lora_a | 复用后 lora_b |
|---|---|---|---|
| f1radgraph | 450.2 s | 51.0 s | 210.3 s |
| ratescore | 557.6 s | 99.0 s | 134.8 s |
| chexbert+bertscore | 289.4 s | 5.2 s | 11.1 s |
| NLG (bleu/cider/rouge) | <2 s | <2 s | <2 s |

- lora_b 的 F1RadGraph 耗时（210 s）明显高于 lora_a（51 s），原因是 lora_b 含异常长报告（max 2530 字符），dygie 长序列 forward 更慢。
- chexbert/bertscore 推理快（GPU 批量），加载占 ~280 s；f1radgraph/ratescore 为逐条推理，加载 + 推理都贡献耗时。

## 5. 读数要点

- **full vs lora**：整体接近。full 在 CIDEr（0.295 vs 0.186/0.172）、BERTScore_f1（0.736 vs 0.733/0.720）、F1RadGraph（0.246 vs 0.236/0.214）上略高；lora 在 BLEU/ROUGE/RaTEScore 上略高。
- **f1chexbert 5 类**：full = 0.000（5 类全阴性，无阳性重叠），lora 非零（lora_a 0.257 / lora_b 0.203）——lora 报告更倾向"写出"5 类阳性词，但准确率低于 full（accuracy 0.69–0.78 vs 0.80）。读这条时需结合 accuracy 看：lora 召回更高但误报更多。
- **lora 两半**：lora_a 普遍略优于 lora_b（chexbert/bertscore/f1radgraph 均 a > b），且 lora_b 截断更多（43 vs 34）并含异常长报告。若需 lora 单值，建议用 `lora_a` 或两半平均。
- **绝对分数偏低**：IU-Xray test 的 `answer` 与各模型生成在措辞/格式上差异大（参考报告含 `XXXX` 脱敏占位、Findings/Impression 结构），导致字面 metric（BLEU/CIDEr）整体偏低属正常；语义 metric（BERTScore/RaTEScore）更能反映实际相似度。

## 6. 产物文件

| 文件 | 说明 |
|---|---|
| `eval-kit/EVAL_REPORT.md` | 本报告 |
| `uni-rg-cxr/artifacts/full_hypos.txt` | full 预测提取（590 行，与 ref 行对齐） |
| `uni-rg-cxr/artifacts/lora_hypos_a.txt` | lora 前半预测提取（590 行） |
| `uni-rg-cxr/artifacts/lora_hypos_b.txt` | lora 后半预测提取（590 行） |
