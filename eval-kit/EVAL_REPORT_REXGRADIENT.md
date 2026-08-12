# ReXGradient-160K 报告生成评测报告

> 评测对象：Qwen3-VL-8B-Instruct 的 full / LoRA SFT 最佳模型在 ReXGradient test 集（10000 条）上的生成结果。
> 最佳 checkpoint：full = `v0/checkpoint-1094`（eval_loss 0.6296），lora = `v0/checkpoint-1641`（eval_loss 0.8029）。
> 评测工具：`eval-kit`（本仓库 `eval-kit/`），7 个 metric 全量 10000 条。

## 1. 评测对象与数据

| 来源 | 文件 | 行数 | 角色 |
|---|---|---|---|
| 参考报告 (ref) | `uni-rg-cxr/artifacts/rexgradient_test.jsonl`（`answer` 字段） | 10000 | ground truth |
| 生成结果 full | `uni-rg-cxr/artifacts/rexgradient_test_full_pred.jsonl`（`response` 字段） | 10000 | full 模型生成 |
| 生成结果 lora | `uni-rg-cxr/artifacts/rexgradient_test_lora_pred.jsonl`（`response` 字段） | 10000 | lora 模型生成 |

两个预测文件的 `labels` 字段与 `rexgradient_test.jsonl.answer` **行序全对齐**（10000/10000），可直接按行配对评测。推理为 8× A100（TP 8）vLLM，temperature 0，max_new_tokens 512。

## 2. 截断 / 异常检查

| 检查项 | full | lora |
|---|---|---|
| 行数 | 10000 | 10000 |
| 空 response | 0 | 0 |
| 缺 `labels` | 0 | 0 |
| `labels` 对齐 ref.answer | 10000/10000 | 10000/10000 |
| response 长度 p5 / p50 / p95 | 146 / 197 / 421 | 169 / 197 / 411 |
| 重复 response | 7025 | 7979 |
| 重复 labels（真值，数据集特性） | 1807 | 1807 |

- 两套生成均无截断、无空输出（对比 IU-Xray 的 lora 拼接/截断问题，本次 lora 为干净的 10000 行单次推理）。
- 重复 response 比例（70–80%）明显高于真值重复（18%）：模型输出同质化偏强，读指标时需结合具体病例核查，可能使字面指标（BLEU/CIDEr）虚高。

## 3. 评测方法

- **工具**：`eval-kit`（`eval-kit/evalkit/`），metric：BLEU 1-4 / CIDEr / ROUGE-L / F1 CheXbert + SembScore / BERTScore / F1 RadGraph / RaTEScore。
- **参考来源**：`artifacts/rexgradient_test.jsonl` 的 `answer`；**生成来源**：预测文件 `response` 列。
- **权重**（`eval-repo-reference/eval-kit/weights/`）：CheXbert、bert-base-uncased（BERTScore）、RadGraph + BiomedNLP-PubMedBERT、RaTE-NER-Deberta + BioLORD-2023-C。
- **规模**：所有 metric 全量 10000 条。
- **时间线**（UTC，模型权重加载为瓶颈）：full 16:09:42 起（约 21 min），lora 17:08:12 起（约 26 min）。

## 4. 评测结果（全量 10000）

| metric | full | lora | Δ (full − lora) |
|---|---|---|---|
| Bleu_1 | 0.3251 | 0.2876 | +0.0375 |
| Bleu_2 | 0.2572 | 0.2162 | +0.0410 |
| Bleu_3 | 0.2202 | 0.1794 | +0.0408 |
| Bleu_4 | 0.1962 | 0.1568 | +0.0394 |
| CIDEr | 1.5061 | 1.2218 | +0.2843 |
| ROUGE_L | 0.3866 | 0.3389 | +0.0477 |
| f1chexbert（5 类 micro-F1） | 0.3150 | 0.2464 | +0.0686 |
| f1chexbert_accuracy | 0.6198 | 0.6174 | +0.0024 |
| sembscore | 0.5216 | 0.4777 | +0.0439 |
| f1chexbert_micro_f1_14 | 0.4255 | 0.3854 | +0.0401 |
| bertscore_precision | 0.8055 | 0.7853 | +0.0203 |
| bertscore_recall | 0.7648 | 0.7442 | +0.0206 |
| bertscore_f1 | 0.7833 | 0.7628 | +0.0205 |
| f1radgraph | 0.3658 | 0.3147 | +0.0510 |
| ratescore | 0.6145 | 0.5730 | +0.0415 |

## 5. 读数要点

- **full 在全部 15 项指标上优于 lora**（唯一接近的是 `f1chexbert_accuracy`，+0.0024）。
- **语义/临床指标差距 > 词汇指标**：CheXbert-F1（5 类）+0.069、F1RadGraph +0.051、CIDEr +0.284，而 BLEU 仅 +0.04、BERTScore-F1 +0.021——full 全参微调的优势集中在临床实体与语义正确性。
- 与训练 eval_loss 一致：full 0.6296 < lora 0.8029。
- **与 IU-Xray（590 条）对比**：IU 上 full/lora 接近（lora 甚至在 BLEU/ROUGE 略高），ReXGradient（10000 条）上 full 全面领先。规模更大、更依赖视觉-语义理解的数据集上，full 的参数容量与视觉塔训练优势更明显。
- 绝对分数受重复响应影响偏乐观，且 ReXGradient 报告模板化程度高于 IU，跨数据集直接比较指标需谨慎。

## 6. 产物文件

| 文件 | 说明 |
|---|---|
| `eval-kit/EVAL_REPORT_REXGRADIENT.md` | 本报告 |
| `uni-rg-cxr/artifacts/rexgradient_test_full_pred.jsonl` | full 预测（10000 行，与 ref 行对齐） |
| `uni-rg-cxr/artifacts/rexgradient_test_lora_pred.jsonl` | lora 预测（10000 行） |
| `uni-rg-cxr/artifacts/rexgradient_test_full_eval.json` / `_lora_eval.json` | 各 metric 聚合结果 |
| `uni-rg-cxr/artifacts/rexgradient_test_{full,lora}_eval_parts/` | 各 metric 分文件结果 |
| `uni-rg-cxr/artifacts/rexgradient_test_{full,lora}_eval.log` | 评测日志 |
