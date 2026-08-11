#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CheXbert + SembScore metric for eval-kit.

Adapted from ``rrg-metric/rrg_metric/chexbert`` (StanfordAIMI/RRG_scorers).
Adaptation is required because the original code path uses
``BertTokenizer.tokenize`` / ``encode_plus``, which were removed in
transformers>=5; here we use ``AutoTokenizer`` against a local
``bert-base-uncased`` directory, and load ``chexbert.pth`` from local weights.

Resources (local, no network; supplied separately under ``eval-kit/weights/``):
- weights: ``weights/chexbert.pth`` (override: ``EVALKIT_CHEXBERT_WEIGHTS``)
- tokenizer: ``weights/bert-base-uncased`` (override: ``EVALKIT_BERT_PATH``)

Public API
----------
>>> from evalkit.chexbert import CheXbert
>>> scorer = CheXbert()
>>> scorer.score_single(ref="heart is normal", hypo="heart is normal")
>>> scorer.score_batch(refs=[...], hypos=[...])
>>> scorer.score_corpus(refs=[...], hypos=[...])

The scorer follows the rrg-metric convention:
- F1 CheXbert: micro-F1 over 5 key findings (Cardiomegaly, Edema, Consolidation,
  Atelectasis, Pleural Effusion), with 4 label types collapsed to binary
  (Positive/Uncertain -> 1, Negative/Not-mentioned -> 0).
- SembScore: cosine similarity of CheXbert CLS embeddings (ref vs hypo).
- Additional keys mirror ``rrg_metric.eval`` (accuracy, micro/macro P/R/F1 for
  5 and 14 labels).
"""
from __future__ import annotations

import logging
import os
import re
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Bundled weights: <eval-kit>/weights (evalkit/ -> up two = eval-kit/).
_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"
)


def _default_weights() -> str:
    return os.environ.get("EVALKIT_CHEXBERT_WEIGHTS") or os.path.join(
        _WEIGHTS_DIR, "chexbert.pth"
    )


def _default_bert() -> str:
    return os.environ.get("EVALKIT_BERT_PATH") or os.path.join(
        _WEIGHTS_DIR, "bert-base-uncased"
    )


TARGET_NAMES: List[str] = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
    "No Finding",
]
TARGET_NAMES_5: List[str] = [
    "Cardiomegaly", "Edema", "Consolidation", "Atelectasis", "Pleural Effusion",
]

# 14 label heads: 13 conditions (4 classes) + "No Finding" (2 classes).
N_CLASSES = [4] * 13 + [2]


class _BertEncoder(nn.Module):
    """Unified BERT encoder/labeler (CheXbert architecture)."""

    def __init__(self, bert_path: str, cache_dir: str | None = None):
        super().__init__()
        from transformers import AutoConfig, AutoModel

        config = AutoConfig.from_pretrained(bert_path, cache_dir=cache_dir)
        self.bert = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.1)
        hidden = self.bert.pooler.dense.in_features
        self.linear_heads = nn.ModuleList(
            [nn.Linear(hidden, 4, bias=True) for _ in range(13)]
        )
        self.linear_heads.append(nn.Linear(hidden, 2, bias=True))

    def forward(self, source, attention_mask):
        final_hidden = self.bert(source, attention_mask=attention_mask)[0]
        cls_hidden = final_hidden[:, 0, :].squeeze(dim=1)
        cls_hidden_with_dropout = self.dropout(cls_hidden)
        logits = [
            self.linear_heads[i](cls_hidden_with_dropout) for i in range(14)
        ]
        return cls_hidden, logits


def _load_chexbert(
    weights_path: str | None = None,
    bert_path: str | None = None,
    device: torch.device | None = None,
    cache_dir: str | None = None,
) -> Tuple[nn.Module, Any]:
    """Load the CheXbert labeler model + tokenizer from local resources."""
    from transformers import AutoTokenizer

    weights_path = weights_path or _default_weights()
    bert_path = bert_path or _default_bert()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = _BertEncoder(bert_path, cache_dir=cache_dir)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"chexbert weights not found: {weights_path}. "
            "Set EVALKIT_CHEXBERT_WEIGHTS or place chexbert.pth under eval-kit/weights/."
        )
    raw = torch.load(weights_path, map_location=device, weights_only=True)
    state_dict = raw["model_state_dict"]
    state_dict = OrderedDict(
        (k.replace("module.", ""), v) for k, v in state_dict.items()
    )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # only dropout buffers are expected to be absent; surface real mismatches.
    real_missing = [m for m in missing if not m.endswith(".num_batches_tracked")]
    if real_missing:
        logger.warning("CheXbert state_dict missing keys: %s", real_missing[:5])
    if unexpected:
        logger.warning("CheXbert state_dict unexpected keys: %s", unexpected[:5])

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(bert_path, cache_dir=cache_dir)
    return model, tokenizer


def _clean(text: str) -> str:
    text = (text or "").strip()
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _labels_to_rrg_binary(labels: Sequence[int]) -> List[int]:
    """Collapse 4-class CheXbert labels to binary (rrg-metric convention).

    0 Blank/NM -> 0, 1 Positive -> 1, 2 Negative -> 0, 3 Uncertain -> 1.
    """
    mapping = {0: 0, 1: 1, 2: 0, 3: 1}
    return [mapping.get(int(c), 0) for c in labels]


class CheXbert:
    """CheXbert + SembScore scorer.

    Parameters
    ----------
    weights_path : str, optional
        Path to ``chexbert.pth``. Defaults to ``eval-kit/weights/chexbert.pth``.
    bert_path : str, optional
        Path to a local ``bert-base-uncased`` tokenizer/model directory.
        Defaults to ``eval-kit/weights/bert-base-uncased``.
    device : str or torch.device, optional
        Defaults to CUDA if available else CPU.
    cache_dir : str, optional
        HuggingFace cache dir (passed through; local paths don't need it).
    offline : bool, default True
        Force ``HF_HUB_OFFLINE=1`` / ``TRANSFORMERS_OFFLINE=1`` so the slow
        transformers 5.x loader does not try the network (proxy-blocked here).
    """

    def __init__(
        self,
        weights_path: str | None = None,
        bert_path: str | None = None,
        device: str | torch.device | None = None,
        cache_dir: str | None = None,
        offline: bool = True,
        batch_size: int | None = None,
    ):
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.batch_size = batch_size or int(
            os.environ.get("EVALKIT_CHEXBERT_BATCH_SIZE", "16")
        )
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        self.weights_path = weights_path or _default_weights()
        self.bert_path = bert_path or _default_bert()
        self.cache_dir = cache_dir

        logger.info("Loading CheXbert (weights=%s, bert=%s, device=%s)",
                    self.weights_path, self.bert_path, self.device)
        self.model, self.tokenizer = _load_chexbert(
            weights_path=self.weights_path,
            bert_path=self.bert_path,
            device=self.device,
            cache_dir=self.cache_dir,
        )
        self.target_names = list(TARGET_NAMES)
        self.target_names_5 = list(TARGET_NAMES_5)
        self.target_names_5_index = np.where(
            np.isin(self.target_names, self.target_names_5)
        )[0]

    # ------------------------------------------------------------------
    # Tokenization / inference
    # ------------------------------------------------------------------
    def _encode(self, texts: Sequence[str]) -> List[List[int]]:
        out: List[List[int]] = []
        cls, sep = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        for t in texts:
            t = _clean(t)
            if not t:
                out.append([cls, sep])
                continue
            ids = self.tokenizer(t, truncation=True, max_length=512)["input_ids"]
            out.append(ids)
        return out

    @torch.no_grad()
    def _label_batch(self, texts: Sequence[str]) -> Tuple[List[List[int]], torch.Tensor]:
        labels: List[List[int]] = []
        embeddings: List[torch.Tensor] = []
        pad_id = self.tokenizer.pad_token_id

        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            ids_list = self._encode(chunk)
            maxlen = max(len(x) for x in ids_list)
            padded = [x + [pad_id] * (maxlen - len(x)) for x in ids_list]
            src = torch.tensor(padded, dtype=torch.long, device=self.device)
            mask = torch.zeros(src.size(0), src.size(1), dtype=torch.float,
                               device=self.device)
            for i, ids in enumerate(ids_list):
                mask[i, :len(ids)] = 1
            cls_hidden, logits = self.model(src, mask)
            labels.extend([
                [lg[i].argmax(0).item() for lg in logits]
                for i in range(len(chunk))
            ])
            # Keep only one inference batch on the GPU at a time.
            embeddings.append(cls_hidden.detach().cpu())

        return labels, torch.cat(embeddings, dim=0)

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------
    def _compute(
        self, refs: Sequence[str], hypos: Sequence[str]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            f1_score,
        )

        if len(refs) != len(hypos):
            raise ValueError(
                f"refs/hypos length mismatch: {len(refs)} != {len(hypos)}"
            )
        n = len(refs)
        if n == 0:
            return {}, []

        ref_labels, ref_embeds = self._label_batch(list(refs))
        hyp_labels, hyp_embeds = self._label_batch(list(hypos))

        ref_bin = [_labels_to_rrg_binary(l) for l in ref_labels]
        hyp_bin = [_labels_to_rrg_binary(l) for l in hyp_labels]

        # ---- SembScore: cosine similarity of CLS embeddings ----
        ref_emb = ref_embeds.detach().cpu().numpy()
        hyp_emb = hyp_embeds.detach().cpu().numpy()
        sembs: List[float] = []
        for le, pe in zip(ref_emb, hyp_emb):
            denom = float(np.linalg.norm(le) * np.linalg.norm(pe))
            sembs.append(float((le * pe).sum() / denom) if denom > 0 else 0.0)

        # ---- F1 CheXbert ----
        ref5 = [np.array(r)[self.target_names_5_index] for r in ref_bin]
        hyp5 = [np.array(h)[self.target_names_5_index] for h in hyp_bin]

        cr = classification_report(
            ref_bin, hyp_bin, target_names=self.target_names,
            output_dict=True, zero_division=0,
        )
        cr5 = classification_report(
            ref5, hyp5, target_names=self.target_names_5,
            output_dict=True, zero_division=0,
        )
        accuracy = float(accuracy_score(ref5, hyp5))

        # per-element accuracy over 5 findings (matches rrg-metric pe_accuracy)
        y_true = np.array(ref5)
        y_pred = np.array(hyp5)
        pe_acc = (y_true == y_pred).all(axis=1).astype(np.float32) if n > 0 else np.array([])

        corpus: Dict[str, float] = {
            "f1chexbert": float(cr5["micro avg"]["f1-score"]),
            "sembscore": float(np.mean(sembs)),
            "f1chexbert_accuracy": accuracy,
            "f1chexbert_accuracy_not_averaged": float(np.mean(pe_acc)) if len(pe_acc) else 0.0,
            "f1chexbert_micro_precision_14": float(cr["micro avg"]["precision"]),
            "f1chexbert_micro_recall_14": float(cr["micro avg"]["recall"]),
            "f1chexbert_micro_f1_14": float(cr["micro avg"]["f1-score"]),
            "f1chexbert_micro_precision_5": float(cr5["micro avg"]["precision"]),
            "f1chexbert_micro_recall_5": float(cr5["micro avg"]["recall"]),
            "f1chexbert_micro_f1_5": float(cr5["micro avg"]["f1-score"]),
            "f1chexbert_macro_precision_14": float(cr["macro avg"]["precision"]),
            "f1chexbert_macro_recall_14": float(cr["macro avg"]["recall"]),
            "f1chexbert_macro_f1_14": float(cr["macro avg"]["f1-score"]),
            "f1chexbert_macro_precision_5": float(cr5["macro avg"]["precision"]),
            "f1chexbert_macro_recall_5": float(cr5["macro avg"]["recall"]),
            "f1chexbert_macro_f1_5": float(cr5["macro avg"]["f1-score"]),
        }

        per_item: List[Dict[str, float]] = []
        for i in range(n):
            ri5 = np.array(ref5[i]) if not isinstance(ref5[i], np.ndarray) else ref5[i]
            hi5 = np.array(hyp5[i]) if not isinstance(hyp5[i], np.ndarray) else hyp5[i]
            f1_i = float(f1_score(ri5, hi5, average="micro", zero_division=0))
            per_item.append({
                "f1chexbert": f1_i,
                "sembscore": sembs[i],
            })

        return corpus, per_item

    # ------------------------------------------------------------------
    # COCO-style adapter for Evaluator integration
    # ------------------------------------------------------------------
    def compute(
        self, gts: Mapping[int, List[str]], res: Mapping[int, List[str]]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        """Score from COCO-style dicts (keys=ids, gts[i]=ref-list, res[i]=[hypo]).

        Returns (corpus_dict, per_item_list). CheXbert expects a single
        reference per item; for multi-ref inputs the first reference is used.
        """
        keys = sorted(gts.keys())
        refs: List[str] = []
        hypos: List[str] = []
        for k in keys:
            r = gts[k]
            refs.append(r[0] if isinstance(r, (list, tuple)) and r else r)
            hypos.append(res[k][0])
        return self._compute(refs, hypos)

    # ------------------------------------------------------------------
    # High-level API (mirrors Evaluator)
    # ------------------------------------------------------------------
    def score_single(self, ref, hypo: str) -> Dict[str, float]:
        refs = [ref] if isinstance(ref, str) else (list(ref) or [""])
        # multi-ref: chexbert uses first reference
        ref0 = refs[0] if refs else ""
        corpus, per = self._compute([ref0], [hypo])
        out: Dict[str, float] = dict(per[0])
        # include corpus-level additional keys (accuracy, P/R/F1 over full set)
        for k, v in corpus.items():
            if k not in out:
                out[k] = v
        return out

    def score_batch(
        self, refs: Sequence, hypos: Sequence[str]
    ) -> List[Dict[str, float]]:
        flat_refs = [
            (r[0] if isinstance(r, (list, tuple)) and r else r)
            if not isinstance(r, str) else r
            for r in refs
        ]
        _, per = self._compute(list(flat_refs), list(hypos))
        return per

    def score_corpus(
        self, refs: Sequence, hypos: Sequence[str]
    ) -> Dict[str, float]:
        flat_refs = [
            (r[0] if isinstance(r, (list, tuple)) and r else r)
            if not isinstance(r, str) else r
            for r in refs
        ]
        corpus, _ = self._compute(list(flat_refs), list(hypos))
        return corpus


__all__ = ["CheXbert", "TARGET_NAMES", "TARGET_NAMES_5"]
