#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""BERTScore metric for eval-kit.

Adapted from the BERTScore algorithm (Zhang et al., 2020) as used by
``rrg-metric`` (which calls ``evaluate.load("bertscore")`` with ``lang="en"``).

Adaptation rationale
--------------------
The original path depends on the third-party ``bert_score`` package and the
HuggingFace ``evaluate`` metric, and downloads ``roberta-large`` from the hub.
Neither ``bert_score`` nor ``roberta-large`` is present in this environment,
and network access is proxy-blocked. To keep the module self-contained and
torch-version-neutral, BERTScore is reimplemented directly on ``transformers``,
loading any local pretrained model directory (default: the already-present
``bert-base-uncased``).

Numerical note: rrg-metric's default uses ``roberta-large`` (layer 17,
baseline-rescaled). This scorer defaults to ``bert-base-uncased`` (last hidden
layer, no baseline rescaling), so absolute scores differ from rrg-metric's
defaults; the BERTScore *method* is identical (greedy cosine matching of
contextual token embeddings). To reproduce rrg-metric numbers, point
``model_path`` at a local ``roberta-large`` snapshot and set ``layer=17``.

Resources (local, no network):
- model: ``weights/bert-base-uncased`` (provided separately under ``eval-kit/weights/``;
  override: ``EVALKIT_BERTSCORE_PATH`` or the ``model_path`` ctor arg)

Public API
----------
>>> from evalkit.bertscore import BertScore
>>> sc = BertScore()
>>> sc.score_single(ref="heart is normal", hypo="heart is normal")
>>> sc.score_batch(refs=[...], hypos=[...])
>>> sc.score_corpus(refs=[...], hypos=[...])
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Bundled weights: <eval-kit>/weights (evalkit/ -> up two = eval-kit/).
_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"
)


def _default_model() -> str:
    return os.environ.get("EVALKIT_BERTSCORE_PATH") or os.path.join(
        _WEIGHTS_DIR, "bert-base-uncased"
    )


def _clean(text: str) -> str:
    import re
    text = (text or "").strip().replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


class BertScore:
    """BERTScore (precision/recall/F1) over contextual token embeddings.

    Parameters
    ----------
    model_path : str, optional
        Local pretrained-model directory. Defaults to
        ``eval-kit/weights/bert-base-uncased``.
    layer : int, optional
        Hidden-state layer index to use (0 = embedding output,
        ``num_layers`` = last). ``-1`` means the last layer. Default ``-1``.
    idf : bool, default False
        If True, weight token contributions by inverse document frequency
        computed from the current corpus (same as BERTScore's idf mode).
    batch_size : int, default 16
        Encoding batch size.
    device : str or torch.device, optional
        Defaults to CUDA if available else CPU.
    offline : bool, default True
        Force HF offline env vars so transformers does not probe the network.
    """

    def __init__(
        self,
        model_path: str | None = None,
        layer: int = -1,
        idf: bool = False,
        batch_size: int | None = None,
        device: str | torch.device | None = None,
        offline: bool = True,
    ):
        if offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

        self.model_path = model_path or _default_model()
        self.layer = layer
        self.idf = idf
        self.batch_size = batch_size or int(
            os.environ.get("EVALKIT_BERTSCORE_BATCH_SIZE", "16")
        )
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)

        from transformers import AutoModel, AutoTokenizer

        logger.info("Loading BERTScore model %s on %s", self.model_path, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModel.from_pretrained(self.model_path).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # number of transformer layers; hidden_states[0] is embeddings.
        with torch.no_grad():
            n = self.model.config.num_hidden_layers
        self.num_layers = n

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _encode(self, texts: Sequence[str]) -> List[torch.Tensor]:
        """Return per-text contextual token embeddings (special tokens removed)."""
        embs: List[torch.Tensor] = []
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        bos_eos = {cls_id, sep_id, self.tokenizer.pad_token_id}
        # some tokenizers (roberta) use bos/eos ids instead of cls/sep
        bos = getattr(self.tokenizer, "bos_token_id", None)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if bos is not None:
            bos_eos.add(bos)
        if eos is not None:
            bos_eos.add(eos)

        for start in range(0, len(texts), self.batch_size):
            batch = [t for t in texts[start:start + self.batch_size]]
            cleaned = [_clean(t) for t in batch]
            enc = self.tokenizer(
                cleaned, return_tensors="pt", truncation=True,
                max_length=512, padding=True,
            ).to(self.device)
            out = self.model(**enc, output_hidden_states=True)
            hs = out.hidden_states[self.layer]  # (B, T, H)
            attn = enc["attention_mask"]  # (B, T)
            ids = enc["input_ids"]  # (B, T)
            for i in range(len(batch)):
                mask = (attn[i].bool()) & (~torch.isin(ids[i], torch.tensor(list(bos_eos), device=self.device)))
                # Do not retain every corpus embedding on the GPU. Without
                # this copy, a full evaluation grows GPU memory with every
                # batch and can eventually OOM even though encoding is batched.
                embs.append(hs[i][mask].detach().cpu())
        return embs

    # ------------------------------------------------------------------
    # IDF
    # ------------------------------------------------------------------
    def _compute_idf(self, refs_embs_ids: List[torch.Tensor]) -> Dict[Any, float]:
        # idf over reference token ids (BERTScore convention).
        from collections import Counter
        cnt: Counter = Counter()
        n_docs = len(refs_embs_ids)
        for ids in refs_embs_ids:
            cnt.update(set(ids.tolist()))
        idf: Dict[Any, float] = {}
        for tok, c in cnt.items():
            idf[tok] = np.log((n_docs + 1) / (c + 1)) + 1.0
        return idf

    # ------------------------------------------------------------------
    # Core compute
    # ------------------------------------------------------------------
    def _pair_scores(
        self,
        ref_emb: torch.Tensor,
        hyp_emb: torch.Tensor,
        ref_ids: torch.Tensor | None,
        hyp_ids: torch.Tensor | None,
        idf: Dict[Any, float] | None,
    ) -> Tuple[float, float, float]:
        if ref_emb.numel() == 0 or hyp_emb.numel() == 0:
            return 0.0, 0.0, 0.0
        r = F.normalize(ref_emb.float(), dim=-1)
        h = F.normalize(hyp_emb.float(), dim=-1)
        sim = r @ h.t()  # (R, H)

        if idf is not None and ref_ids is not None and hyp_ids is not None:
            # BERTScore idf: idf weights on reference tokens for recall,
            # uniform for precision (per the original implementation).
            w_ref = torch.tensor([idf.get(int(t), 1.0) for t in ref_ids.tolist()],
                                 device=ref_emb.device, dtype=torch.float32)
            w_ref = w_ref / w_ref.sum().clamp_min(1e-8)
            recall = (sim.max(dim=1).values * w_ref).sum().item()
            precision = (sim.max(dim=0).values).mean().item()
        else:
            recall = sim.max(dim=1).values.mean().item()
            precision = sim.max(dim=0).values.mean().item()
        f1 = 0.0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        return precision, recall, f1

    def _compute(
        self, refs: Sequence[str], hypos: Sequence[str]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        if len(refs) != len(hypos):
            raise ValueError(f"refs/hypos length mismatch: {len(refs)} != {len(hypos)}")
        n = len(refs)
        if n == 0:
            return {}, []

        ref_embs = self._encode(list(refs))
        hyp_embs = self._encode(list(hypos))

        # idf (optional) uses reference token ids; re-tokenize to recover ids
        # without special tokens (cheap; same cleaning as encoding).
        idf: Dict[Any, float] | None = None
        ref_ids_list: List[torch.Tensor] | None = None
        if self.idf:
            ref_ids_list = self._token_ids(list(refs))
            idf = self._compute_idf(ref_ids_list)

        per_item: List[Dict[str, float]] = []
        precs: List[float] = []
        recs: List[float] = []
        f1s: List[float] = []
        for i in range(n):
            r_ids = ref_ids_list[i] if ref_ids_list is not None else None
            h_ids = None  # precision uses uniform weights in idf mode
            p, r, f = self._pair_scores(ref_embs[i], hyp_embs[i], r_ids, h_ids, idf)
            precs.append(p)
            recs.append(r)
            f1s.append(f)
            per_item.append({
                "bertscore_precision": p,
                "bertscore_recall": r,
                "bertscore_f1": f,
            })
        corpus = {
            "bertscore_precision": float(np.mean(precs)),
            "bertscore_recall": float(np.mean(recs)),
            "bertscore_f1": float(np.mean(f1s)),
        }
        return corpus, per_item

    def _token_ids(self, texts: Sequence[str]) -> List[torch.Tensor]:
        """Token ids per text (special tokens removed), for idf."""
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        drop = {cls_id, sep_id, self.tokenizer.pad_token_id}
        bos = getattr(self.tokenizer, "bos_token_id", None)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if bos is not None:
            drop.add(bos)
        if eos is not None:
            drop.add(eos)
        out: List[torch.Tensor] = []
        for t in texts:
            enc = self.tokenizer(_clean(t), truncation=True, max_length=512)
            ids = torch.tensor(
                [x for x in enc["input_ids"] if x not in drop],
                device=self.device,
            )
            out.append(ids)
        return out

    # ------------------------------------------------------------------
    # COCO-style adapter
    # ------------------------------------------------------------------
    def compute(
        self, gts: Mapping[int, List[str]], res: Mapping[int, List[str]]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        keys = sorted(gts.keys())
        refs: List[str] = []
        hypos: List[str] = []
        for k in keys:
            r = gts[k]
            refs.append(r[0] if isinstance(r, (list, tuple)) and r else r)
            hypos.append(res[k][0])
        return self._compute(refs, hypos)

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------
    def score_single(self, ref, hypo: str) -> Dict[str, float]:
        ref0 = ref[0] if isinstance(ref, (list, tuple)) and ref else ref
        if not isinstance(ref0, str):
            ref0 = str(ref0) if ref0 is not None else ""
        corpus, per = self._compute([ref0], [hypo])
        out: Dict[str, float] = dict(per[0])
        out.update(corpus)
        return out

    def score_batch(
        self, refs: Sequence, hypos: Sequence[str]
    ) -> List[Dict[str, float]]:
        flat = [
            (r[0] if isinstance(r, (list, tuple)) and r else r)
            if not isinstance(r, str) else r
            for r in refs
        ]
        _, per = self._compute(list(flat), list(hypos))
        return per

    def score_corpus(
        self, refs: Sequence, hypos: Sequence[str]
    ) -> Dict[str, float]:
        flat = [
            (r[0] if isinstance(r, (list, tuple)) and r else r)
            if not isinstance(r, str) else r
            for r in refs
        ]
        corpus, _ = self._compute(list(flat), list(hypos))
        return corpus


__all__ = ["BertScore"]
