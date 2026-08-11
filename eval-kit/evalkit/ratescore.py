#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RaTEScore metric for eval-kit.

Adapted from rrg-metric (ratescore branch), which uses
`from RaTEScore import RaTEScore` directly. This module wraps that package
for the eval-kit Evaluator API.

Environment notes (this repo)
- RaTEScore ships as source under <repo>/RaTEScore/ but, from the repo root,
  resolves to a namespace package. We prepend <repo>/RaTEScore to sys.path so
  `import RaTEScore` finds the real package. RaTEScore is also installed
  editable (`pip install -e ./RaTEScore --no-deps`).
- Runtime dep installed into the venv (torch-neutral): medspacy (pulls spacy,
  PyRuSH, PyFastNER). Its import is slow (~40s) but works offline.
- Models (supplied separately under ``eval-kit/weights/``; no network at runtime):
  - NER: Angelakeke/RaTE-NER-Deberta -> ``weights/RaTE-NER-Deberta``
    (override: EVALKIT_RATE_NER_PATH)
  - embedding: FremyCompany/BioLORD-2023-C -> ``weights/BioLORD-2023-C``
    (override: EVALKIT_BIOLORD_PATH)
  Constructor args take precedence over env vars; offline env vars are
  forced so transformers does not probe the network.

Public API
>>> from evalkit.ratescore import RaTEScoreScorer
>>> sc = RaTEScoreScorer()
>>> sc.score_single(ref="...", hypo="...")
>>> sc.score_batch(refs=[...], hypos=[...])
>>> sc.score_corpus(refs=[...], hypos=[...])

Note: RaTEScore.compute_score(candidate_list, reference_list) takes candidates
(hypotheses) first; this wrapper follows rrg-metric's convention
compute_score(preds, gts).
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_RATESCORE_SRC = os.path.join(_REPO_ROOT, "RaTEScore")
# Bundled weights: <eval-kit>/weights (evalkit/ -> up two = eval-kit/).
_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"
)


def _ensure_ratescore_importable() -> None:
    if _RATESCORE_SRC not in sys.path:
        sys.path.insert(0, _RATESCORE_SRC)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


_ensure_ratescore_importable()

# Silence PyRuSH/medspacy DEBUG chatter (loguru-backed).
try:
    from loguru import logger as _loguru_logger
    _loguru_logger.remove()
    _loguru_logger.add(sys.stderr, level="WARNING")
except Exception:
    pass
logging.getLogger("PyRuSH").setLevel(logging.WARNING)
logging.getLogger("medspacy").setLevel(logging.WARNING)
# Heavy import (medspacy + transformers; first load ~2 minutes).
from RaTEScore import RaTEScore as _RaTEScore  # noqa: E402


def _default_ner() -> str | None:
    env = os.environ.get("EVALKIT_RATE_NER_PATH")
    if env:
        return env
    bundled = os.path.join(_WEIGHTS_DIR, "RaTE-NER-Deberta")
    return bundled if os.path.isdir(bundled) else None


def _default_biolord() -> str | None:
    env = os.environ.get("EVALKIT_BIOLORD_PATH")
    if env:
        return env
    bundled = os.path.join(_WEIGHTS_DIR, "BioLORD-2023-C")
    return bundled if os.path.isdir(bundled) else None


def _coerce_ref(ref) -> str:
    if isinstance(ref, (list, tuple)):
        return ref[0] if ref else ""
    return ref if isinstance(ref, str) else ("" if ref is None else str(ref))


class RaTEScoreScorer:
    """eval-kit wrapper around RaTEScore (rrg-metric semantics).

    Parameters
    ----------
    ner_model_path : str, optional
        Local path to Angelakeke/RaTE-NER-Deberta. Defaults to
        EVALKIT_RATE_NER_PATH, else ``eval-kit/weights/RaTE-NER-Deberta``.
    eval_model_path : str, optional
        Local path to FremyCompany/BioLORD-2023-C. Defaults to
        EVALKIT_BIOLORD_PATH, else ``eval-kit/weights/BioLORD-2023-C``.
    batch_size : int, default 1
    use_gpu : bool, optional
        Defaults to True iff CUDA is available.
    affinity_matrix : str, default "long"
        "long" or "short" (built-in matrices); rrg-metric default.
    """

    def __init__(
        self,
        ner_model_path: str | None = None,
        eval_model_path: str | None = None,
        batch_size: int = 1,
        use_gpu: bool | None = None,
        affinity_matrix: str = "long",
    ):
        import torch

        ner = ner_model_path or _default_ner()
        biolord = eval_model_path or _default_biolord()
        if ner is None:
            raise FileNotFoundError(
                "RaTE-NER model not found under eval-kit/weights/. Download "
                "Angelakeke/RaTE-NER-Deberta and set EVALKIT_RATE_NER_PATH "
                "or pass ner_model_path=."
            )
        if biolord is None:
            raise FileNotFoundError(
                "BioLORD model not found under eval-kit/weights/. Download "
                "FremyCompany/BioLORD-2023-C and set EVALKIT_BIOLORD_PATH "
                "or pass eval_model_path=."
            )
        if use_gpu is None:
            use_gpu = torch.cuda.is_available()

        logger.info(
            "Loading RaTEScore (ner=%s, eval=%s, gpu=%s, matrix=%s)",
            ner, biolord, use_gpu, affinity_matrix,
        )
        self._rs = _RaTEScore(
            bert_model=ner,
            eval_model=biolord,
            batch_size=batch_size,
            use_gpu=use_gpu,
            affinity_matrix=affinity_matrix,
        )

    def _compute(
        self, refs: Sequence[str], hypos: Sequence[str]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        if len(refs) != len(hypos):
            raise ValueError(
                f"refs/hypos length mismatch: {len(refs)} != {len(hypos)}"
            )
        n = len(refs)
        if n == 0:
            return {}, []
        # rrg-metric: compute_score(preds, gts) -> candidates first.
        scores = self._rs.compute_score(list(hypos), list(refs))
        per_item = [{"ratescore": float(v)} for v in scores]
        corpus = {"ratescore": float(np.mean(scores))} if len(scores) else {"ratescore": 0.0}
        return corpus, per_item

    def compute(
        self, gts: Mapping[int, List[str]], res: Mapping[int, List[str]]
    ) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        keys = sorted(gts.keys())
        refs = [_coerce_ref(gts[k]) for k in keys]
        hypos = [res[k][0] for k in keys]
        return self._compute(refs, hypos)

    def score_single(self, ref, hypo: str) -> Dict[str, float]:
        ref0 = _coerce_ref(ref)
        corpus, per = self._compute([ref0], [hypo])
        out: Dict[str, float] = dict(per[0])
        out.update(corpus)
        return out

    def score_batch(
        self, refs: Sequence, hypos: Sequence[str]
    ) -> List[Dict[str, float]]:
        flat = [_coerce_ref(r) for r in refs]
        _, per = self._compute(flat, list(hypos))
        return per

    def score_corpus(
        self, refs: Sequence, hypos: Sequence[str]
    ) -> Dict[str, float]:
        flat = [_coerce_ref(r) for r in refs]
        corpus, _ = self._compute(flat, list(hypos))
        return corpus


__all__ = ["RaTEScoreScorer"]
