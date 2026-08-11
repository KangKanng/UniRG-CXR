#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""F1 RadGraph metric for eval-kit.

Adapted verbatim from ``rrg-metric/rrg_metric/radgraph_gpu.py``
(StanfordAIMI/RRG_scorers), which subclasses the ``radgraph`` package. This
module reproduces that scorer and wraps it for the eval-kit Evaluator API.

Environment notes (this repo)
-----------------------------
- ``radgraph`` ships as source under ``<repo>/radgraph/`` but, from the repo
  root, resolves to a *namespace* package. We prepend ``<repo>/radgraph`` to
  ``sys.path`` so ``import radgraph`` finds the real package
  (``radgraph/radgraph/__init__.py``). ``radgraph`` is also installed editable
  (``pip install -e ./radgraph --no-deps``) so ``importlib.metadata.version``
  resolves.
- Runtime deps installed into the venv (torch-neutral): ``dotmap``, ``appdirs``,
  ``jsonpickle``, ``filelock``, ``h5py``, ``nltk``.
- Local weights: supplied separately under ``eval-kit/weights/``; ``<model_type>.tar.gz``
  is extracted to ``weights/<model_type>/`` on first use (no network).
- PubMedBERT: the RadGraph embedder needs
  ``BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext``, placed under
  ``eval-kit/weights/`` (override: ``EVALKIT_PUBMEDBERT_PATH`` or
  ``pubmedbert_path=``); the config's embedder/indexer ``model_name`` is
  patched to that path so no network is needed. Offline env vars are forced.

Public API
----------
>>> from evalkit.radgraph_scorer import F1RadGraphScorer
>>> sc = F1RadGraphScorer()
>>> sc.score_single(ref="...", hypo="...")
>>> sc.score_batch(refs=[...], hypos=[...])
>>> sc.score_corpus(refs=[...], hypos=[...])
"""
from __future__ import annotations

import logging
import os
import sys
import json
import tarfile
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Bundled weights: <eval-kit>/weights (evalkit/ -> up two = eval-kit/).
_WEIGHTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights"
)
_RADGRAPH_SRC = os.path.join(_REPO_ROOT, "radgraph")


def _ensure_radgraph_importable() -> None:
    """Make ``import radgraph`` resolve to the real package in this repo."""
    if _RADGRAPH_SRC not in sys.path:
        sys.path.insert(0, _RADGRAPH_SRC)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


def _patch_transformers_tokenizer_compat() -> None:
    """Polyfill ``encode_plus``/``tokenize`` removed in transformers 5.x so
    radgraph's vendored allennlp tokenizer helpers still work offline."""
    try:
        from transformers import PreTrainedTokenizerBase
    except Exception:
        return
    if not hasattr(PreTrainedTokenizerBase, "encode_plus"):
        def _encode_plus(self, text, text_pair=None, add_special_tokens=True,
                          return_token_type_ids=None, return_attention_mask=None,
                          max_length=None, truncation=None, padding=None,
                          return_tensors=None, **kwargs):
            call_kwargs = {"add_special_tokens": add_special_tokens}
            if return_token_type_ids is not None:
                call_kwargs["return_token_type_ids"] = return_token_type_ids
            if return_attention_mask is not None:
                call_kwargs["return_attention_mask"] = return_attention_mask
            if max_length is not None:
                call_kwargs["max_length"] = max_length
            if truncation is not None:
                call_kwargs["truncation"] = truncation
            if padding is not None:
                call_kwargs["padding"] = padding
            if return_tensors is not None:
                call_kwargs["return_tensors"] = return_tensors
            call_kwargs.update(kwargs)
            return self(text, text_pair, **call_kwargs)
        PreTrainedTokenizerBase.encode_plus = _encode_plus
    if not hasattr(PreTrainedTokenizerBase, "build_inputs_with_special_tokens"):
        def _build_inputs(self, token_ids_0, token_ids_1=None):
            cls = self.cls_token_id
            sep = self.sep_token_id
            ids0 = list(token_ids_0)
            if token_ids_1 is None:
                return [cls] + ids0 + [sep]
            return [cls] + ids0 + [sep] + list(token_ids_1) + [sep]
        PreTrainedTokenizerBase.build_inputs_with_special_tokens = _build_inputs


_ensure_radgraph_importable()
_patch_transformers_tokenizer_compat()

# Heavy import (allennlp + transformers ~minutes on first load). Deferred to
# module import time of this scorer, which itself is lazily loaded by Evaluator.
import importlib.metadata  # noqa: E402
from dotmap import DotMap  # noqa: E402
from appdirs import user_cache_dir  # noqa: E402

from radgraph.allennlp.data import Vocabulary  # noqa: E402
from radgraph.allennlp.data.dataset_readers import AllennlpDataset  # noqa: E402
from radgraph.allennlp.data.dataloader import PyTorchDataLoader  # noqa: E402
from radgraph.allennlp.data import token_indexers  # noqa: E402
from radgraph.allennlp.modules import token_embedders, text_field_embedders  # noqa: E402
from radgraph.allennlp.common.params import Params  # noqa: E402
from radgraph.dygie.data.dataset_readers.dygie import DyGIEReader  # noqa: E402
from radgraph.dygie.models import dygie  # noqa: E402
from radgraph.utils import (  # noqa: E402
    download_model,
    preprocess_reports,
    postprocess_reports,
    batch_to_device,
)
from radgraph import RadGraph as RadGraphCpu  # noqa: E402
from radgraph import F1RadGraph as F1RadGraphCpu  # noqa: E402

logging.getLogger("radgraph").setLevel(logging.CRITICAL)
logging.getLogger("allennlp").setLevel(logging.CRITICAL)

MODEL_MAPPING = {
    "radgraph": "radgraph.tar.gz",
    "radgraph-xl": "radgraph-xl.tar.gz",
    "echograph": "echograph.tar.gz",
}

version = importlib.metadata.version("radgraph")
CACHE_DIR = user_cache_dir("radgraph")
CACHE_DIR = os.path.join(CACHE_DIR, version)


def _default_pubmedbert() -> str | None:
    env = os.environ.get("EVALKIT_PUBMEDBERT_PATH")
    if env:
        return env
    bundled = os.path.join(
        _WEIGHTS_DIR, "BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    )
    return bundled if os.path.isdir(bundled) else None

def _try_patch_pubmedbert(config, path: str) -> None:
    """Patch the RadGraph config embedder/indexer model_name to a local path."""
    try:
        config.dataset_reader.token_indexers.bert.model_name = path
    except Exception:
        pass
    try:
        config.model.embedder.token_embedders.bert.model_name = path
    except Exception:
        pass


class RadGraph(RadGraphCpu):
    """RadGraph with local-weight + local-PubMedBERT support.

    Reproduced from rrg-metric/radgraph_gpu.py with two additions:
    - ``weights_root`` overrides ``temp_dir`` so the bundled tarball under
      ``eval-kit/weights/`` is used without network.
    - ``pubmedbert_path`` patches the config embedder/indexer model_name to a
      local PubMedBERT directory, avoiding a hub download.
    """

    def __init__(
        self,
        batch_size=1,
        cuda=0,
        model_type=None,
        temp_dir=None,
        weights_root: str | None = None,
        pubmedbert_path: str | None = None,
        **kwargs,
    ):
        nn.Module.__init__(self)

        if cuda is None:
            cuda = -1
        if cuda >= 0 and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{cuda}")
        else:
            self.device = torch.device("cpu")

        self.cuda = cuda
        self.batch_size = batch_size

        if model_type is None:
            model_type = "radgraph"
        self.model_type = model_type.lower()
        assert self.model_type in ["radgraph", "radgraph-xl", "echograph"]

        pubmedbert_path = pubmedbert_path or _default_pubmedbert()

        # Resolve weights root: prefer explicit, then the bundled
        # eval-kit/weights dir, then the original cache dir behaviour.
        if temp_dir is None:
            temp_dir = weights_root or _WEIGHTS_DIR
        model_dir = os.path.join(temp_dir, self.model_type)

        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            local_tar = os.path.join(_WEIGHTS_DIR, MODEL_MAPPING[self.model_type])
            if os.path.exists(local_tar):
                archive_path = local_tar
            else:
                archive_path = download_model(
                    repo_id="StanfordAIMI/RRG_scorers",
                    cache_dir=temp_dir,
                    filename=MODEL_MAPPING[self.model_type],
                )
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=model_dir)

        # Read config.
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path) as f:
            config = DotMap(json.load(f))

        # Patch PubMedBERT model_name to a local path (no network).
        if pubmedbert_path:
            _try_patch_pubmedbert(config, pubmedbert_path)

        # Vocab
        vocab_dir = os.path.join(model_dir, "vocabulary")
        vocab_params = config.get("vocabulary", Params({}))
        vocab = Vocabulary.from_files(
            vocab_dir,
            vocab_params.get("padding_token"),
            vocab_params.get("oov_token"),
        )

        # Tokenizer
        tok_indexers = {
            "bert": token_indexers.PretrainedTransformerMismatchedIndexer(
                model_name=config.dataset_reader.token_indexers.bert.model_name,
                max_length=config.dataset_reader.token_indexers.bert.max_length,
            )
        }
        self.reader = DyGIEReader(
            max_span_width=config.dataset_reader.max_span_width,
            token_indexers=tok_indexers,
        )

        # Embedder
        token_embedder = token_embedders.PretrainedTransformerMismatchedEmbedder(
            model_name=config.model.embedder.token_embedders.bert.model_name,
            max_length=config.model.embedder.token_embedders.bert.max_length,
        )
        embedder = text_field_embedders.BasicTextFieldEmbedder({"bert": token_embedder})

        # Model
        model_dict = config.model
        for name in ["type", "embedder", "initializer", "module_initializer"]:
            del model_dict[name]
        model = dygie.DyGIE(vocab=vocab, embedder=embedder, **model_dict)
        model_state_path = os.path.join(model_dir, "weights.th")
        model_state = torch.load(
            model_state_path, map_location=self.device, weights_only=True
        )
        model.load_state_dict(model_state, strict=True)
        model.eval()
        self.model = model.to(self.device)


class F1RadGraph(F1RadGraphCpu):
    """F1RadGraph reproduced from rrg-metric/radgraph_gpu.py.

    Delegates annotation/reward computation to the parent (F1RadGraphCpu), but
    builds a local-weights RadGraph (see above) instead of the upstream one.
    """

    def __init__(self, reward_level, model_type=None, cuda=0, **kwargs):
        nn.Module.__init__(self)
        assert reward_level in ["simple", "partial", "complete", "all"]
        self.reward_level = reward_level
        self.radgraph = RadGraph(model_type=model_type, cuda=cuda, **kwargs)


def _coerce_ref(ref) -> str:
    if isinstance(ref, (list, tuple)):
        return ref[0] if ref else ""
    return ref


class F1RadGraphScorer:
    """eval-kit wrapper around F1RadGraph (rrg-metric semantics).

    Parameters
    ----------
    model_type : str, default "radgraph"
        One of "radgraph", "radgraph-xl", "echograph".
    reward_level : str, default "complete"
        One of "simple", "partial", "complete", "all". "all" yields a
        (precision, recall, f1) triple; the others yield a single float.
    cuda : int, default 0
        Device index; -1 forces CPU.
    weights_root : str, optional
        Root containing ``<model_type>/`` extracted weights; defaults to
        ``eval-kit/weights/``.
    pubmedbert_path : str, optional
        Local PubMedBERT directory; defaults to ``EVALKIT_PUBMEDBERT_PATH``,
        else the bundled ``eval-kit/weights/BiomedNLP-...-fulltext``.
    """

    def __init__(
        self,
        model_type: str = "radgraph",
        reward_level: str = "complete",
        cuda: int = 0,
        weights_root: str | None = None,
        pubmedbert_path: str | None = None,
    ):
        logger.info("Loading F1RadGraph (model=%s, reward=%s)",
                    model_type, reward_level)
        self.reward_level = reward_level
        self._f1 = F1RadGraph(
            reward_level=reward_level,
            model_type=model_type,
            cuda=cuda,
            weights_root=weights_root,
            pubmedbert_path=pubmedbert_path,
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
        mean, reward_list, _pred, _gt = self._f1(
            refs=list(refs), hyps=list(hypos)
        )
        if self.reward_level == "all":
            p_list, r_list, f_list = reward_list
            corpus = {
                "f1radgraph_precision": float(np.mean(p_list)),
                "f1radgraph_recall": float(np.mean(r_list)),
                "f1radgraph_f1": float(np.mean(f_list)),
            }
            per_item = [
                {
                    "f1radgraph_precision": float(p_list[i]),
                    "f1radgraph_recall": float(r_list[i]),
                    "f1radgraph_f1": float(f_list[i]),
                }
                for i in range(n)
            ]
        else:
            corpus = {"f1radgraph": float(mean)}
            per_item = [{"f1radgraph": float(v)} for v in reward_list]
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


__all__ = ["RadGraph", "F1RadGraph", "F1RadGraphScorer"]
