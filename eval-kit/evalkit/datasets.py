#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset loaders for eval-kit.

Reads report-generation manifests and exposes a uniform list of
``{"ref": <str>, "images": [<str>], ...}`` items so the Evaluator can score
them against a hypothesis source (file or another dataset column).

Supported manifests
-------------------
- ``rexrank``: ``<root>/<split>.jsonl`` with fields ``query``/``response``/
  ``images`` (IU-Xray / R2Gen split under uni-rg-cxr/data/rexrank).
- ``artifacts`` (a.k.a. ``iu``): ``<root>/iu_<split>.jsonl`` with fields
  ``id``/``prompt``/``answer``/``images`` (uni-rg-cxr/artifacts).

Roots are resolved from CLI ``--ref-root``/``--hypo-root`` or the env vars
``EVALKIT_REXRANK_ROOT`` / ``EVALKIT_ARTIFACTS_ROOT``; defaults point at this
repository's data and artifacts directories.

Public API
----------
>>> from evalkit.datasets import load_dataset
>>> items = load_dataset("artifacts", "test")
>>> refs = [it["ref"] for it in items]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# Default roots (override via env vars or CLI --ref-root / --hypo-root).
_UNI_RG = str(Path(__file__).resolve().parents[2])
_DEFAULT_REXRANK_ROOT = os.environ.get("EVALKIT_REXRANK_ROOT",
                                       os.path.join(_UNI_RG, "data", "rexrank"))
_DEFAULT_ARTIFACTS_ROOT = os.environ.get("EVALKIT_ARTIFACTS_ROOT",
                                          os.path.join(_UNI_RG, "artifacts"))

DATASET_NAMES = ("rexrank", "artifacts")
SPLIT_NAMES = ("train", "valid", "test")


def _read_jsonl(path: str) -> List[dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                out.append(json.loads(ln))
    return out


def _load_rexrank(split: str, root: Optional[str]) -> List[Dict]:
    root = root or _DEFAULT_REXRANK_ROOT
    path = os.path.join(root, f"{split}.jsonl")
    items = []
    for d in _read_jsonl(path):
        items.append({
            "id": d.get("id"),
            "ref": d.get("response", ""),
            "query": d.get("query", ""),
            "images": d.get("images", []) or [],
        })
    return items


def _load_artifacts(split: str, root: Optional[str]) -> List[Dict]:
    root = root or _DEFAULT_ARTIFACTS_ROOT
    fname = f"iu_{split}.jsonl"
    path = os.path.join(root, fname)
    items = []
    for d in _read_jsonl(path):
        items.append({
            "id": d.get("id"),
            "ref": d.get("answer", ""),
            "prompt": d.get("prompt", ""),
            "indication": d.get("indication", ""),
            "images": d.get("images", []) or [],
        })
    return items


_LOADERS = {
    "rexrank": _load_rexrank,
    "artifacts": _load_artifacts,
    "iu": _load_artifacts,  # alias
}


def load_dataset(name: str, split: str, root: Optional[str] = None) -> List[Dict]:
    """Load a manifest as a list of ``{"ref": str, "images": [...], ...}``.

    Parameters
    ----------
    name : str
        One of ``DATASET_NAMES`` (``rexrank``, ``artifacts``/``iu``).
    split : str
        One of ``train`` / ``valid`` / ``test``.
    root : str, optional
        Manifest root dir; defaults to the dataset's standard location.
    """
    name = name.lower()
    split = split.lower()
    if name not in _LOADERS:
        raise ValueError(f"unknown dataset '{name}'. available: {DATASET_NAMES}")
    if split not in SPLIT_NAMES:
        raise ValueError(f"unknown split '{split}'. available: {SPLIT_NAMES}")
    items = _LOADERS[name](split, root)
    if not items:
        raise ValueError(f"dataset '{name}' split '{split}' loaded empty from {root}")
    return items


def refs_from(items: List[Dict]) -> List[str]:
    """Extract the reference-text column from loaded items."""
    return [it["ref"] for it in items]


__all__ = ["load_dataset", "refs_from", "DATASET_NAMES", "SPLIT_NAMES"]
