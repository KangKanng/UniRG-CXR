#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""eval-kit: lightweight, self-contained NLG metrics for chest X-ray reports.

Adapted from R2GenGPT/evalcap. Metrics ported: BLEU(1-4), CIDEr, ROUGE-L.
METEOR is NOT ported here because it requires a Java jar; use it separately.

Public API
----------
>>> from evalkit import Evaluator
>>> ev = Evaluator(metrics=["bleu", "cider", "rouge"])
>>> ev.score_single(ref="heart is normal", hypo="heart is normal")
{'Bleu_1': 1.0, 'Bleu_2': 1.0, 'Bleu_3': 1.0, 'Bleu_4': 1.0,
 'CIDEr': 7.5, 'ROUGE_L': 1.0}
>>> ev.score_batch(refs=["heart is normal", "lungs are clear"],
...                hypos=["heart is normal", "lungs are clear"])
[{'Bleu_1': 1.0, ...}, ...]

The evaluator also supports multiple references per hypothesis (a list of
reference strings), both in single and batch mode. The batch interface returns
per-item scores; aggregate corpus means are available via ``score_corpus``.

CLI
---
python -m evalkit --metrics bleu cider rouge \
    --ref-file a.txt --hypo-file b.txt            # batch from two aligned files
python -m evalkit --metrics cider \
    --ref "heart is normal" --hypo "heart is normal"   # single pair
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Dict, List, Mapping, Sequence, Tuple, Union

from .bleu import Bleu
from .cider import Cider
from .rouge import Rouge

__all__ = ["Evaluator", "METRIC_NAMES", "get_metric"]

# Registry of available metrics.
# - "bleu"/"cider"/"rouge": pure-Python NLG scorers (R2GenGPT/evalcap).
# - "chexbert": model-based; yields F1 CheXbert + SembScore (and additional
#   keys). Lazily loaded from local weights (see evalkit.chexbert).
# - "bertscore": model-based; precision/recall/F1 over contextual embeddings
#   (see evalkit.bertscore).
# - "f1radgraph": model-based; F1 RadGraph reward over entity graphs
#   (see evalkit.radgraph_scorer). Requires local PubMedBERT.
# - "ratescore": model-based; entity-aware RaTEScore (see evalkit.ratescore).
#   Requires local RaTE-NER + BioLORD models.
# - "chexprompt": remote GPT-based factual error counts and paper reward.
# METEOR is intentionally absent (java jar dependency).
METRIC_NAMES = ("bleu", "cider", "rouge", "chexbert", "bertscore", "f1radgraph", "ratescore", "chexprompt")

# Model metrics are instantiated lazily (they load torch weights); their
# output keys are declared up-front so the Evaluator API stays uniform.
_MODEL_METRICS = {
    "chexbert": ("f1chexbert", "sembscore"),
    "bertscore": ("bertscore_precision", "bertscore_recall", "bertscore_f1"),
    "f1radgraph": ("f1radgraph",),
    "ratescore": ("ratescore",),
    "chexprompt": (
        "chexprompt_errors",
        "chexprompt_significant_errors",
        "chexprompt_insignificant_errors",
        "chexprompt_reward",
    ),
}


def _as_ref_list(ref: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(ref, str):
        return [ref]
    return list(ref)


def get_metric(name: str):
    """Return (scorer, method_names) for a pure-Python NLG metric.

    Model metrics ("chexbert") are NOT instantiated here; they are reserved
    via ``_MODEL_METRICS`` and lazily built by ``Evaluator._get_model``.
    """
    name = name.lower()
    if name == "bleu":
        return Bleu(n=4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]
    if name == "cider":
        return Cider(n=4, sigma=6.0), ["CIDEr"]
    if name == "rouge":
        return Rouge(beta=1.2), ["ROUGE_L"]
    if name in _MODEL_METRICS:
        return None, list(_MODEL_METRICS[name])
    raise ValueError(
        f"unknown metric '{name}'. available: {METRIC_NAMES} "
        "(meteor requires java and is not bundled)"
    )


class Evaluator:
    """Select one or several metrics and score single or batch inputs.

    Pure metrics (bleu/cider/rouge) are instantiated eagerly. Model metrics
    (chexbert) are loaded lazily on first use because they pull torch weights.
    """

    def __init__(self, metrics: Union[str, Sequence[str]] = METRIC_NAMES):
        if isinstance(metrics, str):
            metrics = [metrics]
        if len(metrics) == 0:
            raise ValueError("at least one metric must be selected")
        # unknown names raise via get_metric
        for m in metrics:
            get_metric(m)
        self._metric_names = [m.lower() for m in metrics]

        # pure-Python scorers: (scorer, methods)
        self._scorers: List[Tuple[object, List[str]]] = []
        # model metrics: (name, methods)
        self._model_specs: List[Tuple[str, List[str]]] = []
        for m in self._metric_names:
            if m in _MODEL_METRICS:
                self._model_specs.append((m, list(_MODEL_METRICS[m])))
            else:
                self._scorers.append(get_metric(m))
        self._model_instances: Dict[str, object] = {}

    @property
    def metrics(self) -> List[str]:
        return list(self._metric_names)

    def _get_model(self, name: str):
        if name not in self._model_instances:
            if name == "chexbert":
                from .chexbert import CheXbert
                self._model_instances[name] = CheXbert()
            elif name == "bertscore":
                from .bertscore import BertScore
                self._model_instances[name] = BertScore()
            elif name == "f1radgraph":
                from .radgraph_scorer import F1RadGraphScorer
                self._model_instances[name] = F1RadGraphScorer()
            elif name == "ratescore":
                from .ratescore import RaTEScoreScorer
                self._model_instances[name] = RaTEScoreScorer()
            elif name == "chexprompt":
                from .chexprompt import CheXpromptScorer
                self._model_instances[name] = CheXpromptScorer()
            else:
                raise ValueError(f"unknown model metric: {name}")
        return self._model_instances[name]

    # ------------------------------------------------------------------
    # Single pair
    # ------------------------------------------------------------------
    def score_single(self, ref: Union[str, Sequence[str]], hypo: str) -> Dict[str, float]:
        refs = _as_ref_list(ref)
        if len(refs) == 0:
            raise ValueError("at least one reference required")
        if not isinstance(hypo, str):
            raise TypeError("hypo must be a str")

        gts = {0: refs}
        res = {0: [hypo]}
        out: Dict[str, float] = self._run(gts, res)
        # model metrics: single-pair per-item + corpus additional keys
        for name, methods in self._model_specs:
            scorer = self._get_model(name)
            corpus, per = scorer.compute(gts, res)
            out.update(per[0])
            # also surface corpus-level additional keys (accuracy, P/R/F1, ...)
            for k, v in corpus.items():
                if k not in out:
                    out[k] = float(v)
        return out

    # ------------------------------------------------------------------
    # Batch (per-item)
    # ------------------------------------------------------------------
    def score_batch(
        self,
        refs: Sequence[Union[str, Sequence[str]]],
        hypos: Sequence[str],
    ) -> List[Dict[str, float]]:
        if len(refs) != len(hypos):
            raise ValueError(
                f"refs/hypos length mismatch: {len(refs)} != {len(hypos)}"
            )
        n = len(refs)
        if n == 0:
            return []
        gts: Dict[int, List[str]] = {}
        res: Dict[int, List[str]] = {}
        for i, (r, h) in enumerate(zip(refs, hypos)):
            gts[i] = _as_ref_list(r)
            res[i] = [h]
        per_item = self._run_batch(gts, res)
        # model metrics: per-item primary keys only
        for name, methods in self._model_specs:
            scorer = self._get_model(name)
            _corpus, per = scorer.compute(gts, res)
            for i, item in enumerate(per):
                per_item[i].update({k: float(v) for k, v in item.items()})
        return per_item

    # ------------------------------------------------------------------
    # Corpus (aggregate)
    # ------------------------------------------------------------------
    def score_corpus(
        self,
        refs: Sequence[Union[str, Sequence[str]]],
        hypos: Sequence[str],
    ) -> Dict[str, float]:
        """Aggregate corpus-level scores (one value per selected metric).

        BLEU/CIDEr/ROUGE-L compute a corpus score from the whole set, rather
        than averaging per-item scores, matching the MS-COCO convention.
        Model metrics (chexbert) return corpus aggregates (micro-F1, mean
        SembScore, accuracy, micro/macro P/R/F1).
        """
        if len(refs) != len(hypos):
            raise ValueError(
                f"refs/hypos length mismatch: {len(refs)} != {len(hypos)}"
            )
        n = len(refs)
        if n == 0:
            raise ValueError("empty corpus")
        gts: Dict[int, List[str]] = {}
        res: Dict[int, List[str]] = {}
        for i, (r, h) in enumerate(zip(refs, hypos)):
            gts[i] = _as_ref_list(r)
            res[i] = [h]

        out: Dict[str, float] = {}
        for scorer, methods in self._scorers:
            score, _scores = scorer.compute_score(gts, res)
            if isinstance(score, (list, tuple)):
                for m, s in zip(methods, score):
                    out[m] = float(s)
            else:
                out[methods[0]] = float(score)
        for name, methods in self._model_specs:
            scorer = self._get_model(name)
            corpus, _per = scorer.compute(gts, res)
            out.update({k: float(v) for k, v in corpus.items()})
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run(self, gts: Mapping[int, List[str]], res: Mapping[int, List[str]]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for scorer, methods in self._scorers:
            score, _scores = scorer.compute_score(gts, res)
            if isinstance(score, (list, tuple)):
                for m, s in zip(methods, score):
                    out[m] = float(s)
            else:
                out[methods[0]] = float(score)
        return out

    def _run_batch(
        self, gts: Mapping[int, List[str]], res: Mapping[int, List[str]]
    ) -> List[Dict[str, float]]:
        n = len(gts)
        per_item: List[Dict[str, float]] = [{} for _ in range(n)]
        for scorer, methods in self._scorers:
            score, scores = scorer.compute_score(gts, res)
            if isinstance(score, (list, tuple)):
                # BLEU: corpus means in `score`, per-item in `scores`
                # `scores` shape: (ngram, n_items)
                for mi, m in enumerate(methods):
                    item_scores = scores[mi] if len(scores) > mi else None
                    if item_scores is None:
                        continue
                    for i, s in enumerate(item_scores):
                        per_item[i][m] = float(s)
            else:
                for i, s in enumerate(scores):
                    per_item[i][methods[0]] = float(s)
        return per_item


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [ln.rstrip("\n") for ln in f]


def _read_refs_file(path: str) -> List[List[str]]:
    """Read references; one JSON list of strings per line OR one ref per line."""
    out: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                out.append([])
                continue
            if ln.startswith("["):
                try:
                    out.append(json.loads(ln))
                    continue
                except json.JSONDecodeError:
                    pass
            out.append([ln])
    return out


def _read_prediction_jsonl(
    path: str, ref_key: str = "labels", hypo_key: str = "response"
) -> Tuple[List[str], List[str]]:
    """Read references and hypotheses from a prediction JSONL file."""
    refs: List[str] = []
    hypos: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{lineno}: expected a JSON object")
            missing = [key for key in (ref_key, hypo_key) if key not in item]
            if missing:
                raise ValueError(
                    f"{path}:{lineno}: missing field(s): {', '.join(missing)}"
                )
            ref = item[ref_key]
            hypo = item[hypo_key]
            if not isinstance(ref, str) or not isinstance(hypo, str):
                raise ValueError(
                    f"{path}:{lineno}: fields '{ref_key}' and '{hypo_key}' "
                    "must both be strings"
                )
            refs.append(ref)
            hypos.append(hypo)
    if not refs:
        raise ValueError(f"prediction file is empty: {path}")
    return refs, hypos


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="evalkit",
        description="Self-contained BLEU/CIDEr/ROUGE-L evaluator (R2GenGPT metrics).",
    )
    p.add_argument(
        "--metrics", "-m",
        nargs="+",
        default=list(METRIC_NAMES),
        choices=METRIC_NAMES,
        help="metrics to compute (default: all)",
    )
    p.add_argument("--ref", help="reference: a literal string")
    p.add_argument("--hypo", help="hypothesis: a literal string")
    p.add_argument(
        "--mode", choices=["single", "batch", "corpus"], default="auto",
        help="single = one pair; batch = per-item; corpus = aggregate mean",
    )
    p.add_argument(
        "--ref-file", help="references file (one ref per line, or JSON list per line for multi-ref)",
    )
    p.add_argument(
        "--hypo-file", help="hypotheses file (one hypo per line)",
    )
    p.add_argument(
        "--pred-file",
        help="prediction JSONL containing both references and hypotheses",
    )
    p.add_argument(
        "--ref-key", default="labels",
        help="reference field used with --pred-file (default: labels)",
    )
    p.add_argument(
        "--hypo-key", default="response",
        help="hypothesis field used with --pred-file (default: response)",
    )
    p.add_argument(
        "--ref-dataset",
        help="dataset name for references (rexrank|artifacts|iu); reads the "
             "reference column (rexrank.response / artifacts.answer)",
    )
    p.add_argument("--ref-split", default="test",
                   help="dataset split for --ref-dataset (train|valid|test)")
    p.add_argument("--ref-root", help="manifest root dir for --ref-dataset")
    p.add_argument(
        "--hypo-dataset",
        help="dataset name for hypotheses (same column as refs); use to "
             "self-evaluate or when hypos come from another manifest column",
    )
    p.add_argument("--hypo-split", help="dataset split for --hypo-dataset (default: --ref-split)")
    p.add_argument("--hypo-root", help="manifest root dir for --hypo-dataset")
    p.add_argument("--limit", type=int, help="use only the first N items (debug/smoke)")
    p.add_argument(
        "--multi-ref", action="store_true",
        help="ref file has multiple references per line, JSON-encoded as a list",
    )
    p.add_argument("--out", help="write JSON results to this path (default: stdout)")
    args = p.parse_args(argv)

    ev = Evaluator(metrics=args.metrics)

    # A prediction JSONL supplies both sides and is exclusive with all other
    # reference/hypothesis sources.
    if args.pred_file is not None:
        if any((args.ref is not None, args.ref_file is not None,
                args.ref_dataset is not None, args.hypo is not None,
                args.hypo_file is not None, args.hypo_dataset is not None)):
            p.error("--pred-file cannot be combined with other reference/hypothesis sources")
        try:
            refs, hypos = _read_prediction_jsonl(
                args.pred_file, args.ref_key, args.hypo_key
            )
        except (OSError, ValueError) as exc:
            p.error(str(exc))
        ref_source = "prediction JSONL"

    # --- resolve references ---
    elif args.ref is not None:
        refs = [args.ref]
        ref_source = "literal"
    elif args.ref_file is not None:
        refs = (_read_refs_file(args.ref_file) if args.multi_ref
                else _read_lines(args.ref_file))
        ref_source = "file"
    elif args.ref_dataset is not None:
        from .datasets import load_dataset, refs_from
        refs = refs_from(load_dataset(args.ref_dataset, args.ref_split, args.ref_root))
        ref_source = "dataset"
    else:
        p.error("provide --ref, --ref-file, or --ref-dataset")

    # --- resolve hypotheses ---
    if args.pred_file is not None:
        pass  # resolved together above
    elif args.hypo is not None:
        hypos = [args.hypo]
    elif args.hypo_file is not None:
        hypos = _read_lines(args.hypo_file)
    elif args.hypo_dataset is not None:
        from .datasets import load_dataset, refs_from
        hsplit = args.hypo_split or args.ref_split
        hypos = refs_from(load_dataset(args.hypo_dataset, hsplit, args.hypo_root))
    else:
        p.error("provide --hypo, --hypo-file, or --hypo-dataset")

    # exclusive sources: literals vs file vs dataset
    n_ref_src = sum([args.ref is not None, args.ref_file is not None,
                     args.ref_dataset is not None, args.pred_file is not None])
    n_hyp_src = sum([args.hypo is not None, args.hypo_file is not None,
                     args.hypo_dataset is not None, args.pred_file is not None])
    if n_ref_src != 1:
        p.error("provide exactly one reference source: --ref / --ref-file / --ref-dataset")
    if n_hyp_src != 1:
        p.error("provide exactly one hypothesis source: --hypo / --hypo-file / --hypo-dataset")

    if args.limit:
        refs = refs[:args.limit]
        hypos = hypos[:args.limit]
    if len(refs) != len(hypos):
        p.error(f"count mismatch: ref={len(refs)} hypo={len(hypos)}")

    # single = both literals given, or exactly one item and mode auto
    is_literal_pair = (args.ref is not None and args.hypo is not None)
    mode = args.mode
    if mode == "auto":
        mode = "single" if (is_literal_pair or len(refs) == 1) else "batch"
    if mode == "single":
        if len(refs) != 1:
            p.error("--mode single requires exactly one item")
        res = ev.score_single(refs[0], hypos[0])
    elif mode == "corpus":
        res = ev.score_corpus(refs, hypos)
    else:
        res = ev.score_batch(refs, hypos)

    text = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
