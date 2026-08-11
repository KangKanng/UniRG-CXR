#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify eval-kit reproduces R2GenGPT/evalcap numbers and exercises the API/CLI.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))   # .../eval-kit/tests
ROOT = os.path.dirname(HERE)                          # .../eval-kit
R2GEN = os.environ.get("R2GEN_ROOT")

sys.path.insert(0, ROOT)

from evalkit import Evaluator, Bleu, Cider, Rouge


def _coerce_refs(refs):
    """Accept str | list[str] -> list[str]."""
    if isinstance(refs, str):
        return [refs]
    return list(refs)


def to_coco(refs_list, hypos_list):
    gts = {i: _coerce_refs(r) for i, r in enumerate(refs_list)}
    res = {i: [h] for i, h in enumerate(hypos_list)}
    return gts, res


def main():
    refs = [
        "the heart is normal in size and contour",
        "the lungs are clear no focal consolidation",
        "no pneumothorax or pleural effusion",
    ]
    hypos = [
        "heart size is normal and contour is unremarkable",
        "lung fields are clear without consolidation",
        "no pneumothorax pleural effusion identified",
    ]

    # ---- eval-kit API ----
    ev = Evaluator(metrics=["bleu", "cider", "rouge"])
    single = ev.score_single(refs[0], hypos[0])
    batch = ev.score_batch(refs, hypos)
    corpus = ev.score_corpus(refs, hypos)
    print("=== eval-kit single ===")
    print(json.dumps(single, indent=2))
    print("=== eval-kit corpus ===")
    print(json.dumps(corpus, indent=2))
    assert len(batch) == len(refs)
    assert set(batch[0].keys()) == {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "CIDEr", "ROUGE_L"}

    # ---- optional comparison against an external R2GenGPT checkout ----
    expected = corpus
    if R2GEN:
        sys.path.insert(0, R2GEN)
        from evalcap.bleu.bleu import Bleu as OrigBleu
        from evalcap.cider.cider import Cider as OrigCider
        from evalcap.rouge.rouge import Rouge as OrigRouge
        gts, res = to_coco(refs, hypos)

        b_score, _ = OrigBleu(4).compute_score(gts, res)
        c_score, _ = OrigCider().compute_score(gts, res)
        r_score, _ = OrigRouge().compute_score(gts, res)
        expected = {
            "Bleu_1": b_score[0], "Bleu_2": b_score[1],
            "Bleu_3": b_score[2], "Bleu_4": b_score[3],
            "CIDEr": c_score, "ROUGE_L": r_score,
        }
        for key, value in expected.items():
            assert abs(corpus[key] - float(value)) <= 1e-9
        print("\n[pass] corpus scores match R2GenGPT/evalcap exactly")
    else:
        print("\n[skip] set R2GEN_ROOT to compare against an R2GenGPT checkout")

    # ---- CLI: single literal ----
    out = subprocess.run(
        [sys.executable, "-m", "evalkit", "-m", "cider",
         "--ref", "heart is normal", "--hypo", "heart is normal"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    cli = json.loads(out.stdout)
    assert "CIDEr" in cli
    print("\n[pass] CLI single:", cli)

    # ---- CLI: batch from files ----
    ref_path = os.path.join(HERE, "_tmp_refs.txt")
    hypo_path = os.path.join(HERE, "_tmp_hypos.txt")
    with open(ref_path, "w") as f:
        f.write("\n".join(refs) + "\n")
    with open(hypo_path, "w") as f:
        f.write("\n".join(hypos) + "\n")
    out = subprocess.run(
        [sys.executable, "-m", "evalkit", "-m", "bleu", "rouge",
         "--ref-file", ref_path, "--hypo-file", hypo_path, "--mode", "corpus"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    cli = json.loads(out.stdout)
    assert set(cli.keys()) == {"Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4", "ROUGE_L"}
    print("[pass] CLI batch corpus:", cli)
    os.remove(ref_path)
    os.remove(hypo_path)

    # ---- CLI: prediction JSONL (labels=reference, response=hypothesis) ----
    pred_path = os.path.join(HERE, "_tmp_predictions.jsonl")
    with open(pred_path, "w") as f:
        for ref, hypo in zip(refs, hypos):
            f.write(json.dumps({"labels": ref, "response": hypo}) + "\n")
    out = subprocess.run(
        [sys.executable, "-m", "evalkit", "-m", "bleu", "cider", "rouge",
         "--pred-file", pred_path, "--mode", "corpus"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    )
    cli = json.loads(out.stdout)
    for key, value in expected.items():
        assert abs(cli[key] - float(value)) <= 1e-9
    print("[pass] CLI prediction JSONL:", cli)
    os.remove(pred_path)

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
