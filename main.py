"""Small, reproducible UniRG-CXR data/evaluation pipeline.

This does not pretend to train the paper's 8B model on a CPU-only machine. It
prepares the report-generation task for subsequent SFT/GRPO training.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


PROMPT_BOTH = (
    "This is a radiology report generation task. Here is the context: {context} "
    "Given the image and the context, provide the report in the following format: "
    "Findings: [write the findings] Impression: [write the impression] "
    "Now write the report in the format above."
)
PROMPT_FINDINGS = (
    "This is a radiology report generation task. Here is the context: {context} "
    "Given the image and the context, provide the findings in the following format: "
    "Findings: [write the findings] Now write the report in the format above."
)


def clean(value: str | None) -> str:
    return (value or "").strip()


def context(indication: str, comparison: str) -> str:
    fields = []
    if clean(indication):
        fields.append(f"Indication: {clean(indication)}")
    if clean(comparison):
        fields.append(f"Comparison: {clean(comparison)}")
    return " ".join(fields) if fields else "No additional clinical context."


def target(findings: str, impression: str) -> str:
    chunks = []
    if clean(findings):
        chunks.append(f"Findings: {clean(findings)}")
    if clean(impression):
        chunks.append(f"Impression: {clean(impression)}")
    return " ".join(chunks)


def stable_split(uid: str, train_pct: int = 80, valid_pct: int = 10) -> str:
    # Stable across Python versions and machines, unlike hash().
    bucket = int(hashlib.sha256(uid.encode()).hexdigest()[:8], 16) % 100
    if bucket < train_pct:
        return "train"
    if bucket < train_pct + valid_pct:
        return "valid"
    return "test"


def load_r2gen_splits(split_dir: Path) -> dict[str, str]:
    """Load the R2Gen study split used by ReXrank.

    R2Gen paths have the form ``.../CXR3030_IM-1405/0.png``; the numeric
    component is the UID used by the original Open-I CSV files.
    """
    split_by_uid: dict[str, str] = {}
    for split in ("train", "valid", "test"):
        path = split_dir / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ReXrank/R2Gen split file: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                match = re.search(r"CXR(\d+)_", item["images"][0])
                if not match:
                    raise ValueError(f"Cannot extract IU UID from {item['images'][0]!r}")
                uid = match.group(1)
                if uid in split_by_uid:
                    raise ValueError(f"Duplicate IU UID across R2Gen splits: {uid}")
                split_by_uid[uid] = split
    return split_by_uid


def load_iu(data_dir: Path, split_dir: Path) -> list[dict]:
    projection_path = data_dir / "indiana_projections.csv"
    report_path = data_dir / "indiana_reports.csv"
    frontal: dict[str, str] = {}
    any_projection: dict[str, str] = {}
    with projection_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            any_projection.setdefault(row["uid"], row["filename"])
            if row["projection"].lower() == "frontal" and row["uid"] not in frontal:
                frontal[row["uid"]] = row["filename"]

    split_by_uid = load_r2gen_splits(split_dir)
    records = []
    with report_path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            uid = row["uid"]
            if uid not in split_by_uid:
                continue
            findings, impression = clean(row["findings"]), clean(row["impression"])
            # Eight R2Gen studies have both local views mislabeled as Lateral.
            # Keep the official split complete by falling back to the first
            # available view; all 590 ReXrank test studies have a frontal label.
            filename = frontal.get(uid) or any_projection.get(uid)
            if not filename or not (findings or impression):
                continue
            image = data_dir / "images" / "images_normalized" / filename
            if not image.is_file():
                continue
            ctx = context(row["indication"], row["comparison"])
            prompt = (PROMPT_BOTH if impression else PROMPT_FINDINGS).format(context=ctx)
            records.append(
                {
                    "id": uid,
                    "dataset": "iu-xray",
                    "split": split_by_uid[uid],
                    "image": str(image.resolve()),
                    "indication": clean(row["indication"]),
                    "comparison": clean(row["comparison"]),
                    "prompt": prompt,
                    "answer": target(findings, impression),
                }
            )
    return records


def load_rexgradient(data_dir: Path, check_files: bool = False) -> list[dict]:
    """Load the ReXGradient-160K processed splits.

    Expects ``processed/{train,valid,test}.jsonl`` produced by the dataset's
    own ``prepare_data.py``: each line has ``image_path`` (relative to the
    dataset root), ``indication``, ``comparison``, ``findings`` and
    ``impression``. Frontal selection and 512 px resizing are already done
    upstream, so here we only cast to the ms-swift messages format with the
    same prompt template as IU-Xray. The dataset root is resolved once and
    per-record existence checks are off by default (160k stat calls are slow
    on networked storage); pass ``check_files=True`` to enable them.
    """
    root = data_dir.resolve()
    processed = root / "processed"
    records: list[dict] = []
    for split in ("train", "valid", "test"):
        path = processed / f"{split}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ReXGradient processed file: {path}")
        with path.open(encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                findings = clean(row.get("findings"))
                impression = clean(row.get("impression"))
                if not (findings or impression):
                    continue
                image = root / row["image_path"]
                if check_files and not image.is_file():
                    continue
                ctx = context(row.get("indication"), row.get("comparison"))
                prompt = (PROMPT_BOTH if impression else PROMPT_FINDINGS).format(context=ctx)
                records.append(
                    {
                        "id": str(row.get("id") or row.get("study_uid") or ""),
                        "dataset": "rexgradient",
                        "split": split,
                        "image": str(image),
                        "indication": clean(row.get("indication")),
                        "comparison": clean(row.get("comparison")),
                        "prompt": prompt,
                        "answer": target(findings, impression),
                    }
                )
    return records


def write_jsonl(records: list[dict], output_dir: Path, prefix: str = "iu") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid", "test"):
        path = output_dir / f"{prefix}_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for row in records:
                if row["split"] != split:
                    continue
                # ms-swift messages format, while retaining fields used by baseline.py.
                item = dict(row)
                item["messages"] = [
                    {"role": "user", "content": f"<image>{row['prompt']}"},
                    {"role": "assistant", "content": row["answer"]},
                ]
                item["images"] = [row["image"]]
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "prepare-rexgradient"))
    parser.add_argument("--iu-dir", type=Path, default=Path("../iu-xray"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/rexrank"))
    parser.add_argument("--rexgradient-dir", type=Path, default=Path("../ReXGradient-160K"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--check-files", action="store_true",
                        help="stat each image during ReXGradient prep (slow; off by default)")
    args = parser.parse_args()
    if args.command == "prepare":
        records = load_iu(args.iu_dir, args.split_dir)
        write_jsonl(records, args.output_dir, prefix="iu")
    else:
        records = load_rexgradient(args.rexgradient_dir, check_files=args.check_files)
        write_jsonl(records, args.output_dir, prefix="rexgradient")
    counts = Counter(r["split"] for r in records)
    print(json.dumps({"prepared": len(records), "splits": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
