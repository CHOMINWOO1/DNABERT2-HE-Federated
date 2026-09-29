from __future__ import annotations

"""Create a reproducible exact/reverse-complement split-leakage audit.

The audit intentionally uses only Python's standard library.  It does not load a
model or compute task performance, so it is safe to run before a locked test
evaluation.
"""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "dev", "test")
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reverse_complement(sequence: str) -> str:
    return sequence.translate(COMPLEMENT)[::-1]


def canonical_sequence(sequence: str) -> str:
    rc = reverse_complement(sequence)
    return sequence if sequence <= rc else rc


def load_split(path: Path) -> tuple[list[str], list[int]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["sequence", "label"]:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        sequences: list[str] = []
        labels: list[int] = []
        for row_number, row in enumerate(reader, start=2):
            sequence = row["sequence"].upper()
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Non-binary label in {path}:{row_number}")
            if not sequence or set(sequence) - set("ACGT"):
                raise ValueError(f"Invalid DNA sequence in {path}:{row_number}")
            sequences.append(sequence)
            labels.append(label)
    return sequences, labels


def label_map(sequences: list[str], labels: list[int], *, canonical: bool) -> dict[str, set[int]]:
    mapping: dict[str, set[int]] = {}
    for sequence, label in zip(sequences, labels, strict=True):
        key = canonical_sequence(sequence) if canonical else sequence
        mapping.setdefault(key, set()).add(label)
    return mapping


def split_summary(path: Path, sequences: list[str], labels: list[int]) -> dict[str, Any]:
    exact = label_map(sequences, labels, canonical=False)
    canonical = label_map(sequences, labels, canonical=True)
    lengths = sorted(map(len, sequences))
    counts = Counter(labels)
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "n": len(sequences),
        "class_counts": {str(label): counts.get(label, 0) for label in (0, 1)},
        "positive_prevalence": counts.get(1, 0) / len(sequences),
        "length_min": lengths[0],
        "length_median": lengths[len(lengths) // 2],
        "length_max": lengths[-1],
        "exact_duplicate_rows": len(sequences) - len(exact),
        "exact_conflicting_label_groups": sum(len(values) > 1 for values in exact.values()),
        "canonical_duplicate_or_rc_rows": len(sequences) - len(canonical),
        "canonical_conflicting_label_groups": sum(
            len(values) > 1 for values in canonical.values()
        ),
    }


def pair_summary(
    left_name: str,
    right_name: str,
    loaded: dict[str, tuple[list[str], list[int]]],
) -> dict[str, Any]:
    left_sequences, left_labels = loaded[left_name]
    right_sequences, right_labels = loaded[right_name]
    left_exact = label_map(left_sequences, left_labels, canonical=False)
    right_exact = label_map(right_sequences, right_labels, canonical=False)
    left_canonical = label_map(left_sequences, left_labels, canonical=True)
    right_canonical = label_map(right_sequences, right_labels, canonical=True)
    exact_overlap = set(left_exact) & set(right_exact)
    canonical_overlap = set(left_canonical) & set(right_canonical)
    return {
        "left": left_name,
        "right": right_name,
        "exact_overlap_groups": len(exact_overlap),
        "exact_label_conflicts": sum(
            left_exact[key].isdisjoint(right_exact[key]) for key in exact_overlap
        ),
        "canonical_exact_or_rc_overlap_groups": len(canonical_overlap),
        "canonical_label_conflicts": sum(
            left_canonical[key].isdisjoint(right_canonical[key])
            for key in canonical_overlap
        ),
    }


def audit_task(task_dir: Path) -> dict[str, Any]:
    loaded: dict[str, tuple[list[str], list[int]]] = {}
    summaries: dict[str, Any] = {}
    for split in SPLITS:
        path = task_dir / f"{split}.csv"
        sequences, labels = load_split(path)
        loaded[split] = (sequences, labels)
        summaries[split] = split_summary(path, sequences, labels)
    pairs = [
        pair_summary("train", "dev", loaded),
        pair_summary("train", "test", loaded),
        pair_summary("dev", "test", loaded),
    ]
    return {
        "task": task_dir.name,
        "task_dir": str(task_dir.resolve()),
        "splits": summaries,
        "cross_split": pairs,
        "scope_note": (
            "Exact sequence and exact reverse-complement audit only; genomic coordinates "
            "and near-homology clusters are unavailable in the GUE CSV files."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-dir", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    payload = {
        "schema_version": "gue-exact-rc-audit-v1",
        "script_path": str(script_path),
        "script_sha256": sha256_file(script_path),
        "tasks": [audit_task(Path(value)) for value in args.task_dir],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    print(output.resolve())


if __name__ == "__main__":
    main()
