#!/usr/bin/env python
"""Prepare grouped train/val/test splits for Template YAML GEPA experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _counter(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key, "")) for row in rows).items()))


def _split_groups(
    rows: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = row.get("sample_id")
        if not sample_id:
            sample_id = str(row.get("case_id", "")).split("_Q", 1)[0]
        grouped[str(sample_id)].append(row)

    group_ids = sorted(grouped)
    random.Random(seed).shuffle(group_ids)

    n = len(group_ids)
    train_n = max(1, round(n * train_ratio)) if n else 0
    val_n = max(1, round(n * val_ratio)) if n - train_n > 1 else max(0, n - train_n)
    if train_n + val_n > n:
        val_n = max(0, n - train_n)

    train_ids = set(group_ids[:train_n])
    val_ids = set(group_ids[train_n : train_n + val_n])
    test_ids = set(group_ids[train_n + val_n :])

    return {
        "train": [row for gid in group_ids if gid in train_ids for row in grouped[gid]],
        "val": [row for gid in group_ids if gid in val_ids for row in grouped[gid]],
        "test": [row for gid in group_ids if gid in test_ids for row in grouped[gid]],
    }


def _split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(rows),
        "sample_ids": sorted({str(row.get("sample_id", "")) for row in rows}),
        "expected_backend": _counter(rows, "expected_backend"),
        "scenario": _counter(rows, "scenario"),
        "category": _counter(rows, "category"),
    }


def _write_report(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Template YAML GEPA Data Report",
        "",
        f"Input: `{manifest['input']}`",
        f"Input SHA256: `{manifest['input_sha256']}`",
        f"Seed: `{manifest['seed']}`",
        "",
        "| Split | Cases | Conversations |",
        "| --- | ---: | ---: |",
    ]
    for split in ("train", "val", "test"):
        summary = manifest["splits"][split]
        lines.append(f"| {split} | {summary['count']} | {len(summary['sample_ids'])} |")
    lines.extend(["", "## Warnings", ""])
    warnings = manifest.get("warnings", [])
    if warnings:
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.append("No warnings.")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input route-label JSONL.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="locomo_v2_template")
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.20)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    rows = _read_jsonl(input_path)
    splits = _split_groups(rows, args.train_ratio, args.val_ratio, args.seed)

    for split, split_rows in splits.items():
        _write_jsonl(output_dir / f"{args.prefix}.{split}.jsonl", split_rows)

    all_sample_ids = {str(row.get("sample_id", "")) for row in rows}
    warnings: list[str] = []
    if len(all_sample_ids) <= 10:
        warnings.append(
            "Input has 10 or fewer sample_id groups; grouped test results may have high variance."
        )
    if not splits["test"]:
        warnings.append("test split is empty; adjust ratios or input size.")

    manifest = {
        "input": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "total_count": len(rows),
        "total_sample_ids": len(all_sample_ids),
        "splits": {split: _split_summary(split_rows) for split, split_rows in splits.items()},
        "warnings": warnings,
    }
    (output_dir / f"{args.prefix}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_report(output_dir / f"{args.prefix}.data_report.md", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
