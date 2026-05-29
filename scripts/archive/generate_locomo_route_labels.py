#!/usr/bin/env python
"""Generate route labels for LoCoMo E2E evaluation.

Heuristic mapping from LoCoMo category to MemRouter backend::

    category 1 (personal fact)   -> openviking_memory_backend
    category 2 (temporal)        -> temporal_memory_backend
    category 3 (reasoning)       -> openviking_memory_backend
    category 4 (commonality)     -> graph_memory_backend
    category 5 (excluded)        -> skipped

Usage::

    python scripts/generate_locomo_route_labels.py \
      --input D:\\Code\\cursorProject\\OpenViking\\benchmark\\locomo_e2e\\locomo10.json \
      --output D:\\Code\\cursorProject\\EchoMem\\data\\memrouter_eval\\locomo_e2e_route_labels.jsonl

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _map_category_to_backend(category: int | str) -> str:
    cat = int(category)
    if cat == 2:
        return "temporal_memory_backend"
    if cat == 4:
        return "graph_memory_backend"
    # 1 (personal fact) and 3 (reasoning) -> OV as default
    return "openviking_memory_backend"


def _map_category_to_scenario(category: int | str) -> str:
    mapping = {
        1: "personal_fact",
        2: "temporal_fact",
        3: "reasoning",
        4: "commonality",
    }
    return mapping.get(int(category), "unknown")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="Path to LoCoMo JSON file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSONL labels file.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for item in data:
            sample_id = item["sample_id"]
            for qi, qa in enumerate(item.get("qa", []), start=1):
                category = qa.get("category", "")
                if str(category) == "5":
                    continue
                label = {
                    "case_id": f"{sample_id}_Q{qi}",
                    "sample_id": sample_id,
                    "qi": qi,
                    "expected_backend": _map_category_to_backend(category),
                    "scenario": _map_category_to_scenario(category),
                    "question": qa["question"],
                    "category": category,
                }
                out.write(json.dumps(label, ensure_ascii=False) + "\n")
                written += 1

    print(f"Generated {written} labels -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
