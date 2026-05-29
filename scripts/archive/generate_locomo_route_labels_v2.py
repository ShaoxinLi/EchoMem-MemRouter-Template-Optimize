#!/usr/bin/env python
"""Generate route labels for LoCoMo E2E evaluation based on template semantics.

Uses accumulated templates under echomem/templates_data/ instead of naive
category->backend heuristic.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def classify(question: str, category: int | str) -> tuple[str, str]:
    """Return (expected_backend, scenario) based on template semantics."""
    q = question.strip()
    q_lower = q.lower()

    # --- Temporal (timeline fact / duration / sequence) ---
    temporal_starts = [
        r"^when did\b",
        r"^when is\b",
        r"^when will\b",
        r"^when was\b",
        r"^when has\b",
        r"^how long\b",
        r"^how many days\b",
        r"^how many weeks\b",
        r"^how many months\b",
        r"^how many years\b",
        r"^which year did\b",
        r"^which month did\b",
        r"^which week did\b",
    ]
    for pat in temporal_starts:
        if re.search(pat, q_lower):
            return "temporal_memory_backend", "temporal_fact"

    if re.search(r"\bhappened first\b|\bhappened last\b|\bcame first\b|\bcame later\b", q_lower):
        return "temporal_memory_backend", "sequence_reasoning"

    if re.search(r"\bwho did\b.*\bon\s+\w+\s+\d+", q_lower):
        return "temporal_memory_backend", "temporal_fact"

    # "on the evening/morning/afternoon of DATE" implies temporal anchor
    if re.search(r"\bon\s+the\s+(evening|morning|afternoon)\s+of\s+\d{1,2}\s+\w+", q_lower):
        return "temporal_memory_backend", "temporal_fact"

    # --- OpenViking subjective experience (personal_fact_lookup) ---
    # "realize", "feel", "think about" as inner-state should go to OV, not graph
    subjective_patterns = [
        r"\brealize\b",
        r"\bfeels?\b",
        r"\bfeeling\b",
        r"^what did .* think about\b",
    ]
    for pat in subjective_patterns:
        if re.search(pat, q_lower):
            return "openviking_memory_backend", "personal_fact_lookup"

    # --- Graph (entity relation / causal / structured attributes) ---
    graph_patterns = [
        r"relationship between",
        r"discuss with",
        r"\bboth like to\b",
        r"\bboth enjoy\b",
        r"\bboth have in common\b",
        r"\bhave in common\b",
        r"\bsimilarities between\b",
        r"\bdifferences between\b",
        r"\bcompare\b.*\band\b",
        r"\bconnect\b.*\band\b",
        r"\blink\b.*\band\b",
        r"\bassociated with\b",
        r"\brelated to\b",
        r"\btogether\b.*\bwith\b",
        r"^why did\b",
        r"^what caused\b",
        r"^what led to\b",
        r"^what is .*\bidentity\b",
        r"^what is .*\brelationship status\b",
        r"^what is .*\bstatus\b",
        r"^where did .* move from\b",
        r"^where has .*\b",
        r"^what .* has .* (participated in|attended|visited|done)\b",
        r"^what organizations?\b",
        r"^which organization\b",
        r"^who (worked|collaborated) with\b",
        r"^who are .* friends\b",
    ]
    for pat in graph_patterns:
        if re.search(pat, q_lower):
            return "graph_memory_backend", "entity_relation"

    # --- Causal multihop (subset of graph) ---
    causal_patterns = [
        r"^why did .* decide to\b",
        r"^why did .* choose\b",
        r"^why did .* start\b",
        r"^what made .*\b",
        r"^what explains\b",
        r"^what caused\b",
    ]
    for pat in causal_patterns:
        if re.search(pat, q_lower):
            return "graph_memory_backend", "causal_multihop"

    # --- Aggregation / summary / suggestion (OpenViking) ---
    aggregation_patterns = [
        r"^would .* prefer\b",
        r"^would .* likely\b",
        r"^would .* be considered\b",
        r"^would .* still want\b",
        r"^would .* be more interested\b",
        r"^can you suggest\b",
        r"^what would .* likely be\b",
        r"^what might .* be\b",
        r"^compare\b",
        r"^summarize\b",
    ]
    for pat in aggregation_patterns:
        if re.search(pat, q_lower):
            return "openviking_memory_backend", "aggregation_summary"

    # --- Preference / profile (OpenViking) ---
    pref_patterns = [
        r"\bpreference\b",
        r"\bhobby\b",
        r"\bhobbies\b",
        r"\blifestyle\b",
        r"\bethnicity\b",
        r"\bbackground\b",
        r"\bhabit\b",
        r"\bdestress\b",
        r"\bself-care\b",
        r"\bexcited about\b",
        r"\bthink about\b",
        r"\bopinion\b",
        r"\bpersonality\b",
        r"\bgoals?\b",
        r"\bplans?\b",
    ]
    for pat in pref_patterns:
        if re.search(pat, q_lower):
            return "openviking_memory_backend", "preference_profile"

    # --- Previous chat recall (OpenViking) ---
    chat_patterns = [
        r"^remind me\b",
        r"^what did you (recommend|suggest|say)\b",
        r"^what did we discuss\b",
        r"^do you remember\b",
        r"^can you recall\b",
    ]
    for pat in chat_patterns:
        if re.search(pat, q_lower):
            return "openviking_memory_backend", "previous_chat_recall"

    # --- Fallback by original category ---
    cat = int(category)
    if cat == 2:
        return "temporal_memory_backend", "temporal_fact"
    if cat == 4:
        return "graph_memory_backend", "commonality"
    # cat 1 & 3 -> openviking
    return "openviking_memory_backend", "personal_fact" if cat == 1 else "reasoning"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Path to LoCoMo JSON file.")
    parser.add_argument("--output", required=True, help="Path to output JSONL labels file.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    written = 0
    changes = []
    with output_path.open("w", encoding="utf-8") as out:
        for item in data:
            sample_id = item["sample_id"]
            for qi, qa in enumerate(item.get("qa", []), start=1):
                category = qa.get("category", "")
                if str(category) == "5":
                    continue
                backend, scenario = classify(qa["question"], category)
                label = {
                    "case_id": f"{sample_id}_Q{qi}",
                    "sample_id": sample_id,
                    "qi": qi,
                    "expected_backend": backend,
                    "scenario": scenario,
                    "question": qa["question"],
                    "category": category,
                }
                out.write(json.dumps(label, ensure_ascii=False) + "\n")
                written += 1

    print(f"Generated {written} labels -> {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
