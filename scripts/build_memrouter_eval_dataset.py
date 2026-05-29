"""Build MemRouter evaluation datasets.

Usage:
    python scripts/build_memrouter_eval_dataset.py
"""

import json
import random
import sys
from pathlib import Path

# Reproducible sampling
random.seed(42)

# Canonical scenario labels aligned with v1.4 test plan
CANONICAL_SCENARIOS = {
    # LoCoMo
    "temporal_fact_lookup": "temporal_fact",
    "factual_attribute_lookup": "personal_memory",
    "causal_multihop": "causal_multihop",
    "entity_relation_query": "entity_relation",
    "detail_event_lookup": "personal_memory",
    # LongMemEval
    "temporal_reasoning": "temporal_fact",
    "personal_fact_lookup": "personal_memory",
    "preference_lookup": "personal_memory",
    "assistant_context_recall": "personal_memory",
    "multi_session_aggregation": "aggregation",
    "knowledge_update_tracking": "knowledge_update",
    # EvolvingEvents
    "medium_complexity_lookup": "personal_memory",
    "simple_fact_lookup": "personal_memory",
    "fuzzy_semantic_search": "personal_memory",
    # Synthetic
    "personal_memory_recall": "personal_memory",
    "timeline_fact_query": "temporal_fact",
    "weather_query_hard_negative": "hard_negative",
    "math_query_hard_negative": "hard_negative",
    "general_knowledge_hard_negative": "hard_negative",
}


def _make_eval_policy(
    allow_llm_fallback: bool = True,
    strict_primary_backend: bool = True,
) -> dict:
    """Standard eval_policy for backend-only routing."""
    return {
        "allow_llm_fallback": allow_llm_fallback,
        "strict_primary_backend": strict_primary_backend,
    }


def _canonical_scenario(scenario: str) -> str:
    return CANONICAL_SCENARIOS.get(scenario, scenario)


def _infer_temporal_query_shape(question: str) -> str:
    """Classify temporal query shape for dataset analysis helpers."""
    q = question.lower()
    sequence_markers = [
        "last",
        "earlier",
        "later",
        "before",
        "after",
        "between",
        "passed",
        "how many days",
        "how many weeks",
        "how many months",
        "how long",
        "order",
        "which came",
        "which event",
        "which task",
        "which trip",
        "which device",
        "which pet",
        "what happened first",
        "happened first",
        "attend first",
        "complete first",
        "became a parent first",
        "did i do first",
        "哪个更早",
        "哪件事先",
        "先后顺序",
        "隔了多久",
        "过去了多少",
    ]
    if any(marker in q for marker in sequence_markers):
        return "sequence_reasoning"
    return "timeline_lookup"


def _load_locomo_data() -> tuple[dict[int, list[dict]], list[dict]]:
    """Load and categorize LoCoMo QA pairs."""
    with open(
        "D:/Code/cursorProject/locomo/data/locomo10.json", "r", encoding="utf-8"
    ) as f:
        data = json.load(f)

    by_category: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: [], 5: []}
    for conv_idx, conv in enumerate(data):
        for q in conv.get("qa", []):
            cat = q.get("category", 1)
            by_category.setdefault(cat, []).append(
                {
                    "conversation_idx": conv_idx,
                    "category": cat,
                    "question": q["question"],
                    "answer": q.get("answer", ""),
                }
            )
    return by_category, data


def build_locomo_samples(count: int = 10) -> list[dict]:
    """Extract representative samples from LoCoMo."""
    by_category, _ = _load_locomo_data()

    samples = []

    # Category 2 (temporal) -> temporal_memory_backend
    temporal_n = max(3, count // 3)
    temporal_pool = by_category[2]
    temporal_selected = random.sample(temporal_pool, min(temporal_n, len(temporal_pool)))
    for i, item in enumerate(temporal_selected):
        samples.append(
            {
                "case_id": f"locomo_t{i + 1}",
                "benchmark": "locomo",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "temporal_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("temporal_fact_lookup"),
                "source_info": {
                    "conversation_idx": item["conversation_idx"],
                    "category": item["category"],
                },
            }
        )

    # Category 1 (factual attribute) -> openviking_memory_backend
    fact_n = max(4, count // 2 - 1)
    fact_pool = by_category[1]
    fact_selected = random.sample(fact_pool, min(fact_n, len(fact_pool)))
    for i, item in enumerate(fact_selected):
        samples.append(
            {
                "case_id": f"locomo_f{i + 1}",
                "benchmark": "locomo",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "openviking_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("factual_attribute_lookup"),
                "source_info": {
                    "conversation_idx": item["conversation_idx"],
                    "category": item["category"],
                },
            }
        )

    # Category 3 (inference/causal) -> openviking_memory_backend
    infer_n = max(2, count // 5)
    infer_pool = by_category[3]
    infer_selected = random.sample(infer_pool, min(infer_n, len(infer_pool)))
    for i, item in enumerate(infer_selected):
        samples.append(
            {
                "case_id": f"locomo_i{i + 1}",
                "benchmark": "locomo",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "openviking_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("causal_multihop"),
                "source_info": {
                    "conversation_idx": item["conversation_idx"],
                    "category": item["category"],
                },
            }
        )

    # Category 4 or 5 with relation-like wording -> graph_memory_backend
    relation_pool = [
        q
        for q in by_category[4] + by_category[5]
        if any(
            kw in q["question"].lower()
            for kw in ["who", "with", "friend", "relationship", "support"]
        )
    ]
    relation_n = max(1, count // 10)
    relation_selected = random.sample(relation_pool, min(relation_n, len(relation_pool)))
    for i, item in enumerate(relation_selected):
        samples.append(
            {
                "case_id": f"locomo_g{i + 1}",
                "benchmark": "locomo",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "graph_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("entity_relation_query"),
                "source_info": {
                    "conversation_idx": item["conversation_idx"],
                    "category": item["category"],
                },
            }
        )

    # Fill remainder from category 4
    if len(samples) < count:
        used = {id(s) for s in temporal_selected + fact_selected + infer_selected + relation_selected}
        detail_pool = [q for q in by_category[4] if id(q) not in used]
        needed = count - len(samples)
        extra = random.sample(detail_pool, min(needed, len(detail_pool)))
        for i, item in enumerate(extra):
            samples.append(
                {
                    "case_id": f"locomo_d{i + 1}",
                    "benchmark": "locomo",
                    "question": item["question"],
                    "expected": {
                        "primary_backend_id": "openviking_memory_backend",
                        "expected_routing_mode": "template_preferred",
                    },
                    "eval_policy": _make_eval_policy(),
                    "is_hard_negative": False,
                    "scenario": _canonical_scenario("detail_event_lookup"),
                    "source_info": {
                        "conversation_idx": item["conversation_idx"],
                        "category": item["category"],
                    },
                }
            )

    return samples[:count]


def build_longmemeval_samples(count: int = 10) -> list[dict]:
    """Extract representative samples from LongMemEval."""
    with open(
        "D:/Code/cursorProject/LongMemEval/data/longmemeval_oracle.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    by_type: dict[str, list[dict]] = {}
    for item in data:
        qt = item["question_type"]
        by_type.setdefault(qt, []).append(item)

    samples = []

    # temporal-reasoning -> temporal_memory_backend
    temporal_n = max(3, count // 3)
    temporal_pool = by_type.get("temporal-reasoning", [])
    temporal_selected = random.sample(temporal_pool, min(temporal_n, len(temporal_pool)))
    for i, item in enumerate(temporal_selected):
        samples.append(
            {
                "case_id": f"lme_t{i + 1}",
                "benchmark": "longmemeval",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "temporal_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("temporal_reasoning"),
                "source_info": {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                },
            }
        )

    # single-session-user -> openviking_memory_backend
    user_n = max(2, count // 5)
    user_pool = by_type.get("single-session-user", [])
    user_selected = random.sample(user_pool, min(user_n, len(user_pool)))
    for i, item in enumerate(user_selected):
        samples.append(
            {
                "case_id": f"lme_u{i + 1}",
                "benchmark": "longmemeval",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "openviking_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("personal_fact_lookup"),
                "source_info": {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                },
            }
        )

    # single-session-preference -> openviking_memory_backend
    pref_n = max(1, count // 10)
    pref_pool = by_type.get("single-session-preference", [])
    pref_selected = random.sample(pref_pool, min(pref_n, len(pref_pool)))
    for i, item in enumerate(pref_selected):
        samples.append(
            {
                "case_id": f"lme_p{i + 1}",
                "benchmark": "longmemeval",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "openviking_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("preference_lookup"),
                "source_info": {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                },
            }
        )

    # single-session-assistant -> openviking_memory_backend
    asst_n = max(1, count // 10)
    asst_pool = by_type.get("single-session-assistant", [])
    asst_selected = random.sample(asst_pool, min(asst_n, len(asst_pool)))
    for i, item in enumerate(asst_selected):
        samples.append(
            {
                "case_id": f"lme_a{i + 1}",
                "benchmark": "longmemeval",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "openviking_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("assistant_context_recall"),
                "source_info": {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                },
            }
        )

    # multi-session -> openviking_memory_backend
    multi_n = max(1, count // 10)
    multi_pool = by_type.get("multi-session", [])
    multi_selected = random.sample(multi_pool, min(multi_n, len(multi_pool)))
    for i, item in enumerate(multi_selected):
        samples.append(
            {
                "case_id": f"lme_m{i + 1}",
                "benchmark": "longmemeval",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "openviking_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("multi_session_aggregation"),
                "source_info": {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                },
            }
        )

    # knowledge-update -> openviking_memory_backend
    ku_n = max(2, count // 5)
    ku_pool = by_type.get("knowledge-update", [])
    ku_selected = random.sample(ku_pool, min(ku_n, len(ku_pool)))
    for i, item in enumerate(ku_selected):
        samples.append(
            {
                "case_id": f"lme_k{i + 1}",
                "benchmark": "longmemeval",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "openviking_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("knowledge_update_tracking"),
                "source_info": {
                    "question_id": item["question_id"],
                    "question_type": item["question_type"],
                },
            }
        )

    return samples[:count]


def build_evolvingevents_samples(count: int = 10) -> list[dict]:
    """Extract representative samples from EvolvingEvents."""
    with open(
        "D:/Code/cursorProject/mflow-benchmarks/benchmarks/evolving-events/data/qa_pairs.json",
        "r",
        encoding="utf-8",
    ) as f:
        data = json.load(f)

    multi_hop = [item for item in data if "multi_hop" in item.get("tags", "")]
    non_multi = [item for item in data if "multi_hop" not in item.get("tags", "")]

    samples = []

    # multi_hop -> graph_memory_backend
    mh_n = max(4, count // 2 - 2)
    mh_selected = random.sample(multi_hop, min(mh_n, len(multi_hop)))
    for i, item in enumerate(mh_selected):
        samples.append(
            {
                "case_id": f"ee_mh{i + 1}",
                "benchmark": "evolvingevents",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "graph_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("causal_multihop"),
                "source_info": {"tags": item.get("tags", ""), "coarse": item.get("coarse", "")},
            }
        )

    # Medium non-multi -> openviking_memory_backend
    med_non_multi = [
        item for item in non_multi if item.get("tags", "").startswith("M")
    ]
    m_n = max(4, count // 2 - 2)
    m_selected = random.sample(med_non_multi, min(m_n, len(med_non_multi)))
    for i, item in enumerate(m_selected):
        samples.append(
            {
                "case_id": f"ee_m{i + 1}",
                "benchmark": "evolvingevents",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "openviking_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("medium_complexity_lookup"),
                "source_info": {
                    "tags": item.get("tags", ""),
                    "coarse": item.get("coarse", ""),
                },
            }
        )

    # Small non-multi -> openviking_memory_backend
    small_non_multi = [
        item for item in non_multi if item.get("tags", "").startswith("S")
    ]
    s_n = max(2, count // 5)
    s_selected = random.sample(small_non_multi, min(s_n, len(small_non_multi)))
    for i, item in enumerate(s_selected):
        samples.append(
            {
                "case_id": f"ee_s{i + 1}",
                "benchmark": "evolvingevents",
                "question": item["question"],
                "expected": {
                    "primary_backend_id": "openviking_memory_backend",
                    "expected_routing_mode": "template_preferred",
                },
                "eval_policy": _make_eval_policy(),
                "is_hard_negative": False,
                "scenario": _canonical_scenario("simple_fact_lookup"),
                "source_info": {
                    "tags": item.get("tags", ""),
                    "coarse": item.get("coarse", ""),
                },
            }
        )

    return samples[:count]


def build_synthetic_smoke() -> list[dict]:
    """Build synthetic smoke set from template prototypes and hand-crafted cases."""
    prototypes = [
        # openviking
        {
            "case_id": "synth_openviking_001",
            "benchmark": "synthetic",
            "question": "我之前说过我喜欢什么沟通方式吗？",
            "expected": {
                "primary_backend_id": "openviking_memory_backend",
                "expected_routing_mode": "template_preferred",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=False),
            "is_hard_negative": False,
            "scenario": _canonical_scenario("personal_memory_recall"),
        },
        {
            "case_id": "synth_openviking_002",
            "benchmark": "synthetic",
            "question": "你记得我的代码风格偏好吗？",
            "expected": {
                "primary_backend_id": "openviking_memory_backend",
                "expected_routing_mode": "template_preferred",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=False),
            "is_hard_negative": False,
            "scenario": _canonical_scenario("personal_memory_recall"),
        },
        {
            "case_id": "synth_openviking_003",
            "benchmark": "synthetic",
            "question": "按我的偏好来处理这个任务",
            "expected": {
                "primary_backend_id": "openviking_memory_backend",
                "expected_routing_mode": "template_preferred",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=False),
            "is_hard_negative": False,
            "scenario": _canonical_scenario("personal_memory_recall"),
        },
        # graph
        {
            "case_id": "synth_graph_001",
            "benchmark": "synthetic",
            "question": "Jon 和 Gina 是什么关系？",
            "expected": {
                "primary_backend_id": "graph_memory_backend",
                "expected_routing_mode": "template_preferred",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=False),
            "is_hard_negative": False,
            "scenario": _canonical_scenario("entity_relation_query"),
        },
        {
            "case_id": "synth_graph_002",
            "benchmark": "synthetic",
            "question": "Jon 的朋友有哪些？",
            "expected": {
                "primary_backend_id": "graph_memory_backend",
                "expected_routing_mode": "template_preferred",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=False),
            "is_hard_negative": False,
            "scenario": _canonical_scenario("entity_relation_query"),
        },
        {
            "case_id": "synth_graph_003",
            "benchmark": "synthetic",
            "question": "Gina 属于哪个团队？",
            "expected": {
                "primary_backend_id": "graph_memory_backend",
                "expected_routing_mode": "template_preferred",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=False),
            "is_hard_negative": False,
            "scenario": _canonical_scenario("entity_relation_query"),
        },
        # temporal
        {
            "case_id": "synth_temporal_001",
            "benchmark": "synthetic",
            "question": "Jon 什么时候失去工作的？",
            "expected": {
                "primary_backend_id": "temporal_memory_backend",
                "expected_routing_mode": "template_preferred",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=False),
            "is_hard_negative": False,
            "scenario": _canonical_scenario("timeline_fact_query"),
        },
        {
            "case_id": "synth_temporal_002",
            "benchmark": "synthetic",
            "question": "上次我们是什么时候讨论这个问题的？",
            "expected": {
                "primary_backend_id": "temporal_memory_backend",
                "expected_routing_mode": "template_preferred",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=False),
            "is_hard_negative": False,
            "scenario": _canonical_scenario("timeline_fact_query"),
        },
        {
            "case_id": "synth_temporal_003",
            "benchmark": "synthetic",
            "question": "这几件事的先后顺序是什么？",
            "expected": {
                "primary_backend_id": "temporal_memory_backend",
                "expected_routing_mode": "template_preferred",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=False),
            "is_hard_negative": False,
            "scenario": _canonical_scenario("timeline_fact_query"),
        },
        # Hard negatives
        {
            "case_id": "synth_hard_negative_001",
            "benchmark": "synthetic",
            "question": "今天北京天气怎么样？",
            "expected": {
                "primary_backend_id": "openviking_memory_backend",
                "expected_routing_mode": "llm_expected",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=True),
            "is_hard_negative": True,
            "scenario": _canonical_scenario("weather_query_hard_negative"),
        },
        {
            "case_id": "synth_hard_negative_002",
            "benchmark": "synthetic",
            "question": "2+2等于几？",
            "expected": {
                "primary_backend_id": "openviking_memory_backend",
                "expected_routing_mode": "llm_expected",
            },
            "eval_policy": _make_eval_policy(allow_llm_fallback=True),
            "is_hard_negative": True,
            "scenario": _canonical_scenario("math_query_hard_negative"),
        },
    ]
    return prototypes


def build_golden_mini() -> list[dict]:
    """Build Golden-mini dataset: 60 cases (20 per benchmark)."""
    samples = []
    samples.extend(build_locomo_samples(count=20))
    samples.extend(build_longmemeval_samples(count=20))
    samples.extend(build_evolvingevents_samples(count=20))
    return samples


def build_balanced_set() -> list[dict]:
    """Build balanced dataset: ~20 per backend + 3 hard negatives.

    Guarantees roughly equal backend representation for backend routing tests.
    """
    random.seed(2025)

    # Pull maximum available from each benchmark (count=999 effectively unconstrained)
    locomo_all = build_locomo_samples(count=999)
    lme_all = build_longmemeval_samples(count=999)
    ee_all = build_evolvingevents_samples(count=999)
    all_samples = locomo_all + lme_all + ee_all

    temporal_pool = [
        s for s in all_samples
        if s["expected"]["primary_backend_id"] == "temporal_memory_backend"
    ]
    graph_pool = [
        s for s in all_samples
        if s["expected"]["primary_backend_id"] == "graph_memory_backend"
    ]
    openviking_pool = [
        s for s in all_samples
        if s["expected"]["primary_backend_id"] == "openviking_memory_backend"
    ]

    def select_balanced(pool: list[dict], n: int) -> list[dict]:
        """Round-robin select across benchmarks, then fill remainder randomly."""
        by_bench: dict[str, list[dict]] = {}
        for s in pool:
            by_bench.setdefault(s["benchmark"], []).append(s)

        selected: list[dict] = []
        benches = list(by_bench.keys())
        idx = {b: 0 for b in benches}
        while len(selected) < n and any(
            idx[b] < len(by_bench[b]) for b in benches
        ):
            for b in benches:
                if len(selected) < n and idx[b] < len(by_bench[b]):
                    selected.append(by_bench[b][idx[b]])
                    idx[b] += 1

        if len(selected) < n:
            remaining = [
                s for b in benches for s in by_bench[b][idx[b]:]
            ]
            needed = n - len(selected)
            selected.extend(
                random.sample(remaining, min(needed, len(remaining)))
            )
        return selected[:n]

    samples: list[dict] = []
    samples.extend(select_balanced(temporal_pool, 20))
    samples.extend(select_balanced(graph_pool, 20))
    samples.extend(select_balanced(openviking_pool, 20))

    # Hard negatives — ensure 3 distinct ones
    hard_negs = [
        s for s in build_synthetic_smoke() if s.get("is_hard_negative")
    ]
    if not any(
        "general" in s["case_id"] for s in hard_negs
    ):
        hard_negs.append(
            {
                "case_id": "synth_hard_negative_003",
                "benchmark": "synthetic",
                "question": "法国的首都是哪里？",
                "expected": {
                    "primary_backend_id": "openviking_memory_backend",
                    "expected_routing_mode": "llm_expected",
                },
                "eval_policy": _make_eval_policy(allow_llm_fallback=True),
                "is_hard_negative": True,
                "scenario": _canonical_scenario(
                    "general_knowledge_hard_negative"
                ),
            }
        )
    random.shuffle(hard_negs)
    samples.extend(hard_negs[:3])

    # Re-assign sequential case_ids
    for i, s in enumerate(samples):
        s["case_id"] = f"balanced_{i + 1:03d}"

    return samples


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Build MemRouter evaluation datasets")
    parser.add_argument(
        "--dataset-type",
        choices=["smoke", "golden-mini", "balanced", "all"],
        default="all",
        help="Which dataset to build (default: all)",
    )
    args = parser.parse_args()

    out_dir = Path("data/memrouter_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    from collections import Counter

    if args.dataset_type in ("smoke", "all"):
        # Build benchmark smoke set
        benchmark_samples = []
        benchmark_samples.extend(build_locomo_samples())
        benchmark_samples.extend(build_longmemeval_samples())
        benchmark_samples.extend(build_evolvingevents_samples())

        smoke_path = out_dir / "smoke_routes.jsonl"
        with open(smoke_path, "w", encoding="utf-8") as f:
            for sample in benchmark_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"Wrote {len(benchmark_samples)} benchmark samples to {smoke_path}")

        backend_dist = Counter(s["expected"]["primary_backend_id"] for s in benchmark_samples)
        scenario_dist = Counter(s["scenario"] for s in benchmark_samples)
        print("Backend distribution:", dict(backend_dist))
        print("Scenario distribution:", dict(scenario_dist))

        # Build synthetic smoke set
        synthetic_samples = build_synthetic_smoke()
        synthetic_path = out_dir / "synthetic_smoke_routes.jsonl"
        with open(synthetic_path, "w", encoding="utf-8") as f:
            for sample in synthetic_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"Wrote {len(synthetic_samples)} synthetic samples to {synthetic_path}")

        syn_backend_dist = Counter(
            s["expected"]["primary_backend_id"] for s in synthetic_samples
        )
        syn_scenario_dist = Counter(s["scenario"] for s in synthetic_samples)
        print("Synthetic backend distribution:", dict(syn_backend_dist))
        print("Synthetic scenario distribution:", dict(syn_scenario_dist))

    if args.dataset_type in ("golden-mini", "all"):
        # Build Golden-mini set
        random.seed(2024)  # Different seed for Golden-mini reproducibility
        golden_samples = build_golden_mini()
        golden_path = out_dir / "golden_mini_routes.jsonl"
        with open(golden_path, "w", encoding="utf-8") as f:
            for sample in golden_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"Wrote {len(golden_samples)} Golden-mini samples to {golden_path}")

        golden_backend_dist = Counter(
            s["expected"]["primary_backend_id"] for s in golden_samples
        )
        golden_scenario_dist = Counter(s["scenario"] for s in golden_samples)
        golden_benchmark_dist = Counter(s["benchmark"] for s in golden_samples)
        print("Golden-mini backend distribution:", dict(golden_backend_dist))
        print("Golden-mini scenario distribution:", dict(golden_scenario_dist))
        print("Golden-mini benchmark distribution:", dict(golden_benchmark_dist))

    if args.dataset_type in ("balanced", "all"):
        # Build balanced set
        balanced_samples = build_balanced_set()
        balanced_path = out_dir / "golden_mini_balanced_routes.jsonl"
        with open(balanced_path, "w", encoding="utf-8") as f:
            for sample in balanced_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"Wrote {len(balanced_samples)} balanced samples to {balanced_path}")

        balanced_backend_dist = Counter(
            s["expected"]["primary_backend_id"] for s in balanced_samples
        )
        balanced_scenario_dist = Counter(s["scenario"] for s in balanced_samples)
        balanced_benchmark_dist = Counter(s["benchmark"] for s in balanced_samples)
        print("Balanced backend distribution:", dict(balanced_backend_dist))
        print("Balanced scenario distribution:", dict(balanced_scenario_dist))
        print("Balanced benchmark distribution:", dict(balanced_benchmark_dist))

    return 0


if __name__ == "__main__":
    sys.exit(main())
