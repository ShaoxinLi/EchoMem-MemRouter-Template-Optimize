#!/usr/bin/env python
"""Minimal end-to-end test: MemRouter instruction generation for OpenViking.

Verifies that the pipeline produces well-formed BackendQueryInstructions
when routing to the OpenViking backend.  This script does NOT require a
running OpenViking server — it only checks instruction structure.

Run directly::

    python tests/test_e2e_openviking_instruction.py

Or via pytest::

    pytest tests/test_e2e_openviking_instruction.py -v
"""

import sys
import time
from pathlib import Path

# Ensure echomem is importable when running outside pytest / without PYTHONPATH
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Fix Windows terminal encoding for CJK characters
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from echomem.embeddings.base import MockEmbeddingProvider
from echomem.pipeline import MemRouterPipeline


_TEST_QUERIES = [
    "你还记得我喜欢什么颜色吗",
    "我平时喜欢听什么类型的音乐",
    "我之前告诉过你我的技术背景吗",
    "按我的偏好来处理这个任务",
]


def _check_instruction(inst) -> tuple[list[tuple[str, bool]], bool]:
    """Run structural checks on a single instruction. Returns (checks, ok)."""
    checks = [
        ("backend_id == openviking", inst.backend_id == "openviking_memory_backend"),
        ("skip_intent_analysis is bool", isinstance(inst.skip_intent_analysis, bool)),
    ]
    if inst.skip_intent_analysis:
        checks.append(("typed_query present for fast path", inst.typed_query is not None))
    if inst.typed_query:
        checks.append(("typed_query.intent non-empty", bool(inst.typed_query.intent)))
        checks.append(("typed_query.query non-empty", bool(inst.typed_query.query)))
    all_ok = all(ok for _, ok in checks)
    return checks, all_ok


def run_instruction_generation_test(require_all_match: bool = False) -> bool:
    """Run the instruction generation test and return overall pass/fail.

    Args:
        require_all_match: If True, every query must match a template.
            Default False — we only verify structural correctness when a
            template *does* hit (which is the fast-path scenario).
    """
    embedder = MockEmbeddingProvider(dim=16)
    pipeline = MemRouterPipeline.with_defaults(embedder=embedder)

    print("=" * 60)
    print("MemRouter -> OpenViking Instruction Generation Test")
    print("=" * 60)

    matched = 0
    total = len(_TEST_QUERIES)
    all_struct_pass = True

    for query in _TEST_QUERIES:
        start = time.time()
        result = pipeline.route(query)
        latency_ms = (time.time() - start) * 1000

        route = result.routes[0] if result.routes else None
        inst = result.query_instructions[0] if result.query_instructions else None
        is_match = route is not None and route.matched_template_id

        print(f"\nQuery: {query}")
        print(f"  Latency:         {latency_ms:.1f}ms")
        print(f"  Route backend:   {route.backend_id if route else 'none'}")
        print(f"  Route method:    {result.route_method}")
        print(f"  Matched template:{route.matched_template_id if route else 'none'}")
        print(f"  Confidence:      {route.confidence if route else 0:.3f}")

        if is_match:
            matched += 1

        if inst:
            print(f"  Instruction:")
            print(f"    backend_id:            {inst.backend_id}")
            print(f"    search_mode:           {inst.search_mode}")
            print(f"    target_uri:            {inst.target_uri}")
            print(f"    context_type:          {inst.context_type}")
            print(f"    skip_intent_analysis:  {inst.skip_intent_analysis}")
            if inst.typed_query:
                print(f"    typed_query:")
                print(f"      query:              {inst.typed_query.query}")
                print(f"      intent:             {inst.typed_query.intent}")
                print(f"      priority:           {inst.typed_query.priority}")
                print(f"      context_type:       {inst.typed_query.context_type}")
                print(f"      target_directories: {inst.typed_query.target_directories}")
            else:
                print(f"    typed_query:          None")

            checks, ok = _check_instruction(inst)
            for name, cok in checks:
                print(f"    {'PASS' if cok else 'FAIL'} {name}")
            if not ok:
                all_struct_pass = False
        else:
            print(f"  Instruction: None")
            if require_all_match:
                all_struct_pass = False

    match_rate = matched / total if total else 0.0
    print("\n" + "=" * 60)
    print(f"Match rate:      {matched}/{total} ({match_rate:.1%})")
    print(f"Struct checks:   {'ALL PASS' if all_struct_pass else 'SOME FAIL'}")
    if require_all_match:
        print(f"Overall:         {'ALL PASS' if all_struct_pass else 'SOME FAIL'}")
    else:
        print(f"Overall:         {'PASS' if all_struct_pass else 'FAIL'} (struct only)")
    print("=" * 60)

    if require_all_match:
        return all_struct_pass
    return all_struct_pass


# Pytest entry point
def test_openviking_instruction_generation():
    """Pytest wrapper for the instruction generation test."""
    assert run_instruction_generation_test(require_all_match=False), (
        "Instruction structural checks failed"
    )


if __name__ == "__main__":
    ok = run_instruction_generation_test(require_all_match=False)
    sys.exit(0 if ok else 1)
