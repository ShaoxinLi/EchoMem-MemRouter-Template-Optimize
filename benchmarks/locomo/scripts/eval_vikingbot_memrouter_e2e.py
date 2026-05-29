#!/usr/bin/env python
"""VikingBot + MemRouter + OpenViking backend E2E route evaluation.

This runner intentionally goes through VikingBot's ``openviking_search`` tool
instead of calling ``MemRouterPipeline`` directly. That keeps the measured path
close to the production integration:

    VikingSearchTool -> MemRouterVikingClient -> OpenViking AsyncHTTPClient

Only OpenViking is executable today. Graph and temporal backends are evaluated
as MemRouter route decisions; they are not dispatched to real backends yet.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


BACKENDS = [
    "openviking_memory_backend",
    "graph_memory_backend",
    "temporal_memory_backend",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_openviking_root() -> Path:
    return _repo_root().parent / "OpenViking"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _safe_parse_tool_output(text: str) -> dict[str, Any] | None:
    """Parse VikingSearchTool string output when it is a dict representation."""
    stripped = text.strip()
    if not stripped.startswith("{"):
        return None
    try:
        value = ast.literal_eval(stripped)
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _is_template_route(route_method: str) -> bool:
    return route_method in {
        "template_embedding",
        "template_embedding_multi_backend",
        "template_rerank",
    }


def _is_valid_ov_instruction(instruction: Any) -> tuple[bool, str]:
    if not isinstance(instruction, dict):
        return False, "instruction_missing_or_not_object"
    if instruction.get("backend_id") != "openviking_memory_backend":
        return False, "backend_not_openviking"
    if not (instruction.get("query") or "").strip():
        return False, "missing_query"
    if instruction.get("search_mode") not in {"search", "find"}:
        return False, "invalid_search_mode"
    if instruction.get("context_type") != "memory":
        return False, "invalid_context_type"
    if not (instruction.get("target_uri") or "").strip():
        return False, "missing_target_uri"
    skip_intent_analysis = instruction.get("skip_intent_analysis")
    if not isinstance(skip_intent_analysis, bool):
        return False, "skip_intent_analysis_not_bool"
    typed = instruction.get("typed_query")
    # Fast path instructions must carry a typed query so OpenViking can bypass
    # IntentAnalyzer. Conservative fallback instructions may intentionally set
    # skip_intent_analysis=false and let OpenViking native search analyze intent.
    if skip_intent_analysis:
        typed = typed or {}
        if not isinstance(typed, dict):
            return False, "typed_query_not_object"
        if not (typed.get("query") or "").strip():
            return False, "typed_query_missing_query"
        if typed.get("context_type") != "memory":
            return False, "typed_query_invalid_context_type"
        if not (typed.get("intent") or "").strip():
            return False, "typed_query_missing_intent"
    elif typed is not None:
        if not isinstance(typed, dict):
            return False, "typed_query_not_object"
    return True, "ok"


def _rate(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r.get("is_backend_correct") is True)
    template_hits = sum(1 for r in rows if r.get("is_template_hit") is True)
    invalid = sum(1 for r in rows if r.get("error"))
    fallback = sum(1 for r in rows if r.get("actual_route_method") == "llm_backend_fallback")
    return {
        "count": total,
        "backend_accuracy": _rate(correct, total),
        "template_hit_rate": _rate(template_hits, total),
        "llm_fallback_rate": _rate(fallback, total),
        "invalid_rate": _rate(invalid, total),
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


async def _run_cases(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Path]:
    echomem_root = _repo_root()
    openviking_root = Path(args.openviking_root).resolve()
    ov_config = Path(args.ov_config).resolve()
    memrouter_config = Path(args.memrouter_config).resolve()

    sys.path.insert(0, str(echomem_root))
    sys.path.insert(0, str(openviking_root))
    sys.path.insert(0, str(openviking_root / "bot"))

    os.environ["OPENVIKING_CONFIG_FILE"] = str(ov_config)
    os.environ["MEMROUTER_ENABLED"] = "true"
    os.environ["ECHOMEM_PATH"] = str(echomem_root)
    os.environ["MEMROUTER_CONFIG"] = str(memrouter_config)

    dataset_path = Path(args.dataset).resolve()
    dataset_name = dataset_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir).resolve() / f"{timestamp}_{dataset_name}_vikingbot_memrouter_ov"
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MEMROUTER_RUNS_DIR"] = str(run_dir / "logs")

    # Imports must happen after sys.path and env are prepared.
    from vikingbot.agent.tools.base import ToolContext
    from vikingbot.agent.tools.ov_file import VikingSearchTool
    from vikingbot.config.loader import ensure_config
    from vikingbot.config.schema import SessionKey

    ensure_config(ov_config)

    cases = _load_jsonl(dataset_path)
    if args.filter_backend:
        cases = [
            c for c in cases
            if c.get("expected", {}).get("primary_backend_id") == args.filter_backend
        ]
    if args.limit:
        cases = cases[: args.limit]

    (run_dir / "dataset_snapshot.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases),
        encoding="utf-8",
    )
    _write_json(
        run_dir / "run_config.json",
        {
            "dataset": str(dataset_path),
            "ov_config": str(ov_config),
            "memrouter_config": str(memrouter_config),
            "openviking_root": str(openviking_root),
            "scope": "VikingSearchTool -> MemRouterVikingClient -> OpenViking AsyncHTTPClient",
            "note": "API keys are intentionally omitted from this run config.",
        },
    )

    tool = VikingSearchTool()
    results: list[dict[str, Any]] = []

    for idx, case in enumerate(cases, 1):
        case_id = case.get("case_id", f"case_{idx:03d}")
        question = case.get("question") or case.get("query") or ""
        expected_backend = case.get("expected", {}).get("primary_backend_id", "")
        session_key = SessionKey(
            type="cli",
            channel_id="memrouter_e2e",
            chat_id=case_id,
        )
        ctx = ToolContext(
            session_key=session_key,
            sandbox_manager=None,
            workspace_id=args.workspace_id,
            sender_id=case_id,
        )

        started = time.perf_counter()
        error = ""
        tool_output = ""
        parsed_output: dict[str, Any] | None = None
        try:
            tool_output = await tool.execute(ctx, query=question, target_uri="")
            parsed_output = _safe_parse_tool_output(tool_output)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = int((time.perf_counter() - started) * 1000)

        meta = (parsed_output or {}).get("_memrouter_meta") or {}
        route_method = meta.get("route_method", "")
        actual_backend = meta.get("backend_id", "")
        instruction = meta.get("instruction")
        valid_ov_instruction, ov_instruction_reason = _is_valid_ov_instruction(instruction)

        result = {
            "case_id": case_id,
            "benchmark": case.get("benchmark", ""),
            "scenario": case.get("scenario", ""),
            "question": question,
            "expected_primary_backend_id": expected_backend,
            "actual_primary_backend_id": actual_backend,
            "actual_route_method": route_method,
            "is_backend_correct": actual_backend == expected_backend if actual_backend else False,
            "is_template_hit": _is_template_route(route_method),
            "latency_ms": latency_ms,
            "tool_output_parseable": parsed_output is not None,
            "error": error,
            "memrouter_meta": meta,
            "ov_instruction": {
                "checked": expected_backend == "openviking_memory_backend"
                or actual_backend == "openviking_memory_backend",
                "valid": valid_ov_instruction,
                "reason": ov_instruction_reason,
                "search_mode": instruction.get("search_mode") if isinstance(instruction, dict) else None,
                "intent": (
                    (instruction.get("typed_query") or {}).get("intent")
                    if isinstance(instruction, dict)
                    else None
                ),
            },
        }
        results.append(result)
        print(
            f"[{idx:03d}/{len(cases)}] {case_id} expected={expected_backend} "
            f"actual={actual_backend or 'none'} method={route_method or 'none'} "
            f"ok={result['is_backend_correct']} ov_inst={valid_ov_instruction}"
        )

    if hasattr(tool, "_memrouter_client") and tool._memrouter_client is not None:
        await tool._memrouter_client.close()
    return results, run_dir


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_actual_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matched_templates: dict[str, int] = defaultdict(int)
    for row in results:
        by_backend[row["expected_primary_backend_id"]].append(row)
        by_actual_backend[row.get("actual_primary_backend_id") or "none"].append(row)
        if row.get("is_template_hit"):
            matched_template_id = (
                (row.get("memrouter_meta") or {}).get("matched_template_id")
                or "unknown_template"
            )
            matched_templates[matched_template_id] += 1

    expected_ov = [
        r for r in results
        if r["expected_primary_backend_id"] == "openviking_memory_backend"
    ]
    actual_ov = [
        r for r in results
        if r["actual_primary_backend_id"] == "openviking_memory_backend"
    ]
    expected_ov_correct = [
        r for r in expected_ov
        if r["actual_primary_backend_id"] == "openviking_memory_backend"
    ]

    def count_valid(rows: list[dict[str, Any]]) -> int:
        return sum(1 for r in rows if r["ov_instruction"]["valid"] is True)

    return {
        "overall": _group_metrics(results),
        "by_expected_backend": {
            backend: _group_metrics(rows)
            for backend, rows in sorted(by_backend.items())
        },
        "by_actual_backend": {
            backend: _group_metrics(rows)
            for backend, rows in sorted(by_actual_backend.items())
        },
        "matched_template_usage": dict(sorted(matched_templates.items())),
        "openviking_instruction": {
            "expected_ov_cases": len(expected_ov),
            "actual_ov_cases": len(actual_ov),
            "expected_ov_correct_backend": len(expected_ov_correct),
            "expected_ov_valid_instruction": count_valid(expected_ov),
            "actual_ov_valid_instruction": count_valid(actual_ov),
            "expected_ov_correct_and_valid_instruction": count_valid(expected_ov_correct),
            "expected_ov_valid_instruction_rate": _rate(count_valid(expected_ov), len(expected_ov)),
            "actual_ov_valid_instruction_rate": _rate(count_valid(actual_ov), len(actual_ov)),
            "expected_ov_correct_and_valid_instruction_rate": _rate(
                count_valid(expected_ov_correct), len(expected_ov_correct)
            ),
        },
        "wrong_cases": [
            {
                "case_id": r["case_id"],
                "expected": r["expected_primary_backend_id"],
                "actual": r["actual_primary_backend_id"],
                "route_method": r["actual_route_method"],
                "question": r["question"],
            }
            for r in results
            if r["is_backend_correct"] is not True
        ],
    }


def _write_report(run_dir: Path, summary: dict[str, Any]) -> None:
    pct = lambda v: "N/A" if v is None else f"{v * 100:.2f}%"
    lines = [
        "# VikingBot + MemRouter + OpenViking E2E Summary",
        "",
        "Scope: VikingBot `openviking_search` tool -> MemRouterVikingClient -> OpenViking AsyncHTTPClient.",
        "Graph and temporal backends are route-only in this phase; only OpenViking instruction executability is checked.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Backend route accuracy | {pct(summary['overall']['backend_accuracy'])} |",
        f"| Template hit rate | {pct(summary['overall']['template_hit_rate'])} |",
        f"| LLM fallback rate | {pct(summary['overall']['llm_fallback_rate'])} |",
        f"| Invalid/error rate | {pct(summary['overall']['invalid_rate'])} |",
        "",
        "## By Expected Backend",
        "",
        "This table groups cases by benchmark label. `Template hit` means the case was routed by any template, not necessarily a template for the expected backend.",
        "",
        "| Expected backend | Cases | Accuracy | Template hit | LLM fallback |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for backend, values in summary["by_expected_backend"].items():
        lines.append(
            f"| `{backend}` | {values['count']} | {pct(values['backend_accuracy'])} | "
            f"{pct(values['template_hit_rate'])} | {pct(values['llm_fallback_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## By Actual Backend",
            "",
            "This table shows which backend MemRouter actually selected at runtime.",
            "",
            "| Actual backend | Cases | Template hit | LLM fallback |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for backend, values in summary["by_actual_backend"].items():
        lines.append(
            f"| `{backend}` | {values['count']} | "
            f"{pct(values['template_hit_rate'])} | {pct(values['llm_fallback_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Matched Template Usage",
            "",
            "Only accepted template routes are counted here.",
            "",
            "| Matched template | Hits |",
            "| --- | ---: |",
        ]
    )
    if summary["matched_template_usage"]:
        for template_id, count in summary["matched_template_usage"].items():
            lines.append(f"| `{template_id}` | {count} |")
    else:
        lines.append("| N/A | 0 |")

    ov = summary["openviking_instruction"]
    lines.extend(
        [
            "",
            "## OpenViking Instruction",
            "",
            "| Scope | Count | Valid executable instruction | Rate |",
            "| --- | ---: | ---: | ---: |",
            f"| Expected OV cases | {ov['expected_ov_cases']} | {ov['expected_ov_valid_instruction']} | {pct(ov['expected_ov_valid_instruction_rate'])} |",
            f"| Actually routed to OV | {ov['actual_ov_cases']} | {ov['actual_ov_valid_instruction']} | {pct(ov['actual_ov_valid_instruction_rate'])} |",
            f"| Expected OV and correctly routed | {ov['expected_ov_correct_backend']} | {ov['expected_ov_correct_and_valid_instruction']} | {pct(ov['expected_ov_correct_and_valid_instruction_rate'])} |",
            "",
            "## Wrong Cases",
            "",
        ]
    )
    if summary["wrong_cases"]:
        lines.extend(["| Case | Expected | Actual | Method | Question |", "| --- | --- | --- | --- | --- |"])
        for row in summary["wrong_cases"]:
            q = row["question"].replace("|", "\\|")
            lines.append(
                f"| `{row['case_id']}` | `{row['expected']}` | `{row['actual']}` | "
                f"`{row['route_method']}` | {q} |"
            )
    else:
        lines.append("No wrong cases.")

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Golden-mini JSONL dataset path.")
    parser.add_argument(
        "--ov-config",
        default=str(_default_openviking_root() / "benchmark" / "memrouter_ov_e2e" / "config" / "ov.conf"),
        help="OpenViking ov.conf path.",
    )
    parser.add_argument(
        "--memrouter-config",
        default=str(_repo_root() / "configs" / "memrouter_eval.local.yaml"),
        help="MemRouter local config path.",
    )
    parser.add_argument(
        "--openviking-root",
        default=str(_default_openviking_root()),
        help="OpenViking repository root.",
    )
    parser.add_argument(
        "--runs-dir",
        default=str(_repo_root() / "runs" / "vikingbot_memrouter_e2e"),
        help="Directory for E2E run artifacts.",
    )
    parser.add_argument("--workspace-id", default="memrouter-e2e", help="VikingBot workspace/agent id.")
    parser.add_argument("--filter-backend", choices=BACKENDS, default=None)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results, run_dir = asyncio.run(_run_cases(args))
    summary = _build_summary(results)

    with (run_dir / "route_results.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(run_dir / "metrics_summary.json", summary)
    _write_report(run_dir, summary)

    print(f"\nRun directory: {run_dir}")
    print(json.dumps(summary["overall"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
