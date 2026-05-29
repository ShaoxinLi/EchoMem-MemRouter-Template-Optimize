#!/usr/bin/env python3
"""Post-process answer judge for completed LoCoMo E2E run.

Reads qa_results.csv, grades unjudged answers, then regenerates
metrics_summary.json and report.md.

Usage:
    python benchmarks/locomo/scripts/postprocess_judge.py \
        --qa-csv runs/20260525_120000_locomo_e2e/results/qa_results.csv \
        --judge-token sk-your-deepseek-token
"""

import argparse
import asyncio
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    sys.exit(1)

JUDGE_MODEL = "deepseek-v4-flash"
JUDGE_BASE_URL = "https://api.deepseek.com/v1"
BATCH_SIZE = 10
BATCH_DELAY = 1.0


async def _grade_answer(client: AsyncOpenAI, question: str, gold_answer: str, response: str) -> tuple[bool | None, str]:
    system_prompt = (
        "You are an expert grader that determines if answers to questions "
        "match a gold standard answer"
    )
    prompt = f"""Your task is to label an answer to a question as 'CORRECT' or 'WRONG'.

Question: {question}
Gold answer: {gold_answer}
Generated answer: {response}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response.

Respond with JSON only: {{"is_correct": "CORRECT" or "WRONG", "reasoning": "your explanation"}}
"""
    try:
        resp = await client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            timeout=60,
        )
        content = resp.choices[0].message.content.strip()
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx != -1 and end_idx != -1:
            json_str = content[start_idx : end_idx + 1].strip()
            result = json.loads(json_str)
            is_correct = result.get("is_correct", "WRONG").strip().upper() == "CORRECT"
            reasoning = result.get("reasoning", "")
            return is_correct, reasoning
        return False, f"[PARSE ERROR] Invalid response: {content[:200]}"
    except Exception as e:
        return None, f"[API ERROR] {str(e)[:200]}"


def _rate(num: int, denom: int) -> float | None:
    return round(num / denom, 4) if denom else None


def _compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rows)
    labeled = [r for r in rows if r.get("expected_backend")]
    def _evt_cnt(row):
        try:
            return int(row.get("route_event_count", 0) or 0)
        except (ValueError, TypeError):
            return 0
    effective = [r for r in labeled if r.get("actual_backend") or _evt_cnt(r) > 0]
    infra_fail = [r for r in labeled if not r.get("actual_backend") and _evt_cnt(r) == 0]

    backend_correct = sum(1 for r in effective if r.get("is_backend_correct") == "True")
    first_backend_correct = backend_correct
    any_backend_hit = backend_correct
    template_hits = sum(1 for r in rows if r.get("is_template_hit") == "True")
    fallback = sum(1 for r in rows if r.get("route_method") == "llm_backend_fallback")
    invalid = sum(1 for r in rows if r.get("route_method") == "none" or not r.get("actual_backend"))

    by_expected: dict[str, list] = {}
    by_actual: dict[str, list] = {}
    matched_templates: Counter = Counter()
    for r in rows:
        eb = r.get("expected_backend", "")
        ab = r.get("actual_backend", "")
        if eb:
            by_expected.setdefault(eb, []).append(r)
        if ab:
            by_actual.setdefault(ab, []).append(r)
        mt = r.get("matched_template_id", "")
        if mt:
            matched_templates[mt] += 1

    ov_routed = [r for r in rows if r.get("actual_backend") == "openviking_memory_backend"]
    ov_expected = [r for r in rows if r.get("expected_backend") == "openviking_memory_backend"]
    ov_expected_correct = [r for r in ov_expected if r.get("is_backend_correct") == "True"]

    def count_valid(rows_: list) -> int:
        return sum(1 for r in rows_ if r.get("ov_instruction_valid") == "True")

    judged = [r for r in rows if r.get("judge_correct") in ("True", "False")]
    correct_answers = sum(1 for r in judged if r.get("judge_correct") == "True")
    template_hit_judged = [r for r in judged if r.get("is_template_hit") == "True"]
    template_hit_correct = sum(1 for r in template_hit_judged if r.get("judge_correct") == "True")

    joint_both = sum(1 for r in rows if r.get("is_backend_correct") == "True" and r.get("judge_correct") == "True")
    joint_route_only = sum(1 for r in rows if r.get("is_backend_correct") == "True" and r.get("judge_correct") == "False")
    joint_answer_only = sum(1 for r in rows if r.get("is_backend_correct") != "True" and r.get("judge_correct") == "True")
    joint_neither = sum(1 for r in rows if r.get("is_backend_correct") != "True" and r.get("judge_correct") == "False")

    non_ov_selected = sum(1 for r in rows if r.get("actual_backend") in ("graph_memory_backend", "temporal_memory_backend"))

    by_category: dict[str, dict] = {}
    for r in rows:
        cat = r.get("category", "")
        if not cat:
            continue
        by_category.setdefault(cat, []).append(r)

    by_category_out = {}
    for cat, cat_rows in sorted(by_category.items()):
        by_category_out[cat] = {
            "count": len(cat_rows),
            "backend_accuracy": _rate(
                sum(1 for r in cat_rows if r.get("is_backend_correct") == "True"),
                len(cat_rows),
            ),
            "answer_accuracy": _rate(
                sum(1 for r in cat_rows if r.get("judge_correct") == "True"),
                sum(1 for r in cat_rows if r.get("judge_correct") in ("True", "False")),
            ),
            "template_hit_rate": _rate(
                sum(1 for r in cat_rows if r.get("is_template_hit") == "True"),
                len(cat_rows),
            ),
        }

    return {
        "overall": {
            "count": total,
            "with_route_observed": sum(1 for r in rows if r.get("actual_backend")),
            "labeled_count": len(labeled),
            "effective_labeled_count": len(effective),
            "infra_fail_count": len(infra_fail),
            "backend_accuracy": _rate(backend_correct, len(effective)) if effective else None,
            "first_backend_accuracy": _rate(first_backend_correct, len(effective)) if effective else None,
            "any_backend_hit_rate": _rate(any_backend_hit, len(effective)) if effective else None,
            "template_hit_rate": _rate(template_hits, total),
            "llm_fallback_rate": _rate(fallback, total),
            "invalid_rate": _rate(invalid, total),
            "answer_accuracy": _rate(correct_answers, len(judged)) if judged else None,
            "token_savings": {
                "template_hit_count": template_hits,
                "llm_fallback_count": fallback,
                "token_savings_rate": _rate(template_hits, total),
                "estimated_tokens_saved": template_hits * 4500,
                "estimated_tokens_consumed": fallback * 4500,
                "note": "Estimated at ~4500 tokens per fallback: MemRouter LLM fallback (~1000) + OV IntentAnalyzer (~3500). Template hits bypass both.",
                "baseline_tokens_per_query": 3500,
                "memrouter_avg_tokens_per_query": round(fallback * 4500 / total, 1) if total else 0,
                "tokens_saved_vs_baseline_per_query": round((4500 * template_hits / total - 1000) if total else 0, 1),
                "savings_pct_vs_baseline": round(((4500 * template_hits / total - 1000) / 3500) if total else 0, 4),
            },
        },
        "by_expected_backend": {
            backend: {
                "count": len(rows_),
                "backend_accuracy": _rate(sum(1 for r in rows_ if r.get("is_backend_correct") == "True"), len(rows_)),
                "template_hit_rate": _rate(sum(1 for r in rows_ if r.get("is_template_hit") == "True"), len(rows_)),
                "llm_fallback_rate": _rate(sum(1 for r in rows_ if r.get("route_method") == "llm_backend_fallback"), len(rows_)),
            }
            for backend, rows_ in sorted(by_expected.items())
        },
        "by_actual_backend": {
            backend: {
                "count": len(rows_),
                "template_hit_rate": _rate(sum(1 for r in rows_ if r.get("is_template_hit") == "True"), len(rows_)),
                "llm_fallback_rate": _rate(sum(1 for r in rows_ if r.get("route_method") == "llm_backend_fallback"), len(rows_)),
            }
            for backend, rows_ in sorted(by_actual.items())
        },
        "matched_template_usage": dict(sorted(matched_templates.items())),
        "openviking_instruction": {
            "expected_ov_cases": len(ov_expected),
            "actual_ov_cases": len(ov_routed),
            "expected_ov_correct_backend": len(ov_expected_correct),
            "expected_ov_valid_instruction": count_valid(ov_expected),
            "actual_ov_valid_instruction": count_valid(ov_routed),
            "expected_ov_correct_and_valid_instruction": count_valid(ov_expected_correct),
            "expected_ov_valid_instruction_rate": _rate(count_valid(ov_expected), len(ov_expected)),
            "actual_ov_valid_instruction_rate": _rate(count_valid(ov_routed), len(ov_routed)),
            "expected_ov_correct_and_valid_instruction_rate": _rate(count_valid(ov_expected_correct), len(ov_expected_correct)),
        },
        "non_ov_backend_selected_but_not_executed_count": non_ov_selected,
        "answer": {
            "judged": len(judged),
            "correct": correct_answers,
            "accuracy": _rate(correct_answers, len(judged)) if judged else None,
            "template_hit_judged": len(template_hit_judged),
            "template_hit_correct": template_hit_correct,
            "template_hit_accuracy": _rate(template_hit_correct, len(template_hit_judged)) if template_hit_judged else None,
        },
        "joint": {
            "both_correct": joint_both,
            "route_only": joint_route_only,
            "answer_only": joint_answer_only,
            "neither": joint_neither,
        },
        "by_category": by_category_out,
    }


def _write_report(summary: dict[str, Any]) -> str:
    pct = lambda v: "N/A" if v is None else f"{v * 100:.2f}%"
    lines = [
        "# LoCoMo + VikingBot + MemRouter + OpenViking Agent-level E2E Summary",
        "",
        "Scope: VikingBot `/bot/v1/chat` -> MemRouterVikingClient -> OpenViking.",
        "",
        "> **Note**: Graph and temporal backends represent MemRouter route",
        "> decisions only. Real graph/temporal backends are not connected in this",
        "> phase. When MemRouter selects them, the physical execution currently falls",
        "> back to OpenViking native search as a stop-gap.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total cases | {summary['overall']['count']} |",
        f"| Route observed | {summary['overall']['with_route_observed']} |",
        f"| Labeled cases | {summary['overall']['labeled_count']} |",
        f"| Infra failures (no route) | {summary['overall']['infra_fail_count']} |",
        f"| Effective labeled cases* | {summary['overall']['effective_labeled_count']} |",
        f"| Backend route accuracy (first event) | {pct(summary['overall']['first_backend_accuracy'])} |",
        f"| Backend route accuracy (any event) | {pct(summary['overall']['any_backend_hit_rate'])} |",
        f"| Template hit rate | {pct(summary['overall']['template_hit_rate'])} |",
        f"| LLM fallback rate | {pct(summary['overall']['llm_fallback_rate'])} |",
        f"| Invalid/error rate | {pct(summary['overall']['invalid_rate'])} |",
        f"| Answer accuracy | {pct(summary['overall']['answer_accuracy'])} |",
        "",
        "> *Backend accuracy is computed over **effective labeled cases** (labeled cases excluding infrastructure failures where no route event was observed).",
        "",
        "## Token Cost Efficiency",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Template hits (saved LLM calls) | {summary['overall']['token_savings']['template_hit_count']} |",
        f"| LLM fallback calls | {summary['overall']['token_savings']['llm_fallback_count']} |",
        f"| Token savings rate | {pct(summary['overall']['token_savings']['token_savings_rate'])} |",
        f"| Estimated tokens saved (MemRouter+OV) | {summary['overall']['token_savings']['estimated_tokens_saved']:,} |",
        f"| Estimated tokens consumed by fallback | {summary['overall']['token_savings']['estimated_tokens_consumed']:,} |",
        f"| Baseline (original OV) tokens/query | {summary['overall']['token_savings']['baseline_tokens_per_query']:,} |",
        f"| MemRouter avg tokens/query | {summary['overall']['token_savings']['memrouter_avg_tokens_per_query']:,} |",
        f"| Tokens saved vs baseline / query | {summary['overall']['token_savings']['tokens_saved_vs_baseline_per_query']:,} |",
        f"| **Token savings vs baseline** | **{pct(summary['overall']['token_savings']['savings_pct_vs_baseline'])}** |",
        "",
        f"> {summary['overall']['token_savings']['note']}",
        "",
        "## By Expected Backend",
        "",
        "| Expected backend | Cases | Accuracy | Template hit | LLM fallback |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for backend, vals in summary["by_expected_backend"].items():
        lines.append(
            f"| `{backend}` | {vals['count']} | {pct(vals['backend_accuracy'])} | "
            f"{pct(vals['template_hit_rate'])} | {pct(vals['llm_fallback_rate'])} |"
        )

    lines.extend([
        "",
        "## By Actual Backend",
        "",
        "| Actual backend | Cases | Template hit | LLM fallback |",
        "| --- | ---: | ---: | ---: |",
    ])
    for backend, vals in summary["by_actual_backend"].items():
        lines.append(
            f"| `{backend}` | {vals['count']} | "
            f"{pct(vals['template_hit_rate'])} | {pct(vals['llm_fallback_rate'])} |"
        )

    lines.extend([
        "",
        "## Matched Template Usage",
        "",
        "| Matched template | Hits |",
        "| --- | ---: |",
    ])
    if summary["matched_template_usage"]:
        for template_id, count in summary["matched_template_usage"].items():
            lines.append(f"| `{template_id}` | {count} |")
    else:
        lines.append("| N/A | 0 |")

    ov = summary["openviking_instruction"]
    lines.extend([
        "",
        "## OpenViking Instruction",
        "",
        "| Scope | Count | Valid executable instruction | Rate |",
        "| --- | ---: | ---: | ---: |",
        f"| Expected OV cases | {ov['expected_ov_cases']} | {ov['expected_ov_valid_instruction']} | {pct(ov['expected_ov_valid_instruction_rate'])} |",
        f"| Actually routed to OV | {ov['actual_ov_cases']} | {ov['actual_ov_valid_instruction']} | {pct(ov['actual_ov_valid_instruction_rate'])} |",
        f"| Expected OV and correctly routed | {ov['expected_ov_correct_backend']} | {ov['expected_ov_correct_and_valid_instruction']} | {pct(ov['expected_ov_correct_and_valid_instruction_rate'])} |",
        "",
        "## Non-OV Backend Note",
        "",
        f"Cases where MemRouter selected graph/temporal (physical execution falls back to OV native search): **{summary['non_ov_backend_selected_but_not_executed_count']}**",
        "",
        "## Answer Correctness",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Judged | {summary['answer']['judged']} |",
        f"| Correct | {summary['answer']['correct']} |",
        f"| Accuracy | {pct(summary['answer']['accuracy'])} |",
        f"| Template-hit judged | {summary['answer']['template_hit_judged']} |",
        f"| Template-hit correct | {summary['answer']['template_hit_correct']} |",
        f"| Template-hit accuracy | {pct(summary['answer']['template_hit_accuracy'])} |",
        "",
        "## Joint Analysis (Routing + Answer)",
        "",
        "| Routing | Answer | Count |",
        "| --- | --- | ---: |",
        f"| Correct | Correct | {summary['joint']['both_correct']} |",
        f"| Correct | Wrong | {summary['joint']['route_only']} |",
        f"| Wrong | Correct | {summary['joint']['answer_only']} |",
        f"| Wrong | Wrong | {summary['joint']['neither']} |",
        "",
        "## By Category",
        "",
        "| Category | Cases | Backend Accuracy | Answer Accuracy | Template Hit |",
        "| --- | ---: | ---: | ---: | ---: |",
    ])
    for cat, vals in summary["by_category"].items():
        lines.append(
            f"| {cat} | {vals['count']} | {pct(vals['backend_accuracy'])} | "
            f"{pct(vals['answer_accuracy'])} | {pct(vals['template_hit_rate'])} |"
        )

    lines.append("")
    return "\n".join(lines)


async def main():
    parser = argparse.ArgumentParser(description="Post-process answer judge for LoCoMo E2E")
    parser.add_argument("--qa-csv", required=True, help="Path to qa_results.csv")
    parser.add_argument("--judge-token", required=True, help="DeepSeek API token for judge")
    parser.add_argument("--judge-base-url", default=JUDGE_BASE_URL, help="Judge API base URL")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="Judge model name")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for judge API")
    parser.add_argument("--batch-delay", type=float, default=BATCH_DELAY, help="Delay between batches (seconds)")
    args = parser.parse_args()

    csv_path = Path(args.qa_csv)
    results_dir = csv_path.parent
    json_path = results_dir / "metrics_summary.json"
    report_path = results_dir / "report.md"

    print("=" * 60)
    print("Post-process Answer Judge")
    print("=" * 60)
    print(f"CSV: {csv_path}")
    print(f"API: {args.judge_base_url} / {args.judge_model}")
    print("")

    with open(csv_path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    to_judge = [
        (i, r) for i, r in enumerate(rows)
        if r.get("judge_correct") not in ("True", "False")
        and (r.get("response") or "").strip()
    ]
    already_judged = total - len(to_judge)

    print(f"Total rows: {total}")
    print(f"Already judged: {already_judged}")
    print(f"To judge: {len(to_judge)}")
    print("")

    if not to_judge:
        print("Nothing to judge. Regenerating metrics...")
        summary = _compute_metrics(rows)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(_write_report(summary))
        print("Done.")
        return

    client = AsyncOpenAI(base_url=args.judge_base_url, api_key=args.judge_token)

    judged_count = 0
    error_count = 0

    for batch_start in range(0, len(to_judge), args.batch_size):
        batch = to_judge[batch_start : batch_start + args.batch_size]
        tasks = []
        for idx, row in batch:
            tasks.append((idx, _grade_answer(
                client,
                row.get("question", ""),
                row.get("expected_answer", ""),
                row.get("response", ""),
            )))

        results = await asyncio.gather(*[t[1] for t in tasks])

        for (idx, _), (is_correct, reasoning) in zip(tasks, results):
            if is_correct is not None:
                rows[idx]["judge_correct"] = str(is_correct)
                rows[idx]["judge_reasoning"] = reasoning[:300]
                judged_count += 1
            else:
                rows[idx]["judge_correct"] = ""
                rows[idx]["judge_reasoning"] = reasoning[:300]
                error_count += 1

        print(f"  [{batch_start + len(batch)}/{len(to_judge)}] judged={judged_count} errors={error_count}")
        if batch_start + args.batch_size < len(to_judge):
            await asyncio.sleep(args.batch_delay)

    print("")
    print(f"Judging complete: {judged_count} graded, {error_count} errors")

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Updated CSV: {csv_path}")

    summary = _compute_metrics(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Updated JSON: {json_path}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(_write_report(summary))
    print(f"Updated Report: {report_path}")

    print("")
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    o = summary["overall"]
    print(f"Backend Accuracy:  {o['backend_accuracy']*100:.2f}%" if o['backend_accuracy'] else "N/A")
    print(f"Answer Accuracy:   {o['answer_accuracy']*100:.2f}%" if o['answer_accuracy'] else "N/A")
    print(f"Template Hit:      {o['template_hit_rate']*100:.2f}%")
    print(f"Fallback Rate:     {o['llm_fallback_rate']*100:.2f}%")
    j = summary["joint"]
    print(f"Joint Both Correct: {j['both_correct']}")
    print(f"Joint Route Only:   {j['route_only']}")
    print(f"Joint Answer Only:  {j['answer_only']}")
    print(f"Joint Neither:      {j['neither']}")


if __name__ == "__main__":
    asyncio.run(main())
