#!/usr/bin/env python3
"""
Generate v2 route labels from adjudication suggestions and recalculate metrics.
"""
import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path("D:/Code/cursorProject")
ORIG_LABELS = BASE / "OpenViking/benchmark/locomo_e2e/locomo_e2e_route_labels.jsonl"
SUGGESTIONS = BASE / "OpenViking/benchmark/locomo_e2e/label_adjudication/locomo_e2e_label_adjudication_suggestions_20260524.jsonl"
V2_LABELS = BASE / "OpenViking/benchmark/locomo_e2e/locomo_e2e_route_labels.v2.jsonl"
RUN_DIR = BASE / "OpenViking/benchmark/locomo_e2e/runs/20260523_203256_locomo10_locomo_agent_e2e/results"
QA_CSV = RUN_DIR / "qa_results.csv"
REPORT_PATH = RUN_DIR / "report_v2_labels.md"

def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def generate_v2_labels():
    original = {r["case_id"]: r for r in load_jsonl(ORIG_LABELS)}
    suggestions = load_jsonl(SUGGESTIONS)

    applied = 0
    by_priority = Counter()
    by_change = Counter()

    for sug in suggestions:
        cid = sug["case_id"]
        if cid not in original:
            continue
        orig = original[cid]
        old = orig["expected_backend"]
        new = sug["suggested_backend"]
        if old != new:
            orig["expected_backend"] = new
            orig["scenario"] = sug.get("suggested_scenario", orig.get("scenario", ""))
            applied += 1
            by_priority[sug.get("priority", "?")] += 1
            by_change[f"{old} -> {new}"] += 1

    with open(V2_LABELS, "w", encoding="utf-8") as f:
        for cid in sorted(original.keys(), key=lambda x: (x.split("_Q")[0], int(x.split("_Q")[1]))):
            f.write(json.dumps(original[cid], ensure_ascii=False) + "\n")

    print(f"=== V2 Labels Generated ===")
    print(f"Total cases: {len(original)}")
    print(f"Suggestions applied: {applied} / {len(suggestions)}")
    print(f"By priority: {dict(by_priority)}")
    print(f"By change: {dict(by_change)}")
    print(f"Written to: {V2_LABELS}")
    return original

def recalc_metrics(v2_labels_map):
    # Load qa_results.csv
    rows = []
    with open(QA_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Build v2 lookup
    v2_map = {k: v for k, v in v2_labels_map.items()}

    total = len(rows)
    infra_failures = 0
    backend_correct = 0
    backend_any_correct = 0
    answer_correct = 0
    template_hits = 0
    category_stats = defaultdict(lambda: {"total": 0, "backend": 0, "answer": 0, "template_hit": 0})
    backend_matrix = defaultdict(lambda: {"total": 0, "correct": 0, "answer_correct": 0})

    for row in rows:
        cid = row["case_id"]
        v2 = v2_map.get(cid)
        if not v2:
            continue

        expected = v2["expected_backend"]
        actual = row.get("actual_backend", "")
        first_route = row.get("first_route_backend", "")
        all_routes = row.get("all_route_backends", "[]")
        is_template_hit = row.get("is_template_hit", "False").lower() == "true"
        judge_correct = row.get("judge_correct", "False").lower() == "true"
        category = int(row.get("category", 0))

        if row.get("error"):
            infra_failures += 1
            continue

        # Backend accuracy (first)
        if actual == expected:
            backend_correct += 1

        # Backend any-hit
        try:
            all_backends = eval(all_routes) if all_routes.startswith("[") else []
        except Exception:
            all_backends = []
        if expected in all_backends:
            backend_any_correct += 1

        # Answer accuracy
        if judge_correct:
            answer_correct += 1

        # Template hit
        if is_template_hit:
            template_hits += 1

        # Category stats
        cat = category_stats[category]
        cat["total"] += 1
        if actual == expected:
            cat["backend"] += 1
        if judge_correct:
            cat["answer"] += 1
        if is_template_hit:
            cat["template_hit"] += 1

        # Backend matrix
        bm = backend_matrix[(expected, actual)]
        bm["total"] += 1
        if actual == expected:
            bm["correct"] += 1
        if judge_correct:
            bm["answer_correct"] += 1

    effective = total - infra_failures

    # Build report
    lines = []
    lines.append("# MemRouter E2E Metrics Recalculated with V2 Labels")
    lines.append(f"\n**Run**: 20260523_203256")
    lines.append(f"**Original labels**: {ORIG_LABELS.name}")
    lines.append(f"**V2 labels applied**: {len([s for s in load_jsonl(SUGGESTIONS) if s.get('priority')])} suggestions")
    lines.append("")
    lines.append("## Overall Metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total questions | {total} |")
    lines.append(f"| Infra failures | {infra_failures} |")
    lines.append(f"| Effective cases | {effective} |")
    lines.append(f"| **Backend accuracy (first)** | {backend_correct}/{effective} = **{backend_correct/effective*100:.2f}%** |")
    lines.append(f"| Backend accuracy (any) | {backend_any_correct}/{effective} = {backend_any_correct/effective*100:.2f}% |")
    lines.append(f"| **Answer accuracy** | {answer_correct}/{effective} = **{answer_correct/effective*100:.2f}%** |")
    lines.append(f"| Template hit rate | {template_hits}/{total} = {template_hits/total*100:.2f}% |")
    lines.append(f"| LLM fallback rate | {total-template_hits}/{total} = {(total-template_hits)/total*100:.2f}% |")
    lines.append("")

    # Token savings
    fallback = total - template_hits
    savings_pct = round(((4500 * template_hits / total - 1000) / 3500) if total else 0, 4)
    lines.append("## Token Savings")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Template hits | {template_hits} |")
    lines.append(f"| LLM fallbacks | {fallback} |")
    lines.append(f"| Baseline tokens/query | 3500 |")
    lines.append(f"| MemRouter avg tokens/query | {round(fallback * 4500 / total, 1) if total else 0} |")
    lines.append(f"| **Savings % vs baseline** | **{savings_pct*100:.2f}%** |")
    lines.append("")

    # Category breakdown
    lines.append("## Category Breakdown")
    lines.append("")
    lines.append("| Category | Cases | Backend Acc | Answer Acc | Template Hit |")
    lines.append("|----------|-------|-------------|------------|--------------|")
    for cat in sorted(category_stats.keys()):
        s = category_stats[cat]
        lines.append(f"| {cat} | {s['total']} | {s['backend']/s['total']*100:.2f}% | {s['answer']/s['total']*100:.2f}% | {s['template_hit']/s['total']*100:.2f}% |")
    lines.append("")

    # Backend matrix
    lines.append("## Backend Routing Matrix (Expected → Actual)")
    lines.append("")
    lines.append("| Expected | Actual | Cases | Answer Acc |")
    lines.append("|----------|--------|-------|------------|")
    for (exp, act), data in sorted(backend_matrix.items(), key=lambda x: -x[1]["total"]):
        lines.append(f"| {exp} | {act} | {data['total']} | {data['answer_correct']/data['total']*100:.2f}% |")
    lines.append("")

    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n=== Metrics Recalculated ===")
    print(f"Backend accuracy (first): {backend_correct}/{effective} = {backend_correct/effective*100:.2f}%")
    print(f"Backend accuracy (any): {backend_any_correct}/{effective} = {backend_any_correct/effective*100:.2f}%")
    print(f"Answer accuracy: {answer_correct}/{effective} = {answer_correct/effective*100:.2f}%")
    print(f"Template hit rate: {template_hits}/{total} = {template_hits/total*100:.2f}%")
    print(f"Token savings vs baseline: {savings_pct*100:.2f}%")
    print(f"Report written to: {REPORT_PATH}")

if __name__ == "__main__":
    v2_labels = generate_v2_labels()
    recalc_metrics(v2_labels)
