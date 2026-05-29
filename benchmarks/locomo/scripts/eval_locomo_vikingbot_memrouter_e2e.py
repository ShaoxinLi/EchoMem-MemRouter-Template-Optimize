#!/usr/bin/env python
"""LoCoMo + VikingBot + MemRouter + OpenViking Agent-level E2E evaluator.

This runner goes through VikingBot's ``/bot/v1/chat`` endpoint (Agent-level)
rather than calling ``VikingSearchTool`` directly (tool-level).  It measures
both MemRouter routing observability and VikingBot answer correctness.

**Important**: ``route_events.jsonl`` is written by the OpenViking server
process (inside ``MemRouterVikingClient``), NOT by this runner process.
Therefore you must start the server with the same ``MEMROUTER_ROUTE_EVENTS``
path that this runner prints at startup.

Workflow::

    1. (Optional) Ingest LoCoMo conversations into OpenViking
    2. Start OpenViking server with MEMROUTER_ROUTE_EVENTS set
    3. Run this evaluator
    4. For each QA question:
       a. Send via ``/bot/v1/chat`` (VikingBot agent loop)
       b. Read back MemRouter ``route_events.jsonl``
       c. (Optional) LLM-judge answer correctness
    5. Merge routing metrics + answer accuracy into a single report

Usage (pilot, 1 sample, 3 questions, no judge)::

    python scripts/eval_locomo_vikingbot_memrouter_e2e.py ^
      --dataset D:\\Code\\cursorProject\\OpenViking\\benchmark\\locomo_e2e\\locomo10.json ^
      --route-labels D:\\Code\\cursorProject\\EchoMem\\data\\memrouter_eval\\locomo_route_labels.jsonl ^
      --route-events-path D:\\Code\\cursorProject\\EchoMem\\runs\\locomo_shared_route_events.jsonl ^
      --limit-samples 1 ^
      --limit-questions 3

"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

BACKENDS = [
    "openviking_memory_backend",
    "graph_memory_backend",
    "temporal_memory_backend",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_openviking_root() -> Path:
    return _repo_root().parent / "OpenViking"


def _load_locomo_data(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_route_labels(path: Path | None) -> dict[str, dict[str, Any]]:
    """Load route labels JSONL. Key is case_id.

    Supports two schema shapes:
    - ``{"case_id": "x", "expected_backend": "...", "scenario": "..."}``
    - ``{"case_id": "x", "expected": {"primary_backend_id": "..."}}`` (golden_mini style)
    """
    labels: dict[str, dict[str, Any]] = {}
    if path is None or not path.exists():
        return labels
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                case_id = row.get("case_id") or f"{row.get('sample_id')}_Q{row.get('qi')}"
                if not case_id:
                    continue
                # Normalize expected_backend from multiple schema shapes
                expected_backend = row.get("expected_backend", "")
                if not expected_backend and isinstance(row.get("expected"), dict):
                    expected_backend = row["expected"].get("primary_backend_id", "")
                if expected_backend:
                    row["expected_backend"] = expected_backend
                labels[case_id] = row
            except json.JSONDecodeError:
                continue
    return labels


def _load_case_ids(path: Path | None) -> set[str]:
    """Load case ids from a plain text file or JSONL file.

    Plain text format: one case_id per line.
    JSONL format: each line contains {"case_id": "..."}.
    """
    if path is None or not path.exists():
        return set()
    case_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            if raw.startswith("{"):
                try:
                    case_id = json.loads(raw).get("case_id", "")
                except json.JSONDecodeError:
                    case_id = ""
            else:
                case_id = raw
            if case_id:
                case_ids.add(case_id)
    return case_ids


def _load_route_events(path: Path, consumed: int = 0) -> tuple[list[dict[str, Any]], int]:
    """Read all route events from JSONL. Returns (new_events, total_count).

    ``consumed`` is the number of events already processed in prior calls.
    New events are those with index >= consumed.
    """
    events: list[dict[str, Any]] = []
    if not path.exists():
        return events, 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events[consumed:], len(events)


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
    skip = instruction.get("skip_intent_analysis")
    if not isinstance(skip, bool):
        return False, "skip_intent_analysis_not_bool"
    typed = instruction.get("typed_query")
    if skip:
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


def run_ov_chat(
    question: str,
    endpoint: str,
    headers: dict[str, str],
    session_id: str,
    user_id: str,
    timeout: int = 300,
) -> tuple[str, dict[str, Any]]:
    """Call OpenViking /bot/v1/chat and return (response_text, usage)."""
    url = f"{endpoint.rstrip('/')}/bot/v1/chat"
    payload = {
        "message": question,
        # VikingBot defaults to session_id="default".  E2E evaluation must not
        # reuse that shared chat session, otherwise a later case can be answered
        # from prior Bot context without invoking search tools or MemRouter.
        "session_id": session_id,
        "user_id": user_id,
        "stream": False,
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        body = resp.json()
        message = body.get("message", "")
        usage = body.get("usage") or {}
        return message, usage
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Connection error to {url}: {e}")
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request timeout to {url} after {timeout}s")
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"HTTP error {e.response.status_code} from {url}: {e}")
    except (json.JSONDecodeError, KeyError) as e:
        raise RuntimeError(f"Error parsing response from {url}: {e}")


def get_sample_question_time(sample: dict[str, Any]) -> str | None:
    """Extract the last session date from a LoCoMo sample."""
    from datetime import datetime

    conversation = sample.get("conversation", {})
    session_keys = [
        k
        for k in conversation.keys()
        if k.startswith("session_") and "date_time" not in k
    ]
    if not session_keys:
        return None

    def _num(k: str) -> int:
        try:
            return int(k.replace("session_", ""))
        except ValueError:
            return 0

    session_keys.sort(key=_num, reverse=True)
    for sk in session_keys:
        if conversation.get(sk):
            num = _num(sk)
            dt_key = f"session_{num}_date_time"
            date_str = conversation.get(dt_key)
            if date_str:
                try:
                    if " on " in date_str:
                        date_part = date_str.split(" on ")[-1]
                        dt = datetime.strptime(date_part.strip(), "%d %B, %Y")
                        return dt.strftime("%Y-%m-%d")
                except ValueError:
                    pass
    return None


async def _grade_answer(
    llm_client: Any,
    model: str,
    question: str,
    gold_answer: str,
    response: str,
) -> tuple[bool, str]:
    """Inline LLM judge (same logic as locomo_e2e/judge.py)."""
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
        resp = await llm_client.chat.completions.create(
            model=model,
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
            is_correct = (
                result.get("is_correct", "WRONG").strip().upper() == "CORRECT"
            )
            reasoning = result.get("reasoning", "")
            return is_correct, reasoning
        return False, f"[PARSE ERROR] Invalid response: {content}"
    except Exception as e:
        return False, f"[API ERROR] {str(e)}"


async def _run_cases(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Path, Path, Path]:
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

    if args.fixed_run_dir:
        run_dir = Path(args.fixed_run_dir).resolve()
    else:
        # Backward compat: if --runs-dir is explicitly given, use it as base.
        if args.runs_dir:
            base_dir = Path(args.runs_dir).resolve()
        else:
            base_dir = Path(args.output_base).resolve()
        run_dir = base_dir / f"{timestamp}_{dataset_name}_locomo_agent_e2e"
    logs_dir = run_dir / "logs"
    results_dir = run_dir / "results"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MEMROUTER_RUNS_DIR"] = str(logs_dir)

    # ------------------------------------------------------------------ #
    # Route events path - MUST be shared with the OpenViking server
    # ------------------------------------------------------------------ #
    if args.route_events_path:
        route_events_path = Path(args.route_events_path).resolve()
    else:
        route_events_path = logs_dir / "route_events.jsonl"

    # Set in env so any child server started from this shell inherits it.
    # If the server was started independently, the user must have set it
    # to the same path manually.
    os.environ["MEMROUTER_ROUTE_EVENTS"] = str(route_events_path)

    # Clear stale events so we only read events from this run
    if route_events_path.exists():
        try:
            route_events_path.unlink()
        except PermissionError:
            # File may be held open by a running server; truncate instead
            with open(route_events_path, "w", encoding="utf-8") as f:
                pass
    route_events_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LoCoMo Agent-level E2E Runner")
    print("=" * 60)
    print(f"Run directory:     {run_dir}")
    print(f"Logs directory:    {logs_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Route events path: {route_events_path}")
    print("")
    print("IMPORTANT: The OpenViking server must write route events to the")
    print("same path above. Start the server with:")
    print("")
    print(f"    $env:MEMROUTER_ROUTE_EVENTS=\"{route_events_path}\"")
    print("")
    print("Or ensure it was set before the server was launched.")
    print("=" * 60)

    # ------------------------------------------------------------------ #
    # Load route labels (expected_backend per case)
    # ------------------------------------------------------------------ #
    route_labels = _load_route_labels(
        Path(args.route_labels) if args.route_labels else None
    )
    selected_case_ids = _load_case_ids(
        Path(args.case_ids_file) if args.case_ids_file else None
    )
    if route_labels:
        print(f"Loaded {len(route_labels)} route labels from {args.route_labels}")
    else:
        print("WARNING: No route labels loaded. backend_accuracy will be N/A.")
    if selected_case_ids:
        print(f"Loaded {len(selected_case_ids)} selected case ids from {args.case_ids_file}")

    data = _load_locomo_data(dataset_path)
    if args.limit_samples:
        data = data[: args.limit_samples]

    # Build QA cases
    cases: list[dict[str, Any]] = []
    for item in data:
        sample_id = item["sample_id"]
        question_time = get_sample_question_time(item)
        all_qas = item.get("qa", [])
        filtered_qas = []
        for original_qi, qa in enumerate(all_qas, start=1):
            case_id = f"{sample_id}_Q{original_qi}"
            if selected_case_ids and case_id not in selected_case_ids:
                continue
            cat = str(qa.get("category", ""))
            if cat == "5":
                continue
            if args.category and cat != args.category:
                continue
            filtered_qas.append((original_qi, qa))
        if args.limit_questions:
            filtered_qas = filtered_qas[: args.limit_questions]
        for original_qi, qa in filtered_qas:
            cases.append(
                {
                    "sample_id": sample_id,
                    "qi": original_qi,
                    "question": qa["question"],
                    "expected_answer": str(qa["answer"]),
                    "category": qa.get("category", ""),
                    "evidence": qa.get("evidence", []),
                    "question_time": question_time,
                }
            )

    # Apply --exclude-expected-backend filter using labels
    if args.exclude_expected_backend:
        excluded_set = set(args.exclude_expected_backend)
        before = len(cases)
        cases = [
            c
            for c in cases
            if route_labels.get(f"{c['sample_id']}_Q{c['qi']}", {}).get(
                "expected_backend", ""
            )
            not in excluded_set
        ]
        after = len(cases)
        if after < before:
            print(
                f"Excluded {before - after} case(s) with expected_backend in {excluded_set}"
            )

    # Apply global --limit-questions (total across all conversations)
    if args.limit_questions:
        before_limit = len(cases)
        cases = cases[: args.limit_questions]
        if len(cases) < before_limit:
            print(f"Global limit applied: {before_limit} -> {len(cases)} cases")

    (results_dir / "dataset_snapshot.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases),
        encoding="utf-8",
    )
    _write_json(
        results_dir / "run_config.json",
        {
            "dataset": str(dataset_path),
            "ov_config": str(ov_config),
            "memrouter_config": str(memrouter_config),
            "openviking_root": str(openviking_root),
            "route_events_path": str(route_events_path),
            "route_labels": args.route_labels,
            "limit_samples": args.limit_samples,
            "limit_questions": args.limit_questions,
            "judge": args.judge,
            "scope": "VikingBot /bot/v1/chat -> MemRouterVikingClient -> OpenViking",
            "note": (
                "API keys are intentionally omitted. "
                "Graph and temporal backends are route-only in this phase."
            ),
        },
    )

    # ------------------------------------------------------------------ #
    # OpenViking chat headers
    # ------------------------------------------------------------------ #
    chat_endpoint = args.ov_chat_endpoint
    base_chat_headers = {
        "Content-Type": "application/json",
        "X-API-Key": args.ov_api_key,
        "X-OpenViking-Account": args.ov_account,
    }
    # ov_user / ov_agent can be overridden; otherwise we fall back to sample_id
    # per question to align with import_to_ov.py defaults.
    explicit_user = args.ov_user if args.ov_user else None
    explicit_agent = args.ov_agent if args.ov_agent else None

    # Optional judge client
    judge_client = None
    if args.judge:
        from openai import AsyncOpenAI

        judge_client = AsyncOpenAI(
            base_url=args.judge_base_url, api_key=args.judge_token
        )

    results: list[dict[str, Any]] = []
    consumed_events = 0
    session_prefix = args.session_prefix or run_dir.name

    for idx, case in enumerate(cases, 1):
        case_id = f"{case['sample_id']}_Q{case['qi']}"
        question = case["question"]
        expected_answer = case["expected_answer"]
        question_time = case.get("question_time")
        sample_id = case["sample_id"]

        # Resolve expected backend from labels; default empty means "unknown"
        label_row = route_labels.get(case_id, {})
        expected_backend = label_row.get("expected_backend", "")
        scenario = label_row.get("scenario", "")

        # Build per-sample headers to align with import_to_ov.py
        chat_headers = dict(base_chat_headers)
        chat_headers["X-OpenViking-User"] = explicit_user or sample_id
        chat_headers["X-OpenViking-Agent"] = explicit_agent or sample_id
        chat_user_id = explicit_user or sample_id
        chat_session_id = f"{session_prefix}_{case_id}"

        # Inject time context if available.  For MemRouter route evaluation we
        # can ask VikingBot to invoke its memory search tool before answering;
        # otherwise some questions may be answered from preloaded Bot context
        # and never reach MemRouter.
        task_prefix = (
            "Before answering, search the user's OpenViking memory using the "
            "memory search tool exactly once. Then answer the question directly: "
            if args.force_memory_search
            else "Answer the question directly: "
        )
        if question_time:
            input_msg = (
                f"Current date: {question_time}. "
                f"{task_prefix}{question}"
            )
        else:
            input_msg = f"{task_prefix}{question}"

        started = time.perf_counter()
        error = ""
        response = ""
        usage: dict[str, Any] = {}
        try:
            response, usage = run_ov_chat(
                input_msg,
                chat_endpoint,
                chat_headers,
                session_id=chat_session_id,
                user_id=chat_user_id,
                timeout=args.chat_timeout,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = int((time.perf_counter() - started) * 1000)

        # ------------------------------------------------------------------ #
        # Read route events and correlate by consumption count.
        # Since we run sequentially and the file is append-only, any new
        # events since the last question belong to this question.
        # ------------------------------------------------------------------ #
        new_events, consumed_events = _load_route_events(
            route_events_path, consumed_events
        )

        # A single chat may trigger 0, 1, or multiple search calls.
        first_event = new_events[0] if new_events else None
        all_backends = list(
            dict.fromkeys(
                ev.get("backend_id", "")
                for ev in new_events
                if ev.get("backend_id")
            )
        )
        any_expected_hit = (
            expected_backend in all_backends if expected_backend else False
        )

        route_method = first_event.get("route_method", "") if first_event else ""
        actual_backend = first_event.get("backend_id", "") if first_event else ""
        matched_template = (
            first_event.get("matched_template_id", "") if first_event else ""
        )
        confidence = first_event.get("confidence") if first_event else None
        execution_path = (
            first_event.get("execution_path", "") if first_event else ""
        )
        instruction = first_event.get("instruction") if first_event else None
        route_latency_ms = first_event.get("latency_ms", 0) if first_event else 0

        valid_ov_instruction, ov_instruction_reason = _is_valid_ov_instruction(
            instruction
        )

        # Answer grading
        judge_correct: bool | None = None
        judge_reasoning = ""
        if args.judge and response and not error:
            judge_correct, judge_reasoning = await _grade_answer(
                judge_client,
                args.judge_model,
                question,
                expected_answer,
                response,
            )

        result = {
            "case_id": case_id,
            "sample_id": sample_id,
            "qi": case["qi"],
            "chat_session_id": chat_session_id,
            "chat_user_id": chat_user_id,
            "question": question,
            "expected_answer": expected_answer,
            "expected_backend": expected_backend,
            "scenario": scenario,
            "response": response,
            "category": case["category"],
            "error": error,
            "latency_ms": latency_ms,
            "chat_usage": usage,
            "route_method": route_method,
            "actual_backend": actual_backend,
            "first_route_backend": actual_backend,
            "all_route_backends": all_backends,
            "any_expected_backend_hit": any_expected_hit,
            "matched_template_id": matched_template,
            "confidence": confidence,
            "execution_path": execution_path,
            "route_latency_ms": route_latency_ms,
            "route_event_count": len(new_events),
            "extra_route_events": new_events[1:] if len(new_events) > 1 else [],
            "is_template_hit": _is_template_route(route_method),
            "is_backend_correct": (
                actual_backend == expected_backend
                if expected_backend and actual_backend
                else False
            ),
            "ov_instruction_valid": valid_ov_instruction,
            "ov_instruction_reason": ov_instruction_reason,
            "instruction": instruction,
            "judge_correct": judge_correct,
            "judge_reasoning": judge_reasoning,
        }
        results.append(result)

        judge_status = (
            f" judge={judge_correct}" if judge_correct is not None else ""
        )
        event_status = (
            f" events={len(new_events)}" if len(new_events) != 1 else ""
        )
        print(
            f"[{idx:03d}/{len(cases)}] {case_id} "
            f"expected={expected_backend or '?'} "
            f"actual={actual_backend or 'none'} "
            f"method={route_method or 'none'} "
            f"ov_valid={valid_ov_instruction}"
            f"{judge_status}{event_status} lat={latency_ms}ms"
        )

    return results, run_dir, logs_dir, results_dir


def _build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    with_routes = [r for r in results if r["route_method"]]

    # Backend accuracy: only count samples that have an expected_backend label
    labeled_results = [r for r in results if r.get("expected_backend")]
    backend_correct = sum(
        1 for r in labeled_results if r.get("is_backend_correct")
    )
    first_backend_correct = sum(
        1 for r in labeled_results
        if r.get("first_route_backend") == r.get("expected_backend")
    )
    any_backend_hit = sum(
        1 for r in labeled_results if r.get("any_expected_backend_hit")
    )
    template_hits = sum(1 for r in results if r.get("is_template_hit"))
    fallback = sum(
        1 for r in results if r.get("route_method") == "llm_backend_fallback"
    )
    invalid = sum(1 for r in results if r.get("error"))

    # Group by expected backend
    by_expected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_actual: dict[str, list[dict[str, Any]]] = defaultdict(list)
    matched_templates: dict[str, int] = defaultdict(int)
    non_ov_selected = 0

    for row in results:
        by_expected[row.get("expected_backend") or "unknown"].append(row)
        by_actual[row.get("actual_backend") or "none"].append(row)
        if row.get("is_template_hit"):
            tid = row.get("matched_template_id") or "unknown_template"
            matched_templates[tid] += 1
        if row.get("actual_backend") in {
            "graph_memory_backend",
            "temporal_memory_backend",
        }:
            non_ov_selected += 1

    # Infrastructure failures: cases with no route event observed
    infra_fail = [
        r for r in results
        if not r.get("actual_backend") and r.get("route_event_count", 0) == 0
    ]
    # Effective labeled cases for backend accuracy (exclude infra failures)
    effective_labeled = [r for r in labeled_results if r not in infra_fail]

    # OpenViking instruction validity (only when routed to OV)
    ov_routed = [r for r in results if r.get("actual_backend") == "openviking_memory_backend"]
    ov_expected = [
        r for r in results
        if r.get("expected_backend") == "openviking_memory_backend"
    ]
    ov_expected_correct = [
        r for r in ov_expected if r.get("is_backend_correct")
    ]

    def count_valid(rows: list[dict[str, Any]]) -> int:
        return sum(1 for r in rows if r.get("ov_instruction_valid"))

    # Answer correctness
    judged = [r for r in results if r.get("judge_correct") is not None]
    correct_answers = sum(1 for r in judged if r["judge_correct"] is True)

    # Template-hit answer correctness
    template_hit_judged = [
        r for r in judged if r.get("is_template_hit")
    ]
    template_hit_correct = sum(
        1 for r in template_hit_judged if r["judge_correct"] is True
    )

    # Joint matrix: routing correct vs answer correct
    joint_both = sum(
        1
        for r in results
        if r.get("is_backend_correct") and r.get("judge_correct") is True
    )
    joint_route_only = sum(
        1
        for r in results
        if r.get("is_backend_correct") and r.get("judge_correct") is False
    )
    joint_answer_only = sum(
        1
        for r in results
        if not r.get("is_backend_correct") and r.get("judge_correct") is True
    )
    joint_neither = sum(
        1
        for r in results
        if not r.get("is_backend_correct") and r.get("judge_correct") is False
    )

    return {
        "overall": {
            "count": total,
            "with_route_observed": len(with_routes),
            "labeled_count": len(labeled_results),
            "effective_labeled_count": len(effective_labeled),
            "infra_fail_count": len(infra_fail),
            "backend_accuracy": _rate(backend_correct, len(effective_labeled)) if effective_labeled else None,
            "first_backend_accuracy": _rate(first_backend_correct, len(effective_labeled)) if effective_labeled else None,
            "any_backend_hit_rate": _rate(any_backend_hit, len(effective_labeled)) if effective_labeled else None,
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
                "savings_pct_vs_baseline": round(
                    ((4500 * template_hits / total - 1000) / 3500) if total else 0, 4
                ),
            },
        },
        "by_expected_backend": {
            backend: {
                "count": len(rows),
                "backend_accuracy": _rate(
                    sum(1 for r in rows if r.get("is_backend_correct")),
                    len(rows),
                ),
                "template_hit_rate": _rate(
                    sum(1 for r in rows if r.get("is_template_hit")), len(rows)
                ),
                "llm_fallback_rate": _rate(
                    sum(
                        1
                        for r in rows
                        if r.get("route_method") == "llm_backend_fallback"
                    ),
                    len(rows),
                ),
            }
            for backend, rows in sorted(by_expected.items())
        },
        "by_actual_backend": {
            backend: {
                "count": len(rows),
                "template_hit_rate": _rate(
                    sum(1 for r in rows if r.get("is_template_hit")), len(rows)
                ),
                "llm_fallback_rate": _rate(
                    sum(
                        1
                        for r in rows
                        if r.get("route_method") == "llm_backend_fallback"
                    ),
                    len(rows),
                ),
            }
            for backend, rows in sorted(by_actual.items())
        },
        "matched_template_usage": dict(sorted(matched_templates.items())),
        "openviking_instruction": {
            "expected_ov_cases": len(ov_expected),
            "actual_ov_cases": len(ov_routed),
            "expected_ov_correct_backend": len(ov_expected_correct),
            "expected_ov_valid_instruction": count_valid(ov_expected),
            "actual_ov_valid_instruction": count_valid(ov_routed),
            "expected_ov_correct_and_valid_instruction": count_valid(
                ov_expected_correct
            ),
            "expected_ov_valid_instruction_rate": _rate(
                count_valid(ov_expected), len(ov_expected)
            ),
            "actual_ov_valid_instruction_rate": _rate(
                count_valid(ov_routed), len(ov_routed)
            ),
            "expected_ov_correct_and_valid_instruction_rate": _rate(
                count_valid(ov_expected_correct), len(ov_expected_correct)
            ),
        },
        "non_ov_backend_selected_but_not_executed_count": non_ov_selected,
        "answer": {
            "judged": len(judged),
            "correct": correct_answers,
            "accuracy": _rate(correct_answers, len(judged)) if judged else None,
            "template_hit_judged": len(template_hit_judged),
            "template_hit_correct": template_hit_correct,
            "template_hit_accuracy": _rate(
                template_hit_correct, len(template_hit_judged)
            ),
        },
        "joint": {
            "both_correct": joint_both,
            "route_only": joint_route_only,
            "answer_only": joint_answer_only,
            "neither": joint_neither,
        },
        "by_category": _group_by_category(results),
        "wrong_cases": [
            {
                "case_id": r["case_id"],
                "expected": r["expected_backend"],
                "actual": r["actual_backend"],
                "route_method": r["route_method"],
                "question": r["question"],
            }
            for r in results
            if r["is_backend_correct"] is not True
        ],
        "no_route_cases": [
            {
                "case_id": r["case_id"],
                "expected": r.get("expected_backend", ""),
                "question": r["question"],
                "judge_correct": r.get("judge_correct"),
                "latency_ms": r.get("latency_ms", 0),
                "response": (r.get("response") or "")[:200],
            }
            for r in results
            if not r.get("actual_backend") and not r.get("route_method")
        ],
        "template_hit_wrong_cases": [
            {
                "case_id": r["case_id"],
                "expected": r.get("expected_backend", ""),
                "actual": r.get("actual_backend", ""),
                "route_method": r.get("route_method", ""),
                "matched_template_id": r.get("matched_template_id", ""),
                "question": r["question"],
                "judge_reasoning": (r.get("judge_reasoning") or "")[:300],
                "response": (r.get("response") or "")[:200],
            }
            for r in results
            if r.get("is_template_hit") and r.get("judge_correct") is False
        ],
    }


def _group_by_category(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_cat[str(r.get("category", "unknown"))].append(r)

    summary: dict[str, Any] = {}
    for cat, rows in sorted(by_cat.items()):
        judged = [r for r in rows if r.get("judge_correct") is not None]
        correct = sum(1 for r in judged if r["judge_correct"] is True)
        backend_ok = sum(1 for r in rows if r.get("is_backend_correct"))
        summary[cat] = {
            "count": len(rows),
            "backend_accuracy": _rate(backend_ok, len(rows)),
            "answer_accuracy": _rate(correct, len(judged)) if judged else None,
            "template_hit_rate": _rate(
                sum(1 for r in rows if r.get("is_template_hit")), len(rows)
            ),
        }
    return summary


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(out_dir: Path, results: list[dict[str, Any]]) -> None:
    csv_path = out_dir / "qa_results.csv"
    fieldnames = [
        "case_id",
        "sample_id",
        "qi",
        "chat_session_id",
        "chat_user_id",
        "question",
        "expected_answer",
        "expected_backend",
        "scenario",
        "response",
        "category",
        "error",
        "latency_ms",
        "route_method",
        "actual_backend",
        "first_route_backend",
        "all_route_backends",
        "any_expected_backend_hit",
        "matched_template_id",
        "confidence",
        "execution_path",
        "route_latency_ms",
        "route_event_count",
        "is_template_hit",
        "is_backend_correct",
        "ov_instruction_valid",
        "ov_instruction_reason",
        "judge_correct",
        "judge_reasoning",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)


def _write_report(out_dir: Path, summary: dict[str, Any]) -> None:
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

    lines.extend(
        [
            "",
            "## By Actual Backend",
            "",
            "| Actual backend | Cases | Template hit | LLM fallback |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for backend, vals in summary["by_actual_backend"].items():
        lines.append(
            f"| `{backend}` | {vals['count']} | "
            f"{pct(vals['template_hit_rate'])} | {pct(vals['llm_fallback_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## Matched Template Usage",
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
        ]
    )
    for cat, vals in summary["by_category"].items():
        lines.append(
            f"| {cat} | {vals['count']} | {pct(vals['backend_accuracy'])} | "
            f"{pct(vals['answer_accuracy'])} | {pct(vals['template_hit_rate'])} |"
        )

    # No-route cases analysis
    no_route = summary.get("no_route_cases", [])
    lines.extend(
        [
            "",
            "## No-Route Cases Analysis",
            "",
            f"Cases where no route event was observed: **{len(no_route)}**",
            "",
        ]
    )
    if no_route:
        lines.extend(
            [
                "| Case | Expected | Judge | Latency (ms) | Question | Response Snippet |",
                "| --- | --- | --- | ---: | --- | --- |",
            ]
        )
        for row in no_route:
            q = row["question"].replace("|", "\\|")
            resp = row["response"].replace("|", "\\|").replace("\n", " ")
            judge = "✅" if row["judge_correct"] is True else ("❌" if row["judge_correct"] is False else "N/A")
            lines.append(
                f"| `{row['case_id']}` | `{row['expected']}` | {judge} | {row['latency_ms']} | {q} | {resp} |"
            )
        lines.extend(
            [
                "",
                "> **Analysis**: If latency is short (< 10s) and a full response exists, VikingBot likely answered directly without invoking the memory search tool, despite the `--force-memory-search` instruction. If latency is high (> 100s), suspect a timeout.",
                "",
            ]
        )
    else:
        lines.append("No cases without route events.")

    # Template-hit but wrong answer analysis
    th_wrong = summary.get("template_hit_wrong_cases", [])
    lines.extend(
        [
            "",
            "## Template-Hit but Wrong Answer Analysis",
            "",
            f"Cases where template matched but answer was judged incorrect: **{len(th_wrong)}**",
            "",
        ]
    )
    if th_wrong:
        lines.extend(
            [
                "| Case | Expected | Actual | Matched Template | Question | Judge Reasoning |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in th_wrong:
            q = row["question"].replace("|", "\\|")
            reason = row["judge_reasoning"].replace("|", "\\|").replace("\n", " ")
            tmpl = row["matched_template_id"].replace("|", "\\|")
            lines.append(
                f"| `{row['case_id']}` | `{row['expected']}` | `{row['actual']}` | `{tmpl}` | {q} | {reason} |"
            )
        lines.extend(
            [
                "",
                "> **Analysis**: Routing was correct (template hit) but the final answer failed. Common causes: (1) OpenViking search returned incomplete results (recall issue / 4000-char truncation); (2) VikingBot LLM hallucinated or omitted key facts despite correct search results; (3) Judge applied overly strict matching criteria (e.g., time-range precision).",
                "",
            ]
        )
    else:
        lines.append("No template-hit cases with wrong answers.")

    lines.extend(
        [
            "",
            "## Wrong Cases",
            "",
        ]
    )
    if summary["wrong_cases"]:
        lines.extend(
            [
                "| Case | Expected | Actual | Method | Question |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in summary["wrong_cases"]:
            q = row["question"].replace("|", "\\|")
            lines.append(
                f"| `{row['case_id']}` | `{row['expected']}` | `{row['actual']}` | "
                f"`{row['route_method']}` | {q} |"
            )
    else:
        lines.append("No wrong cases.")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to LoCoMo JSON file (e.g., locomo10.json).",
    )
    parser.add_argument(
        "--route-labels",
        default="",
        help="Path to JSONL with expected_backend per case_id (optional but recommended).",
    )
    parser.add_argument(
        "--case-ids-file",
        default="",
        help=(
            "Optional file with case_ids to evaluate. Supports one case_id per "
            "line or JSONL records with a case_id field."
        ),
    )
    parser.add_argument(
        "--route-events-path",
        default="",
        help="Explicit path for route_events.jsonl. Must match the OpenViking server env var.",
    )
    parser.add_argument(
        "--ov-config",
        default=str(
            _default_openviking_root()
            / "benchmark"
            / "memrouter_ov_e2e"
            / "config"
            / "ov.conf"
        ),
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
        default="",
        help="Deprecated. Use --output-base instead.",
    )
    parser.add_argument(
        "--output-base",
        default=str(_default_openviking_root() / "benchmark" / "locomo_e2e" / "runs"),
        help=(
            "Base directory for evaluation artifacts. "
            "Creates {timestamp}_{dataset}_locomo_agent_e2e/logs + results inside."
        ),
    )
    parser.add_argument(
        "--fixed-run-dir",
        default="",
        help=(
            "Use this exact run directory instead of creating a timestamped child. "
            "Useful for wrapper scripts that start services before launching the evaluator."
        ),
    )
    parser.add_argument(
        "--ov-chat-endpoint",
        default="http://127.0.0.1:1933",
        help="OpenViking /bot/v1/chat endpoint base URL.",
    )
    parser.add_argument(
        "--ov-api-key", default="ov-test-key-12345", help="X-API-Key header."
    )
    parser.add_argument(
        "--ov-account", default="default", help="X-OpenViking-Account header."
    )
    parser.add_argument(
        "--ov-user",
        default="",
        help="X-OpenViking-User header. If empty, uses sample_id per question.",
    )
    parser.add_argument(
        "--ov-agent",
        default="",
        help="X-OpenViking-Agent header. If empty, uses sample_id per question.",
    )
    parser.add_argument(
        "--chat-timeout",
        type=int,
        default=120,
        help="Timeout for /bot/v1/chat calls.",
    )
    parser.add_argument(
        "--session-prefix",
        default="",
        help=(
            "Prefix for per-case VikingBot session_id. Defaults to the run_id. "
            "Each case still gets an isolated session."
        ),
    )
    parser.add_argument(
        "--force-memory-search",
        action="store_true",
        default=False,
        help=(
            "Ask VikingBot to call the memory search tool once before answering. "
            "Use this for MemRouter route E2E measurement."
        ),
    )
    parser.add_argument(
        "--limit-samples", type=int, default=0, help="Limit to first N samples."
    )
    parser.add_argument(
        "--limit-questions",
        type=int,
        default=0,
        help="Limit to first N questions per sample.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="",
        help="Filter to only questions with this category value (e.g. 1, 2, 3, 4). Default: all categories except 5.",
    )
    parser.add_argument(
        "--exclude-expected-backend",
        action="append",
        default=[],
        help="Exclude questions whose expected_backend matches this value. Can be used multiple times.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        default=False,
        help="Run LLM judge for answer correctness.",
    )
    parser.add_argument(
        "--judge-base-url",
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="Judge LLM base URL.",
    )
    parser.add_argument(
        "--judge-token", default="", help="Judge LLM API token."
    )
    parser.add_argument(
        "--judge-model", default="qwen3.5-flash", help="Judge LLM model name."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.judge and not args.judge_token:
        print("Error: --judge requires --judge-token", file=sys.stderr)
        return 1

    results, run_dir, logs_dir, results_dir = asyncio.run(_run_cases(args))
    summary = _build_summary(results)

    with (results_dir / "route_results.jsonl").open("w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _write_json(results_dir / "metrics_summary.json", summary)
    _write_csv(results_dir, results)
    _write_report(results_dir, summary)

    print(f"\nRun directory:     {run_dir}")
    print(f"Logs directory:    {logs_dir}")
    print(f"Results directory: {results_dir}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
