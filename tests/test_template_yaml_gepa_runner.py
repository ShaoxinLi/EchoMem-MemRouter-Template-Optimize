import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from echomem.embeddings.base import MockEmbeddingProvider
from scripts.optimize_memrouter_template_yaml_gepa import (
    COMPONENT_NAME,
    GepaTraceCallback,
    JsonlLogger,
    LLMFeedbackGenerator,
    OpenAIChatLM,
    RouteCase,
    TemplateYamlGepaAdapter,
    TemplateBundleValidator,
    build_proposal_diff_summary,
    deterministic_feedback_for_case,
    normalize_llm_feedback,
    parse_template_bundle,
    read_template_bundle,
    serialize_template_bundle,
)


def test_initial_template_bundle_passes_validator(tmp_path: Path) -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    bundle_text, files = read_template_bundle(template_dir)
    embedder = MockEmbeddingProvider(dim=16)
    validator = TemplateBundleValidator(
        initial_bundle_text=bundle_text,
        initial_files=files,
        tmp_root=tmp_path,
        query_embedder=embedder,
        template_embedder_factory=lambda _bundle_hash: embedder,
        smoke_cases=[
            RouteCase(
                case_id="smoke",
                question="When did PERSON_A visit PLACE_X?",
                expected_backend="temporal_memory_backend",
            )
        ],
    )

    result = validator.validate(bundle_text)

    assert result.ok
    assert result.materialized_dir is not None


def test_validator_rejects_changed_target(tmp_path: Path) -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    bundle_text, files = read_template_bundle(template_dir)
    parsed = parse_template_bundle(bundle_text)
    assert isinstance(parsed, dict)

    filename = sorted(parsed)[0]
    data = yaml.safe_load(parsed[filename])
    data["target"]["primary_backend_id"] = "temporal_memory_backend"
    parsed[filename] = yaml.safe_dump(data, sort_keys=False, allow_unicode=True)
    proposal = serialize_template_bundle(parsed)

    embedder = MockEmbeddingProvider(dim=16)
    validator = TemplateBundleValidator(
        initial_bundle_text=bundle_text,
        initial_files=files,
        tmp_root=tmp_path,
        query_embedder=embedder,
        template_embedder_factory=lambda _bundle_hash: embedder,
        smoke_cases=[],
    )

    result = validator.validate(proposal)

    assert not result.ok
    assert any(issue.type == "immutable_field_changed" and issue.field == "target" for issue in result.issues)


def test_openai_chat_lm_rejects_truncated_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyCompletions:
        def create(self, **_kwargs):
            return types.SimpleNamespace(
                choices=[
                    types.SimpleNamespace(
                        finish_reason="length",
                        message=types.SimpleNamespace(content="partial output"),
                    )
                ]
            )

    class DummyOpenAI:
        def __init__(self, **_kwargs) -> None:
            self.chat = types.SimpleNamespace(completions=DummyCompletions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=DummyOpenAI))

    lm = OpenAIChatLM({"model": "test-model", "api_key": "test-key"}, "proposal_lm")

    with pytest.raises(RuntimeError, match="truncated"):
        lm("prompt")


def test_deterministic_feedback_preserves_correct_case() -> None:
    feedback = deterministic_feedback_for_case(
        {
            "case_id": "case-1",
            "question": "What does Alex like?",
            "expected_backend": "openviking_memory_backend",
            "predicted_backend": "openviking_memory_backend",
            "is_correct": True,
            "is_no_decision": False,
            "expected_backend_rank": 1,
            "expected_backend_score": 0.71,
            "winning_backend": "openviking_memory_backend",
            "winning_backend_score": 0.71,
            "margin_to_win": 0.2,
            "matched_template_id": "openviking.personal_fact_lookup.en.v2",
        }
    )

    assert feedback["suggested_action_type"] == "preserve_behavior"
    assert feedback["action_recommendation"]["direction"] == "preserve"


def test_deterministic_feedback_relaxes_hard_negative_when_expected_template_is_penalized() -> None:
    feedback = deterministic_feedback_for_case(
        {
            "case_id": "case-2",
            "question": "When did Alex move?",
            "expected_backend": "temporal_memory_backend",
            "predicted_backend": "",
            "is_correct": False,
            "is_no_decision": True,
            "expected_backend_rank": 1,
            "expected_backend_score": 0.54,
            "expected_backend_best_template_id": "temporal.timeline_fact.v1",
            "expected_template_score_components": {"penalty": 0.035},
            "winning_backend": "temporal_memory_backend",
            "winning_backend_score": 0.54,
            "margin_to_win": 0.08,
            "matched_template_id": "temporal.timeline_fact.v1",
            "matched_template_accept": 0.56,
            "matched_template_margin": 0.04,
        }
    )

    assert feedback["suggested_action_type"] == "relax_expected_hard_negative_penalty"
    assert feedback["action_recommendation"]["field"] == "hard_negatives/thresholds"


def test_deterministic_feedback_narrows_overbroad_prototypes_for_confident_wrong_route() -> None:
    feedback = deterministic_feedback_for_case(
        {
            "case_id": "case-3",
            "question": "Which movies did Alex and Taylor both watch?",
            "expected_backend": "graph_memory_backend",
            "predicted_backend": "openviking_memory_backend",
            "is_correct": False,
            "is_no_decision": False,
            "expected_backend_rank": 3,
            "expected_backend_score": 0.41,
            "expected_backend_best_template_id": "graph.entity_relation.v1",
            "winning_backend": "openviking_memory_backend",
            "winning_backend_score": 0.66,
            "margin_to_win": -0.21,
            "matched_template_id": "openviking.personal_fact_lookup.en.v2",
        }
    )

    assert feedback["suggested_action_type"] == "narrow_or_remove_overbroad_prototypes"
    assert feedback["action_recommendation"]["target_template_id"] == "openviking.personal_fact_lookup.en.v2"
    assert feedback["action_recommendation"]["secondary_action_type"] == "add_expected_backend_prototypes"


def test_cached_invalid_proposal_restores_validator_feedback(tmp_path: Path) -> None:
    adapter = TemplateYamlGepaAdapter(
        evaluator=None,
        llm_feedback=LLMFeedbackGenerator({"enabled": False}),
        cache_dir=tmp_path,
        candidate_eval_cache_enabled=True,
        initial_summary={},
        parent_val_summary={},
        trace_logger=JsonlLogger(tmp_path / "trace.jsonl"),
    )
    batch = [
        RouteCase(
            case_id="case-1",
            question="When did the meeting happen?",
            expected_backend="temporal_memory_backend",
        )
    ]
    bundle_text = "# FILE: invalid.yaml\nnot: a valid bundle\n"
    feedback = {
        "validator_status": "failed",
        "errors": [{"type": "template_file_set_changed", "message": "missing files"}],
        "required_fix": ["Return complete YAML only."],
    }
    cache_path = adapter._cache_path(batch, bundle_text)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "outputs": [
                    {
                        "case_id": "case-1",
                        "route_method": "invalid_proposal",
                        "validator_feedback": feedback,
                        "score": 0.0,
                    }
                ],
                "scores": [0.0],
                "objective_scores": [{"backend_correct": 0.0}],
            }
        ),
        encoding="utf-8",
    )

    result = adapter.evaluate(batch, {COMPONENT_NAME: bundle_text}, capture_traces=False)

    assert result.scores == [0.0]
    assert adapter.last_validator_feedback == feedback
    assert adapter.consecutive_invalid_proposals == 1
    assert adapter.validator_failure_count == 1
    assert "validator_failed_cache_hit" in (tmp_path / "trace.jsonl").read_text(encoding="utf-8")


def test_make_reflective_dataset_writes_feedback_record(tmp_path: Path) -> None:
    feedback_log = tmp_path / "feedback_records.jsonl"
    adapter = TemplateYamlGepaAdapter(
        evaluator=None,
        llm_feedback=LLMFeedbackGenerator({"enabled": False}),
        cache_dir=tmp_path,
        candidate_eval_cache_enabled=False,
        initial_summary={"val": {"gepa_score": 0.5}},
        parent_val_summary={"gepa_score": 0.4},
        trace_logger=JsonlLogger(tmp_path / "trace.jsonl"),
        feedback_logger=JsonlLogger(feedback_log),
    )
    adapter.update_debug_context(iteration=3, parent_candidate_idx=2, minibatch_ids=[7])
    output = {
        "case_id": "case-1",
        "question": "When did the meeting happen?",
        "expected_backend": "temporal_memory_backend",
        "predicted_backend": "",
        "is_correct": False,
        "is_no_decision": True,
        "expected_backend_rank": 1,
        "expected_backend_score": 0.71,
        "winning_backend": "temporal_memory_backend",
        "winning_backend_score": 0.71,
        "margin_to_win": 0.12,
        "matched_template_id": "temporal.timeline_fact.v1",
        "top_backend_ranking": [],
        "top_template_ranking": [],
        "score": 0.0,
    }
    eval_batch = types.SimpleNamespace(
        trajectories=[
            {
                "output": output,
                "deterministic_feedback": {
                    "case_id": "case-1",
                    "diagnosis": "expected backend ranked first but was not accepted",
                },
            }
        ]
    )

    records = adapter.make_reflective_dataset(
        {COMPONENT_NAME: "# FILE: temporal.timeline_fact.v1.yaml\nquery_prototypes: []\n"},
        eval_batch,
        [COMPONENT_NAME],
    )

    assert COMPONENT_NAME in records
    logged = json.loads(feedback_log.read_text(encoding="utf-8").splitlines()[0])
    assert logged["event"] == "feedback_record"
    assert logged["iteration"] == 3
    assert logged["parent_candidate_idx"] == 2
    assert logged["batch_case_ids"] == ["case-1"]
    assert logged["llm_feedback"]["status"] == "disabled"
    assert logged["reflective_dataset_record"]["Feedback"]


def test_normalize_llm_feedback_uses_stable_fields_and_actions() -> None:
    normalized, diagnostics = normalize_llm_feedback(
        {
            "feedback_type": "llm_feedback",
            "summary": "Prefer general edits.",
            "template_suggestions": [
                {
                    "template_id": "temporal.timeline_fact.v1",
                    "field": "positive prototypes and hard negatives",
                    "action": "tighten discrimination",
                    "guidance": "Separate event-time lookup from relation lookup.",
                },
                {
                    "template_id": "temporal.timeline_fact.v1",
                    "field": "metadata",
                    "action": "change",
                    "guidance": "Not editable.",
                },
            ],
            "preserve": "not a list",
        }
    )

    assert [item["field"] for item in normalized["template_suggestions"]] == [
        "query_prototypes",
        "hard_negatives",
    ]
    assert {item["action"] for item in normalized["template_suggestions"]} == {"increase"}
    assert normalized["preserve"] == []
    assert diagnostics["field_normalizations"]
    assert diagnostics["action_normalizations"]
    assert diagnostics["dropped_suggestions"][0]["reason"] == "unknown_field"


def test_build_proposal_diff_summary_records_changes_and_action_alignment() -> None:
    parent = """# FILE: temporal.timeline_fact.v1.yaml
template_id: temporal.timeline_fact.v1
query_prototypes:
  - What happened during DATE?
hard_negatives:
  - query: Who attended EVENT_X?
    confusing_with_backend: graph_memory_backend
    reason: Relation lookup, not temporal lookup.
thresholds:
  accept: 0.56
  hard_negative_penalty: 0.06
"""
    proposal = """# FILE: temporal.timeline_fact.v1.yaml
template_id: temporal.timeline_fact.v1
query_prototypes:
  - What happened during DATE?
  - When did the meeting happen?
hard_negatives: []
thresholds:
  accept: 0.52
  hard_negative_penalty: 0.08
"""
    feedback_record = {
        "matcher_outputs": [{"question": "When did the meeting happen?"}],
        "deterministic_feedback": {
            "batch_action_recommendations": [
                {
                    "action_type": "add_discriminative_prototypes_or_hard_negatives",
                    "target_template_id": "temporal.timeline_fact.v1",
                },
                {
                    "action_type": "lower_expected_accept_or_margin",
                    "target_template_id": "temporal.timeline_fact.v1",
                },
            ]
        },
        "llm_feedback_debug": {
            "normalization": {
                "field_normalizations": [{"original": "prototype", "normalized": ["query_prototypes"]}]
            }
        },
    }

    summary = build_proposal_diff_summary(parent, proposal, feedback_record)

    assert summary["status"] == "ok"
    assert summary["summary"]["added_query_prototypes"] == 1
    assert summary["summary"]["removed_hard_negatives"] == 1
    assert summary["summary"]["changed_thresholds"] == 2
    assert "add_expected_backend_prototypes" in summary["executed_action_types"]
    assert "lower_expected_accept_or_margin" in summary["executed_action_types"]
    assert summary["action_alignment"]["matched_primary_action_types"] == [
        "add_discriminative_prototypes_or_hard_negatives",
        "lower_expected_accept_or_margin",
    ]
    assert summary["exact_question_copies"][0]["field"] == "query_prototypes"
    assert summary["llm_feedback_normalization"]["field_normalizations"]


def test_gepa_trace_callback_writes_proposal_diff_summary(tmp_path: Path) -> None:
    parent = """# FILE: temporal.timeline_fact.v1.yaml
template_id: temporal.timeline_fact.v1
query_prototypes:
  - What happened during DATE?
hard_negatives: []
thresholds:
  accept: 0.56
"""
    proposal = """# FILE: temporal.timeline_fact.v1.yaml
template_id: temporal.timeline_fact.v1
query_prototypes:
  - What happened during DATE?
  - When was the appointment?
hard_negatives: []
thresholds:
  accept: 0.56
"""
    adapter = TemplateYamlGepaAdapter(
        evaluator=None,
        llm_feedback=LLMFeedbackGenerator({"enabled": False}),
        cache_dir=tmp_path,
        candidate_eval_cache_enabled=False,
        initial_summary={},
        parent_val_summary={},
        trace_logger=JsonlLogger(tmp_path / "trace.jsonl"),
    )
    adapter.last_parent_bundle_text = parent
    adapter.last_feedback_record = {
        "matcher_outputs": [{"question": "When was the appointment?"}],
        "deterministic_feedback": {
            "batch_action_recommendations": [
                {"action_type": "add_expected_backend_prototypes"}
            ]
        },
    }
    callback = GepaTraceCallback(
        trace_logger=JsonlLogger(tmp_path / "optimization_trace.jsonl"),
        candidate_csv=tmp_path / "candidate_scores.csv",
        proposal_dir=tmp_path / "proposals",
        adapter=adapter,
        initial_val_summary={},
    )

    callback.on_proposal_end(
        {
            "iteration": 2,
            "new_instructions": {COMPONENT_NAME: proposal},
        }
    )

    diff_path = tmp_path / "proposal_diffs" / "iter_0002_template_bundle.diff_summary.json"
    assert diff_path.exists()
    diff_summary = json.loads(diff_path.read_text(encoding="utf-8"))
    assert diff_summary["summary"]["added_query_prototypes"] == 1
    trace_row = json.loads((tmp_path / "optimization_trace.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    proposal_record = trace_row["proposals"][0]
    assert proposal_record["diff_summary_path"] == "reports/proposal_diffs/iter_0002_template_bundle.diff_summary.json"
    assert proposal_record["diff_summary"]["action_alignment"]["matched_primary_action_types"] == [
        "add_expected_backend_prototypes"
    ]
