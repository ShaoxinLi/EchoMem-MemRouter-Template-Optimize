import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from echomem.embeddings.base import MockEmbeddingProvider
from echomem.matcher import BackendCandidate
from echomem.templates import BackendRouteTemplateIndex
from scripts.optimize_memrouter_template_yaml_gepa import (
    COMPONENT_NAME,
    BucketRandomBatchSampler,
    EpochStyleBatchSampler,
    GENERATION_PROMPT,
    GepaTraceCallback,
    JsonlLogger,
    LLMFeedbackGenerator,
    NO_DECISION,
    OpenAIChatLM,
    RouteCase,
    TemplateBundleMergingReflectionLM,
    TemplateYamlGepaAdapter,
    TemplateBundleValidator,
    ValidationResult,
    build_proposal_diff_summary,
    decide_backend,
    deterministic_feedback_for_case,
    merge_allowed_template_edits,
    normalize_llm_feedback,
    parse_template_bundle,
    read_template_bundle,
    resolve_max_metric_calls,
    serialize_template_bundle,
    summarize_outputs,
)


def _complete_editable_fields_bundle_from_parent(
    parent_bundle: str,
    overrides: dict[str, dict] | None = None,
) -> str:
    parent_files = parse_template_bundle(parent_bundle)
    assert isinstance(parent_files, dict)
    proposal_files = {}
    for filename, content in parent_files.items():
        data = yaml.safe_load(content)
        proposal_data = {
            "query_prototypes": data.get("query_prototypes", []),
            "hard_negatives": data.get("hard_negatives", []),
            "thresholds": data.get("thresholds", {}),
        }
        if overrides and filename in overrides:
            proposal_data.update(overrides[filename])
        proposal_files[filename] = yaml.safe_dump(proposal_data, sort_keys=False, allow_unicode=True)
    return serialize_template_bundle(proposal_files)


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


def test_bucket_random_sampler_refreshes_buckets() -> None:
    class FakeLoader:
        def __init__(self, ids):
            self.ids = ids

        def all_ids(self):
            return self.ids

        def __len__(self):
            return len(self.ids)

    sampler = BucketRandomBatchSampler(
        buckets={
            "graph_wrong": [1, 2],
            "random_pool": [1, 2],
        },
        minibatch_size=4,
        seed=13,
    )
    first_batch = sampler.next_minibatch_ids(FakeLoader([1, 2]), types.SimpleNamespace(i=0))

    sampler.refresh_buckets(
        {
            "no_decision_cases": [3],
            "random_pool": [3, 4],
        },
        iteration=3,
        candidate_idx=2,
    )
    refreshed_batch = sampler.next_minibatch_ids(FakeLoader([3, 4]), types.SimpleNamespace(i=1))

    assert sampler.refresh_count == 1
    assert sampler.last_refresh_iteration == 3
    assert sampler.last_refresh_candidate_idx == 2
    assert set(first_batch).issubset({1, 2})
    assert set(refreshed_batch).issubset({3, 4})
    assert sampler.summary()["bucket_sizes"]["no_decision_cases"] == 1


def test_epoch_style_sampler_covers_trainset_once_per_epoch() -> None:
    class FakeLoader:
        def all_ids(self):
            return [0, 1, 2, 3, 4]

        def __len__(self):
            return 5

    sampler = EpochStyleBatchSampler(minibatch_size=2, seed=13)
    sampled = []
    for iteration in range(3):
        sampled.extend(sampler.next_minibatch_ids(FakeLoader(), types.SimpleNamespace(i=iteration)))

    assert len(sampled) == 6
    assert set(sampled) == {0, 1, 2, 3, 4}
    assert sampler.summary()["batches_per_epoch"] == 3
    assert sampler.summary()["current_epoch"] == 0


def test_resolve_max_metric_calls_uses_epoch_stopper_when_unset() -> None:
    assert (
        resolve_max_metric_calls(
            types.SimpleNamespace(
                max_metric_calls=None,
                sampler="epoch",
                profile="full",
            )
        )
        is None
    )
    assert (
        resolve_max_metric_calls(
            types.SimpleNamespace(
                max_metric_calls=123,
                sampler="epoch",
                profile="full",
            )
        )
        == 123
    )


def test_resolve_max_metric_calls_keeps_bucket_defaults() -> None:
    assert (
        resolve_max_metric_calls(
            types.SimpleNamespace(
                max_metric_calls=None,
                sampler="bucket_random",
                profile="dry-run",
            )
        )
        == 96
    )
    assert (
        resolve_max_metric_calls(
            types.SimpleNamespace(
                max_metric_calls=None,
                sampler="bucket_random",
                profile="full",
            )
        )
        == 5000
    )


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


def test_merge_allowed_template_edits_preserves_immutable_fields(tmp_path: Path) -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    parent_bundle, files = read_template_bundle(template_dir)

    filename = "openviking.personal_fact_lookup.en.v2.yaml"
    parent_files = parse_template_bundle(parent_bundle)
    assert isinstance(parent_files, dict)
    parent_data = yaml.safe_load(parent_files[filename])
    proposal_data = {
        "query_prototypes": [
            *parent_data.get("query_prototypes", []),
            "Which personal attribute does PERSON_A have?",
        ],
        "hard_negatives": parent_data.get("hard_negatives", []),
        "thresholds": {
            **parent_data["thresholds"],
            "accept": 0.39,
        },
    }
    proposal_data["semantic_card"] = "Invalid non-whitelisted change."
    raw_proposal = _complete_editable_fields_bundle_from_parent(
        parent_bundle,
        {filename: proposal_data},
    )

    merged, diagnostics = merge_allowed_template_edits(parent_bundle, raw_proposal)
    merged_files = parse_template_bundle(merged)
    assert isinstance(merged_files, dict)
    merged_data = yaml.safe_load(merged_files[filename])

    assert diagnostics["status"] == "merged"
    assert len(merged_files) == len(files)
    assert filename not in diagnostics["missing_proposal_files"]
    assert not diagnostics["missing_proposal_files"]
    assert merged_data["query_spec"] == parent_data["query_spec"]
    assert merged_data["semantic_card"] == parent_data["semantic_card"]
    assert "Which personal attribute does PERSON_A have?" in merged_data["query_prototypes"]
    assert merged_data["thresholds"]["accept"] == 0.39

    embedder = MockEmbeddingProvider(dim=16)
    validator = TemplateBundleValidator(
        initial_bundle_text=parent_bundle,
        initial_files=files,
        tmp_root=tmp_path,
        query_embedder=embedder,
        template_embedder_factory=lambda _bundle_hash: embedder,
        smoke_cases=[],
    )
    result = validator.validate(merged)
    assert result.ok


def test_merge_allowed_template_edits_normalizes_file_markers_and_requires_complete_thresholds(
    tmp_path: Path,
) -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    parent_bundle, files = read_template_bundle(template_dir)
    parent_files = parse_template_bundle(parent_bundle)
    assert isinstance(parent_files, dict)

    filename = "openviking.personal_fact_lookup.en.v2.yaml"
    parent_data = yaml.safe_load(parent_files[filename])
    complete_thresholds = {**parent_data["thresholds"], "accept": 0.41}
    raw_proposal = _complete_editable_fields_bundle_from_parent(
        parent_bundle,
        {filename: {"thresholds": complete_thresholds}},
    ).replace(f"# FILE: {filename}", "# FILE: openviking.personal_fact_lookup.en.v2", 1)
    raw_proposal = "The following bundle updates editable fields.\n" + raw_proposal

    merged, diagnostics = merge_allowed_template_edits(parent_bundle, raw_proposal)
    merged_files = parse_template_bundle(merged)
    assert isinstance(merged_files, dict)
    merged_data = yaml.safe_load(merged_files[filename])

    assert diagnostics["status"] == "merged"
    assert diagnostics["complete_editable_fields_bundle_text_normalization"]["removed_prefix_before_file_marker"] is True
    assert diagnostics["complete_editable_fields_bundle_text_normalization"]["normalized_file_markers"] == len(parent_files)
    assert merged_data["thresholds"]["accept"] == 0.41
    assert merged_data["thresholds"]["fallback"] == parent_data["thresholds"]["fallback"]
    assert merged_data["query_spec"] == parent_data["query_spec"]

    incomplete_proposal = _complete_editable_fields_bundle_from_parent(
        parent_bundle,
        {filename: {"thresholds": {"accept": 0.41}}},
    )
    rejected, rejected_diagnostics = merge_allowed_template_edits(parent_bundle, incomplete_proposal)
    assert rejected_diagnostics["status"] == "skipped_output_contract_error"
    assert any(issue["type"] == "incomplete_thresholds" for issue in rejected_diagnostics["output_contract_issues"])
    assert isinstance(parse_template_bundle(rejected), dict)

    additions_only_proposal = _complete_editable_fields_bundle_from_parent(
        parent_bundle,
        {
            filename: {
                "query_prototypes": ["Which personal attribute does PERSON_A have?"],
                "hard_negatives": [
                    {
                        "query": "When did PERSON_A do ACTION_X?",
                        "confusing_with_backend": "temporal_memory_backend",
                        "reason": "Timestamp query, not personal fact lookup.",
                    }
                ],
            }
        },
    )
    _rejected, additions_only_diagnostics = merge_allowed_template_edits(parent_bundle, additions_only_proposal)
    assert additions_only_diagnostics["status"] == "skipped_output_contract_error"
    assert any(
        issue["type"] == "suspicious_incomplete_field_value"
        for issue in additions_only_diagnostics["output_contract_issues"]
    )

    embedder = MockEmbeddingProvider(dim=16)
    validator = TemplateBundleValidator(
        initial_bundle_text=parent_bundle,
        initial_files=files,
        tmp_root=tmp_path,
        query_embedder=embedder,
        template_embedder_factory=lambda _bundle_hash: embedder,
        smoke_cases=[],
    )
    result = validator.validate(rejected)
    assert not result.ok
    assert any(issue.type == "proposal_output_contract_error" for issue in result.issues)


def test_template_bundle_merging_reflection_lm_returns_merged_bundle() -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    parent_bundle, _files = read_template_bundle(template_dir)
    parent_files = parse_template_bundle(parent_bundle)
    assert isinstance(parent_files, dict)

    filename = "openviking.personal_fact_lookup.en.v2.yaml"
    parent_data = yaml.safe_load(parent_files[filename])
    proposal_data = {
        "query_prototypes": [
            *parent_data.get("query_prototypes", []),
            "Which personal preference does PERSON_A mention?",
        ],
        "hard_negatives": parent_data.get("hard_negatives", []),
        "thresholds": parent_data.get("thresholds", {}),
    }
    proposal_data["target"] = {"primary_backend_id": "graph_memory_backend"}
    raw_proposal = _complete_editable_fields_bundle_from_parent(
        parent_bundle,
        {filename: proposal_data},
    )

    wrapped_lm = TemplateBundleMergingReflectionLM(lambda _prompt: raw_proposal)
    prompt = (
        "CURRENT TEMPLATE BUNDLE START\n"
        f"{parent_bundle.rstrip()}\n"
        "CURRENT TEMPLATE BUNDLE END"
    )

    response = wrapped_lm(prompt)
    response_files = parse_template_bundle(response)
    assert isinstance(response_files, dict)
    response_data = yaml.safe_load(response_files[filename])

    assert len(response_files) == len(parent_files)
    assert response_data["query_spec"] == parent_data["query_spec"]
    assert response_data["target"] == parent_data["target"]
    assert "Which personal preference does PERSON_A mention?" in response_data["query_prototypes"]


def test_validator_allows_large_prototype_and_hard_negative_growth(tmp_path: Path) -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    bundle_text, files = read_template_bundle(template_dir)
    parsed = parse_template_bundle(bundle_text)
    assert isinstance(parsed, dict)

    filename = sorted(parsed)[0]
    data = yaml.safe_load(parsed[filename])
    data["query_prototypes"] = [
        *data.get("query_prototypes", []),
        *[f"Generalized synthetic coverage pattern {idx}" for idx in range(180)],
    ]
    data["hard_negatives"] = [
        *data.get("hard_negatives", []),
        *[
            {
                "query": f"Generalized confusing non-target pattern {idx}",
                "confusing_with_backend": "openviking_memory_backend",
                "reason": "Synthetic growth test.",
            }
            for idx in range(120)
        ],
    ]
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

    assert result.ok
    assert not any(issue.type.endswith("_count_limit") for issue in result.issues)


def test_validator_rejects_new_exact_minibatch_question_copy(tmp_path: Path) -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    bundle_text, files = read_template_bundle(template_dir)
    parsed = parse_template_bundle(bundle_text)
    assert isinstance(parsed, dict)

    exact_question = "What exact mini-batch question should not be copied?"
    filename = sorted(parsed)[0]
    data = yaml.safe_load(parsed[filename])
    data["query_prototypes"] = [*data.get("query_prototypes", []), exact_question]
    data["hard_negatives"] = [
        *data.get("hard_negatives", []),
        {
            "query": exact_question,
            "confusing_with_backend": "openviking_memory_backend",
            "reason": "Exact copy test.",
        },
    ]
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

    result = validator.validate(
        proposal,
        forbidden_exact_questions=[exact_question],
        reference_bundle_text=bundle_text,
    )

    assert not result.ok
    assert any(
        issue.type == "exact_minibatch_question_copy" and issue.field == "query_prototypes"
        for issue in result.issues
    )
    assert any(
        issue.type == "exact_minibatch_question_copy" and issue.field == "hard_negatives"
        for issue in result.issues
    )


def test_validator_requires_complete_hard_negative_fields(tmp_path: Path) -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    bundle_text, files = read_template_bundle(template_dir)
    parsed = parse_template_bundle(bundle_text)
    assert isinstance(parsed, dict)

    filename = sorted(parsed)[0]
    data = yaml.safe_load(parsed[filename])
    data["hard_negatives"] = [
        *data.get("hard_negatives", []),
        {
            "query": "Generalized confusing non-target pattern",
            "confusing_with_backend": "openviking_memory_backend",
        },
    ]
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
    assert any(issue.type == "invalid_hard_negative_fields" for issue in result.issues)


def test_validator_feedback_structures_schema_missing_field(tmp_path: Path) -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    bundle_text, files = read_template_bundle(template_dir)
    parsed = parse_template_bundle(bundle_text)
    assert isinstance(parsed, dict)

    filename = "openviking.previous_chat_recall.en.v1.yaml"
    data = yaml.safe_load(parsed[filename])
    missing_item_index = len(data.get("hard_negatives", []))
    data["hard_negatives"] = [
        *data.get("hard_negatives", []),
        {
            "query": "What goal does PERSON_A mention from an earlier conversation?",
            "reason": "Previous chat recall hard negative with a missing backend field.",
        },
    ]
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
    feedback = result.feedback()
    error = next(item for item in feedback["errors"] if item["type"] == "missing_required_field")

    assert not result.ok
    assert error["filename"] == filename
    assert error["field"] == "hard_negatives"
    assert error["path"] == f"hard_negatives.{missing_item_index}.confusing_with_backend"
    assert error["missing_field"] == "confusing_with_backend"
    assert error["item_index"] == missing_item_index
    assert error["details"]["pydantic_error_type"] == "missing"


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


def test_deterministic_feedback_lowers_threshold_when_expected_template_ranks_first() -> None:
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
            "score": 0.62,
        }
    )

    assert feedback["suggested_action_type"] == "lower_expected_accept_or_margin"
    assert feedback["action_recommendation"]["field"] == "thresholds"
    assert feedback["matched_template_accept"] == 0.56
    assert feedback["matched_template_margin"] == 0.04
    assert feedback["GEPA_case_score"] == 0.62


def test_deterministic_feedback_adds_expected_prototypes_when_expected_backend_ranks_low() -> None:
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

    assert feedback["suggested_action_type"] == "add_expected_backend_prototypes"
    assert feedback["action_recommendation"]["target_template_id"] == "graph.entity_relation.v1"
    assert "secondary_action_type" not in feedback["action_recommendation"]


def test_deterministic_feedback_handles_close_no_decision_as_discriminative() -> None:
    feedback = deterministic_feedback_for_case(
        {
            "case_id": "case-4",
            "question": "When did PERSON_A meet PERSON_B?",
            "expected_backend": "temporal_memory_backend",
            "predicted_backend": "",
            "is_correct": False,
            "is_no_decision": True,
            "expected_backend_rank": 2,
            "expected_backend_score": 0.59,
            "expected_backend_best_template_id": "temporal.timeline_fact.v1",
            "winning_backend": "graph_memory_backend",
            "winning_backend_score": 0.61,
            "margin_to_win": -0.02,
            "matched_template_id": "graph.entity_relation.v1",
        }
    )

    assert feedback["suggested_action_type"] == "add_discriminative_prototypes_or_hard_negatives"
    assert feedback["action_recommendation"]["target_template_id"] == "temporal.timeline_fact.v1"


def test_experiment_decide_backend_allows_multi_backend_acceptance() -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    index = BackendRouteTemplateIndex()
    index.load_from_directory(template_dir)
    template_by_id = {template.template_id: template for template in index.enabled_templates()}
    backend_candidates = [
        BackendCandidate(
            backend_id="openviking_memory_backend",
            best_template_id="openviking.personal_fact_lookup.en.v2",
            score=0.60,
        ),
        BackendCandidate(
            backend_id="graph_memory_backend",
            best_template_id="graph.entity_relation.v1",
            score=0.59,
        ),
    ]

    result = decide_backend([], backend_candidates, template_by_id)

    assert result["is_no_decision"] is False
    assert result["predicted_backend"] == "openviking_memory_backend"
    assert result["predicted_backends"] == ["openviking_memory_backend", "graph_memory_backend"]
    assert result["route_method"] == "template_embedding_multi_backend"
    assert result["matched_template_id"] == "openviking.personal_fact_lookup.en.v2"


def test_experiment_decide_backend_uses_explicit_no_decision_label() -> None:
    template_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    index = BackendRouteTemplateIndex()
    index.load_from_directory(template_dir)
    template_by_id = {template.template_id: template for template in index.enabled_templates()}
    backend_candidates = [
        BackendCandidate(
            backend_id="openviking_memory_backend",
            best_template_id="openviking.personal_fact_lookup.en.v2",
            score=0.10,
        )
    ]

    result = decide_backend([], backend_candidates, template_by_id)

    assert result["is_no_decision"] is True
    assert result["predicted_backend"] == NO_DECISION
    assert result["predicted_backends"] == []
    assert result["route_method"] == "no_decision"


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


def test_summarize_outputs_applies_regression_penalty() -> None:
    baseline_outputs = [
        {
            "case_id": "case-1",
            "expected_backend": "temporal_memory_backend",
            "is_correct": True,
            "is_no_decision": False,
            "expected_backend_rank": 1,
            "margin_to_win": 0.20,
        },
        {
            "case_id": "case-2",
            "expected_backend": "graph_memory_backend",
            "is_correct": False,
            "is_no_decision": False,
            "expected_backend_rank": 2,
            "margin_to_win": -0.05,
        },
    ]
    proposal_outputs = [
        {
            "case_id": "case-1",
            "expected_backend": "temporal_memory_backend",
            "is_correct": False,
            "is_no_decision": False,
            "expected_backend_rank": 2,
            "margin_to_win": -0.05,
            "matched_template_id": "graph.entity_relation.v1",
        },
        {
            "case_id": "case-2",
            "expected_backend": "graph_memory_backend",
            "is_correct": True,
            "is_no_decision": False,
            "expected_backend_rank": 1,
            "margin_to_win": 0.20,
            "matched_template_id": "graph.entity_relation.v1",
        },
    ]

    unpenalized = summarize_outputs(proposal_outputs)
    penalized = summarize_outputs(
        proposal_outputs,
        regression_baseline_outputs=baseline_outputs,
    )

    assert penalized["regression_count"] == 1
    assert penalized["regression_rate"] == 0.5
    assert penalized["regression_penalty"] == 0.1
    assert penalized["regression_case_ids"] == ["case-1"]
    assert penalized["gepa_score"] == round(unpenalized["gepa_score"] - 0.1, 6)


def test_adapter_scores_proposal_with_parent_regression_penalty(tmp_path: Path) -> None:
    class FakeEvaluator:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def evaluate(self, cases, bundle_text, **_kwargs):
            self.calls.append(bundle_text)
            outputs_by_bundle = {
                "parent": [
                    {
                        "case_id": "case-1",
                        "expected_backend": "temporal_memory_backend",
                        "is_correct": True,
                        "is_no_decision": False,
                        "expected_backend_rank": 1,
                        "margin_to_win": 0.20,
                        "score": 1.0,
                    },
                    {
                        "case_id": "case-2",
                        "expected_backend": "graph_memory_backend",
                        "is_correct": False,
                        "is_no_decision": False,
                        "expected_backend_rank": 2,
                        "margin_to_win": -0.05,
                        "score": 0.5,
                    },
                ],
                "proposal": [
                    {
                        "case_id": "case-1",
                        "expected_backend": "temporal_memory_backend",
                        "is_correct": False,
                        "is_no_decision": False,
                        "expected_backend_rank": 2,
                        "margin_to_win": -0.05,
                        "matched_template_id": "graph.entity_relation.v1",
                        "score": 0.5,
                    },
                    {
                        "case_id": "case-2",
                        "expected_backend": "graph_memory_backend",
                        "is_correct": True,
                        "is_no_decision": False,
                        "expected_backend_rank": 1,
                        "margin_to_win": 0.20,
                        "matched_template_id": "graph.entity_relation.v1",
                        "score": 1.0,
                    },
                ],
            }
            outputs = outputs_by_bundle[bundle_text]
            return outputs, [float(output["score"]) for output in outputs], ValidationResult(True, [])

    evaluator = FakeEvaluator()
    adapter = TemplateYamlGepaAdapter(
        evaluator=evaluator,
        llm_feedback=LLMFeedbackGenerator({"enabled": False}),
        cache_dir=tmp_path,
        candidate_eval_cache_enabled=False,
        initial_summary={},
        parent_val_summary={},
        trace_logger=JsonlLogger(tmp_path / "trace.jsonl"),
    )
    batch = [
        RouteCase("case-1", "When did EVENT_X happen?", "temporal_memory_backend"),
        RouteCase("case-2", "How are ENTITY_A and ENTITY_B connected?", "graph_memory_backend"),
    ]
    adapter.last_parent_bundle_text = "parent"
    adapter.last_feedback_record = {"batch_case_ids": ["case-1", "case-2"]}

    result = adapter.evaluate(batch, {COMPONENT_NAME: "proposal"}, capture_traces=False)

    expected_summary = summarize_outputs(
        evaluator.evaluate(batch, "proposal")[0],
        regression_baseline_outputs=evaluator.evaluate(batch, "parent")[0],
    )
    assert result.scores == [expected_summary["gepa_score"], expected_summary["gepa_score"]]
    assert expected_summary["regression_penalty"] == 0.1
    assert evaluator.calls[:2] == ["proposal", "parent"]


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
    assert set(logged["llm_payload"]) == {"deterministic_feedback"}
    deterministic_feedback = logged["deterministic_feedback"]
    assert "top_confusion_pairs" not in deterministic_feedback
    assert "suspected_overbroad_templates" not in deterministic_feedback
    assert "top_confusion_pairs" in deterministic_feedback["batch_summary"]
    assert "suspected_overbroad_templates" in deterministic_feedback["batch_summary"]
    assert logged["reflective_dataset_record"]["Feedback"]
    assert set(logged["reflective_dataset_record"]) == {"Feedback"}
    gepa_feedback = json.loads(logged["reflective_dataset_record"]["Feedback"])
    assert set(gepa_feedback) == {
        "validator_feedback",
        "deterministic_feedback",
        "llm_feedback",
        "parent_val_summary",
        "initial_summary",
    }
    assert gepa_feedback["parent_val_summary"] == {"gepa_score": 0.4}
    assert gepa_feedback["initial_summary"] == {"val": {"gepa_score": 0.5}}


def test_generation_prompt_consolidates_rules() -> None:
    assert "Allowed changes:" not in GENERATION_PROMPT
    assert "Forbidden changes:" not in GENERATION_PROMPT
    assert "Feedback records:" not in GENERATION_PROMPT
    assert "Use LLM Feedback as the main editing guidance" in GENERATION_PROMPT
    assert "do not follow it blindly" in GENERATION_PROMPT
    assert "Return the Complete Editable Fields Bundle only." in GENERATION_PROMPT
    assert "A Complete Editable Fields Bundle contains the complete revised editable fields" in GENERATION_PROMPT
    assert "Do not omit unchanged files." in GENERATION_PROMPT
    assert "Do not output only additions, removals, or partial field values." in GENERATION_PROMPT
    assert "Return the complete Template Bundle only." not in GENERATION_PROMPT
    assert "Every hard_negatives item must include:" in GENERATION_PROMPT
    assert "confusing_with_backend" in GENERATION_PROMPT


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
                    "priority": "high",
                    "guidance": "Separate event-time lookup from relation lookup.",
                    "evidence": "repeated temporal relation confusion",
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
    assert {item["priority"] for item in normalized["template_suggestions"]} == {"high"}
    assert {item["evidence"] for item in normalized["template_suggestions"]} == {
        "repeated temporal relation confusion"
    }
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
