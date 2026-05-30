#!/usr/bin/env python
"""Optimize MemRouter YAML templates with GEPA in matcher-only mode."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from echomem.embeddings.base import EmbeddingProvider, create_provider
from echomem.features import QueryFeatureBuilder
from echomem.matcher import BackendCandidate, TemplateCandidate, TemplateMatcher
from echomem.templates import BackendRouteTemplateIndex, MemoryBackendRouteTemplate

COMPONENT_NAME = "template_bundle"
NO_DECISION = "No Decision"
ALLOWED_TOP_LEVEL_FIELDS = {"query_prototypes", "hard_negatives", "thresholds"}
LLM_FEEDBACK_ALLOWED_FIELDS = ("query_prototypes", "hard_negatives", "thresholds")
LLM_FEEDBACK_ALLOWED_ACTIONS = ("add", "rewrite", "remove", "increase", "decrease", "preserve")
DETERMINISTIC_ACTION_TYPE_DEFINITIONS = {
    "preserve_behavior": "Keep the currently correct route as a regression anchor.",
    "add_expected_backend_prototypes": "Add generalized query_prototypes to improve weak expected backend coverage.",
    "add_wrong_template_hard_negatives": "Add hard_negatives to the wrong winning template to reduce false positives.",
    "add_discriminative_prototypes_or_hard_negatives": "Improve a close decision by adding expected prototypes or wrong-template hard negatives.",
    "lower_expected_accept_or_margin": "Slightly lower accept or margin for the expected template when it ranks first but is not accepted.",
}


@dataclass(frozen=True)
class RouteCase:
    case_id: str
    question: str
    expected_backend: str
    sample_id: str = ""
    scenario: str = ""
    category: str = ""


@dataclass
class ValidationIssue:
    type: str
    message: str
    template_id: str | None = None
    field: str | None = None
    filename: str | None = None


@dataclass
class ValidationResult:
    ok: bool
    issues: list[ValidationIssue]
    materialized_dir: Path | None = None
    bundle_hash: str | None = None

    def feedback(self) -> dict[str, Any]:
        return {
            "validator_status": "passed" if self.ok else "failed",
            "errors": [asdict(issue) for issue in self.issues],
            "required_fix": [
                "Return complete YAML only, with one '# FILE: <name>.yaml' marker before each file.",
                "Keep template_id, target, query_spec, and all non-whitelisted fields unchanged.",
                "Only modify query_prototypes, hard_negatives, and thresholds.",
            ],
        }


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, row: dict[str, Any]) -> None:
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")


class FileOnlyLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def log(self, message: str) -> None:
        text = str(message)
        if "Proposed new text for" in text and len(text) > 2000:
            text = text[:2000] + "\n[truncated; full candidate is saved under candidates/ after acceptance]\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text.rstrip() + "\n")


class DiskEmbeddingCacheProvider(EmbeddingProvider):
    """Small disk cache wrapper for deterministic question/template embeddings."""

    def __init__(
        self,
        base: EmbeddingProvider,
        cache_dir: Path,
        namespace: str,
        enabled: bool = True,
        stats: dict[str, int] | None = None,
    ) -> None:
        self._base = base
        self._cache_dir = cache_dir
        self._namespace = namespace
        self._enabled = enabled
        self._stats = stats if stats is not None else {"hits": 0, "misses": 0}
        self._lock = Lock()
        if self._enabled:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension()), dtype=np.float32)
        if not self._enabled:
            return self._base.embed(texts)

        results: list[np.ndarray | None] = []
        misses: list[str] = []
        miss_positions: list[int] = []

        for idx, text in enumerate(texts):
            path = self._path_for(text)
            if path.exists():
                try:
                    with path.open("rb") as f:
                        results.append(np.load(f).astype(np.float32))
                    with self._lock:
                        self._stats["hits"] = self._stats.get("hits", 0) + 1
                    continue
                except Exception:
                    path.unlink(missing_ok=True)
            results.append(None)
            misses.append(text)
            miss_positions.append(idx)
            with self._lock:
                self._stats["misses"] = self._stats.get("misses", 0) + 1

        if misses:
            embedded = self._base.embed(misses)
            for text, pos, vec in zip(misses, miss_positions, embedded, strict=False):
                arr = np.asarray(vec, dtype=np.float32)
                results[pos] = arr
                path = self._path_for(text)
                tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
                try:
                    with tmp.open("wb") as f:
                        np.save(f, arr)
                    tmp.replace(path)
                except Exception:
                    tmp.unlink(missing_ok=True)

        return np.vstack([arr for arr in results if arr is not None]).astype(np.float32)

    def dimension(self) -> int:
        return self._base.dimension()

    def _path_for(self, text: str) -> Path:
        key = _sha256_text(json.dumps({"namespace": self._namespace, "text": text}, sort_keys=True))
        return self._cache_dir / f"{key}.npy"


class OpenAIChatLM:
    """OpenAI-compatible chat completion callable used by GEPA and LLM Feedback."""

    def __init__(
        self,
        config: dict[str, Any],
        role: str,
        debug_logger: JsonlLogger | None = None,
    ) -> None:
        self.config = config
        self.role = role
        self.debug_logger = debug_logger
        self._call_count = 0
        self.model = str(config.get("model") or "")
        self.temperature = float(config.get("temperature", 0.0))
        self.max_tokens = config.get("max_tokens")
        self.reasoning_effort = config.get("reasoning_effort")
        self.extra_body = config.get("extra_body")
        self.system_prompt = config.get("system_prompt")
        if not self.model:
            raise ValueError(f"{role}.model is required")
        try:
            import openai
        except ImportError as exc:
            raise ImportError("openai is required for OpenAI-compatible chat LMs") from exc

        api_key = config.get("api_key")
        if isinstance(api_key, str) and _is_unresolved_secret(api_key):
            raise ValueError(f"{role}.api_key is unresolved or still contains a placeholder.")
        base_url = config.get("base_url")
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

        if self.model.startswith("deepseek-v4"):
            if not self.reasoning_effort:
                self.reasoning_effort = "high"
            if self.extra_body is None:
                self.extra_body = {"thinking": {"type": "enabled"}}
        if self.extra_body is not None and not isinstance(self.extra_body, dict):
            raise ValueError(f"{role}.extra_body must be a mapping when provided.")

    def __call__(self, prompt: str) -> str:
        self._call_count += 1
        request_id = f"{self.role}_{self._call_count:04d}_{_sha256_text(prompt)[:12]}"
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": str(self.system_prompt)})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            kwargs["max_tokens"] = int(self.max_tokens)
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = str(self.reasoning_effort)
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        self._log_lm_debug(
            {
                "event": "lm_request",
                "request_id": request_id,
                "role": self.role,
                "model": self.model,
                "temperature": self.temperature,
                "max_tokens": int(self.max_tokens) if self.max_tokens else None,
                "reasoning_effort": self.reasoning_effort,
                "has_system_prompt": bool(self.system_prompt),
                "prompt_hash": _sha256_text(prompt),
                "prompt_chars": len(prompt),
                "prompt": prompt,
            }
        )

        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            self._log_lm_debug(
                {
                    "event": "lm_error",
                    "request_id": request_id,
                    "role": self.role,
                    "model": self.model,
                    "error": str(exc),
                }
            )
            raise

        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        content = choice.message.content or ""
        self._log_lm_debug(
            {
                "event": "lm_response",
                "request_id": request_id,
                "role": self.role,
                "model": self.model,
                "finish_reason": finish_reason,
                "response_hash": _sha256_text(content),
                "response_chars": len(content),
                "response": content,
            }
        )
        if finish_reason == "length":
            raise RuntimeError(
                f"{self.role} completion was truncated with finish_reason=length. "
                "Increase max_tokens or reduce the prompt/output size."
            )
        return content

    def _log_lm_debug(self, row: dict[str, Any]) -> None:
        if self.debug_logger is not None:
            self.debug_logger.write(row)


class NoopReflectionLM:
    """Dry-run LM that returns the current template bundle unchanged."""

    def __call__(self, prompt: str) -> str:
        match = re.search(
            r"CURRENT TEMPLATE BUNDLE START\n(?P<body>.*?)\nCURRENT TEMPLATE BUNDLE END",
            prompt,
            flags=re.DOTALL,
        )
        if match:
            return match.group("body").strip()
        return ""


def normalize_llm_feedback(feedback: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    normalized = dict(feedback)
    diagnostics: dict[str, Any] = {
        "field_normalizations": [],
        "action_normalizations": [],
        "dropped_suggestions": [],
    }
    raw_suggestions = feedback.get("template_suggestions", [])
    if not isinstance(raw_suggestions, list):
        diagnostics["dropped_suggestions"].append(
            {
                "reason": "template_suggestions_not_list",
                "value_type": type(raw_suggestions).__name__,
            }
        )
        raw_suggestions = []

    template_suggestions: list[dict[str, Any]] = []
    for idx, suggestion in enumerate(raw_suggestions):
        if not isinstance(suggestion, dict):
            diagnostics["dropped_suggestions"].append(
                {
                    "index": idx,
                    "reason": "suggestion_not_mapping",
                    "value_type": type(suggestion).__name__,
                }
            )
            continue

        fields = _normalize_llm_feedback_fields(suggestion.get("field"))
        if not fields:
            diagnostics["dropped_suggestions"].append(
                {
                    "index": idx,
                    "reason": "unknown_field",
                    "field": suggestion.get("field"),
                }
            )
            continue
        if list(fields) != [suggestion.get("field")]:
            diagnostics["field_normalizations"].append(
                {
                    "index": idx,
                    "original": suggestion.get("field"),
                    "normalized": fields,
                }
            )

        for field in fields:
            normalized_suggestion = dict(suggestion)
            normalized_suggestion["field"] = field
            normalized_action = _normalize_llm_feedback_action(suggestion.get("action"), field)
            if normalized_action != suggestion.get("action"):
                diagnostics["action_normalizations"].append(
                    {
                        "index": idx,
                        "field": field,
                        "original": suggestion.get("action"),
                        "normalized": normalized_action,
                    }
                )
            normalized_suggestion["action"] = normalized_action
            template_suggestions.append(normalized_suggestion)

    normalized["template_suggestions"] = template_suggestions
    preserve = feedback.get("preserve", [])
    normalized["preserve"] = preserve if isinstance(preserve, list) else []
    normalized.setdefault("feedback_type", "llm_feedback")
    normalized.setdefault("summary", "")
    return normalized, diagnostics


def _normalize_llm_feedback_fields(field: Any) -> list[str]:
    text = str(field or "").strip().lower()
    if not text:
        return []
    normalized: list[str] = []
    if text in LLM_FEEDBACK_ALLOWED_FIELDS:
        return [text]
    if any(token in text for token in ("prototype", "positive", "example", "coverage")):
        normalized.append("query_prototypes")
    if any(token in text for token in ("hard_negative", "hard negative", "negative", "confusing")):
        normalized.append("hard_negatives")
    if any(token in text for token in ("threshold", "accept", "margin", "fallback", "penalty")):
        normalized.append("thresholds")
    return [field for field in LLM_FEEDBACK_ALLOWED_FIELDS if field in set(normalized)]


def _normalize_llm_feedback_action(action: Any, field: str) -> str:
    text = str(action or "").strip().lower()
    if text in LLM_FEEDBACK_ALLOWED_ACTIONS:
        return text
    if any(token in text for token in ("preserve", "keep", "retain")):
        return "preserve"
    if any(token in text for token in ("add", "append", "create", "expand", "cover")):
        return "add"
    if any(token in text for token in ("remove", "delete", "drop", "prune")):
        return "remove"
    if any(token in text for token in ("increase", "raise", "tighten")):
        return "increase"
    if any(token in text for token in ("decrease", "lower", "relax")):
        return "decrease"
    if any(token in text for token in ("rewrite", "revise", "narrow", "generalize", "refine", "adjust")):
        return "rewrite"
    return "rewrite"


class LLMFeedbackGenerator:
    def __init__(
        self,
        config: dict[str, Any] | None,
        debug_logger: JsonlLogger | None = None,
    ) -> None:
        self.config = config or {}
        self.enabled = bool(self.config.get("enabled", False))
        self.last_debug: dict[str, Any] = {}
        self._lm: OpenAIChatLM | None = None
        if self.enabled:
            self._lm = OpenAIChatLM(self.config, "llm_feedback", debug_logger=debug_logger)

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled or self._lm is None:
            self.last_debug = {
                "enabled": False,
                "status": "disabled",
            }
            return {
                "feedback_type": "llm_feedback",
                "status": "disabled",
                "summary": "LLM Feedback is disabled for this run.",
                "template_suggestions": [],
                "preserve": [],
            }

        prompt = LLM_FEEDBACK_PROMPT.replace(
            "<DETERMINISTIC_FEEDBACK_JSON>",
            json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        )
        raw = ""
        try:
            raw = self._lm(prompt).strip()
            parsed = _extract_json_object(raw)
            parsed, normalization_diagnostics = normalize_llm_feedback(parsed)
            self.last_debug = {
                "enabled": True,
                "status": "parsed",
                "prompt_hash": _sha256_text(prompt),
                "prompt_chars": len(prompt),
                "raw_response_hash": _sha256_text(raw),
                "raw_response_chars": len(raw),
                "raw_response": raw,
                "normalization": normalization_diagnostics,
            }
            return parsed
        except Exception as exc:
            self.last_debug = {
                "enabled": True,
                "status": "failed",
                "prompt_hash": _sha256_text(prompt),
                "prompt_chars": len(prompt),
                "raw_response_hash": _sha256_text(raw),
                "raw_response_chars": len(raw),
                "raw_response": raw,
                "error": str(exc),
            }
            return {
                "feedback_type": "llm_feedback",
                "status": "failed",
                "summary": f"LLM Feedback failed: {exc}",
                "template_suggestions": [],
                "preserve": [],
            }


class TemplateBundleValidator:
    def __init__(
        self,
        initial_bundle_text: str,
        initial_files: dict[str, str],
        tmp_root: Path,
        query_embedder: EmbeddingProvider,
        template_embedder_factory: Any,
        smoke_cases: list[RouteCase],
    ) -> None:
        self.initial_bundle_text = initial_bundle_text
        self.initial_files = initial_files
        self.expected_filenames = sorted(initial_files)
        self.tmp_root = tmp_root
        self.query_embedder = query_embedder
        self.template_embedder_factory = template_embedder_factory
        self.smoke_cases = smoke_cases[:3]

        self.initial_yaml: dict[str, dict[str, Any]] = {}
        self.initial_templates_by_file: dict[str, MemoryBackendRouteTemplate] = {}
        for filename, content in initial_files.items():
            data = yaml.safe_load(content)
            template = MemoryBackendRouteTemplate.model_validate(data)
            self.initial_yaml[filename] = data
            self.initial_templates_by_file[filename] = template

    def validate(self, bundle_text: str) -> ValidationResult:
        bundle_hash = _sha256_text(bundle_text)
        issues: list[ValidationIssue] = []
        parsed_files = parse_template_bundle(bundle_text)
        if isinstance(parsed_files, list):
            return ValidationResult(False, parsed_files, bundle_hash=bundle_hash)

        filenames = sorted(parsed_files)
        if filenames != self.expected_filenames:
            issues.append(
                ValidationIssue(
                    type="template_file_set_changed",
                    message=(
                        f"Expected files {self.expected_filenames}, got {filenames}."
                    ),
                )
            )
            return ValidationResult(False, issues, bundle_hash=bundle_hash)

        total_initial_chars = sum(len(v) for v in self.initial_files.values())
        total_candidate_chars = sum(len(v) for v in parsed_files.values())
        if total_candidate_chars > max(total_initial_chars * 1.6, total_initial_chars + 8000):
            issues.append(
                ValidationIssue(
                    type="bundle_growth_limit_exceeded",
                    message="Proposal text is too large relative to Parent.",
                )
            )

        materialized = self.tmp_root / "validated_bundles" / bundle_hash
        if materialized.exists():
            shutil.rmtree(materialized)
        materialized.mkdir(parents=True, exist_ok=True)

        for filename in self.expected_filenames:
            content = parsed_files[filename]
            try:
                data = yaml.safe_load(content)
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        type="yaml_parse_error",
                        message=str(exc),
                        filename=filename,
                    )
                )
                continue
            if not isinstance(data, dict):
                issues.append(
                    ValidationIssue(
                        type="yaml_root_not_mapping",
                        message="Each template file must contain one YAML mapping.",
                        filename=filename,
                    )
                )
                continue

            try:
                candidate_template = MemoryBackendRouteTemplate.model_validate(data)
            except Exception as exc:
                issues.append(
                    ValidationIssue(
                        type="template_schema_error",
                        message=str(exc),
                        filename=filename,
                    )
                )
                continue

            initial_template = self.initial_templates_by_file[filename]
            issues.extend(self._validate_immutable_fields(filename, data, candidate_template, initial_template))
            issues.extend(self._validate_edit_limits(filename, data, candidate_template, initial_template))
            (materialized / filename).write_text(content.rstrip() + "\n", encoding="utf-8")

        if issues:
            return ValidationResult(False, issues, bundle_hash=bundle_hash)

        try:
            index = BackendRouteTemplateIndex()
            loaded = index.load_from_directory(materialized)
            if loaded != len(self.expected_filenames):
                issues.append(
                    ValidationIssue(
                        type="template_loader_count_mismatch",
                        message=f"Template loader loaded {loaded}, expected {len(self.expected_filenames)}.",
                    )
                )
            matcher = TemplateMatcher(self.template_embedder_factory(bundle_hash), index)
            feature_builder = QueryFeatureBuilder(self.query_embedder)
            for case in self.smoke_cases:
                features = feature_builder.build(case.question)
                matcher.match(features)
        except Exception as exc:
            issues.append(
                ValidationIssue(
                    type="matcher_smoke_error",
                    message=str(exc),
                )
            )

        if issues:
            return ValidationResult(False, issues, bundle_hash=bundle_hash)
        return ValidationResult(True, [], materialized_dir=materialized, bundle_hash=bundle_hash)

    def _validate_immutable_fields(
        self,
        filename: str,
        data: dict[str, Any],
        candidate: MemoryBackendRouteTemplate,
        initial: MemoryBackendRouteTemplate,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        if candidate.template_id != initial.template_id:
            issues.append(
                ValidationIssue(
                    type="immutable_field_changed",
                    message="template_id must match Initial.",
                    template_id=initial.template_id,
                    field="template_id",
                    filename=filename,
                )
            )
        if candidate.target != initial.target:
            issues.append(
                ValidationIssue(
                    type="immutable_field_changed",
                    message="target must match Initial.",
                    template_id=initial.template_id,
                    field="target",
                    filename=filename,
                )
            )
        if candidate.query_spec != initial.query_spec:
            issues.append(
                ValidationIssue(
                    type="immutable_field_changed",
                    message="query_spec must match Initial.",
                    template_id=initial.template_id,
                    field="query_spec",
                    filename=filename,
                )
            )

        initial_without_allowed = _without_allowed_fields(self.initial_yaml[filename])
        candidate_without_allowed = _without_allowed_fields(data)
        if candidate_without_allowed != initial_without_allowed:
            issues.append(
                ValidationIssue(
                    type="non_whitelisted_field_changed",
                    message="Only query_prototypes, hard_negatives, and thresholds may change.",
                    template_id=initial.template_id,
                    filename=filename,
                )
            )
        return issues

    def _validate_edit_limits(
        self,
        filename: str,
        data: dict[str, Any],
        candidate: MemoryBackendRouteTemplate,
        initial: MemoryBackendRouteTemplate,
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        prototype_limit = max(120, len(initial.query_prototypes) + 40)
        hard_negative_limit = max(80, len(initial.hard_negatives) + 40)

        raw_prototypes = data.get("query_prototypes", [])
        if not isinstance(raw_prototypes, list) or not all(isinstance(item, str) for item in raw_prototypes):
            issues.append(
                ValidationIssue(
                    type="invalid_query_prototypes",
                    message="query_prototypes must be a list of strings.",
                    template_id=initial.template_id,
                    field="query_prototypes",
                    filename=filename,
                )
            )
        if len(candidate.query_prototypes) > prototype_limit:
            issues.append(
                ValidationIssue(
                    type="query_prototype_count_limit",
                    message=f"query_prototypes has {len(candidate.query_prototypes)} items; limit is {prototype_limit}.",
                    template_id=initial.template_id,
                    field="query_prototypes",
                    filename=filename,
                )
            )
        for text in candidate.query_prototypes:
            if len(text) > 240:
                issues.append(
                    ValidationIssue(
                        type="query_prototype_text_too_long",
                        message="query_prototypes entries must be at most 240 characters.",
                        template_id=initial.template_id,
                        field="query_prototypes",
                        filename=filename,
                    )
                )
                break

        raw_hard_negatives = data.get("hard_negatives", [])
        if not isinstance(raw_hard_negatives, list) or not all(isinstance(item, dict) for item in raw_hard_negatives):
            issues.append(
                ValidationIssue(
                    type="invalid_hard_negatives",
                    message="hard_negatives must be a list of mappings with query, confusing_with_backend, and reason.",
                    template_id=initial.template_id,
                    field="hard_negatives",
                    filename=filename,
                )
            )
        if len(candidate.hard_negatives) > hard_negative_limit:
            issues.append(
                ValidationIssue(
                    type="hard_negative_count_limit",
                    message=f"hard_negatives has {len(candidate.hard_negatives)} items; limit is {hard_negative_limit}.",
                    template_id=initial.template_id,
                    field="hard_negatives",
                    filename=filename,
                )
            )
        for hard_negative in candidate.hard_negatives:
            if len(hard_negative.query) > 240:
                issues.append(
                    ValidationIssue(
                        type="hard_negative_text_too_long",
                        message="hard_negative.query entries must be at most 240 characters.",
                        template_id=initial.template_id,
                        field="hard_negatives",
                        filename=filename,
                    )
                )
                break

        thresholds = candidate.thresholds.model_dump()
        for field, value in thresholds.items():
            if value < 0 or value > 1.0:
                issues.append(
                    ValidationIssue(
                        type="threshold_out_of_range",
                        message=f"thresholds.{field} must be in [0, 1].",
                        template_id=initial.template_id,
                        field=f"thresholds.{field}",
                        filename=filename,
                    )
                )
        return issues


class MatcherOnlyEvaluator:
    def __init__(
        self,
        validator: TemplateBundleValidator,
        query_embedder: EmbeddingProvider,
        template_embedder_factory: Any,
        num_threads: int,
        parallel_eval: bool,
    ) -> None:
        self.validator = validator
        self.query_embedder = query_embedder
        self.template_embedder_factory = template_embedder_factory
        self.num_threads = max(1, num_threads)
        self.parallel_eval = parallel_eval

    def evaluate(self, cases: list[RouteCase], bundle_text: str) -> tuple[list[dict[str, Any]], list[float], ValidationResult]:
        validation = self.validator.validate(bundle_text)
        if not validation.ok or validation.materialized_dir is None or validation.bundle_hash is None:
            outputs = [self._invalid_output(case, validation) for case in cases]
            return outputs, [0.0 for _ in cases], validation

        if self.parallel_eval and len(cases) > 1 and self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                outputs = list(
                    executor.map(
                        lambda case: self._evaluate_one(case, validation.materialized_dir, validation.bundle_hash),
                        cases,
                    )
                )
        else:
            outputs = [
                self._evaluate_one(case, validation.materialized_dir, validation.bundle_hash)
                for case in cases
            ]
        scores = [float(output["score"]) for output in outputs]
        return outputs, scores, validation

    def _evaluate_one(self, case: RouteCase, template_dir: Path, bundle_hash: str) -> dict[str, Any]:
        index = BackendRouteTemplateIndex()
        index.load_from_directory(template_dir)
        template_by_id = {template.template_id: template for template in index.enabled_templates()}
        matcher = TemplateMatcher(self.template_embedder_factory(bundle_hash), index)
        feature_builder = QueryFeatureBuilder(self.query_embedder)

        try:
            features = feature_builder.build(case.question)
            template_candidates, backend_candidates = matcher.match(features)
            decision = decide_backend(template_candidates, backend_candidates, template_by_id)
            output = build_matcher_output(case, decision, template_candidates, backend_candidates, template_by_id)
            output["score"] = score_case(output)
            return output
        except Exception as exc:
            output = {
                "case_id": case.case_id,
                "question": case.question,
                "expected_backend": case.expected_backend,
                "predicted_backend": "",
                "predicted_backends": [],
                "is_correct": False,
                "is_no_decision": True,
                "route_method": "matcher_error",
                "error": str(exc),
                "score": 0.0,
            }
            return output

    @staticmethod
    def _invalid_output(case: RouteCase, validation: ValidationResult) -> dict[str, Any]:
        return {
            "case_id": case.case_id,
            "question": case.question,
            "expected_backend": case.expected_backend,
            "predicted_backend": "",
            "predicted_backends": [],
            "is_correct": False,
            "is_no_decision": True,
            "route_method": "invalid_proposal",
            "validator_feedback": validation.feedback(),
            "score": 0.0,
        }


class TemplateYamlGepaAdapter:
    def __init__(
        self,
        evaluator: MatcherOnlyEvaluator,
        llm_feedback: LLMFeedbackGenerator,
        cache_dir: Path,
        candidate_eval_cache_enabled: bool,
        initial_summary: dict[str, Any],
        parent_val_summary: dict[str, Any],
        trace_logger: JsonlLogger,
        feedback_logger: JsonlLogger | None = None,
    ) -> None:
        from gepa.core.adapter import EvaluationBatch

        self._evaluation_batch_cls = EvaluationBatch
        self.evaluator = evaluator
        self.llm_feedback = llm_feedback
        self.cache_dir = cache_dir
        self.candidate_eval_cache_enabled = candidate_eval_cache_enabled
        self.initial_summary = initial_summary
        self.parent_val_summary = parent_val_summary
        self.trace_logger = trace_logger
        self.feedback_logger = feedback_logger
        self.last_validator_feedback: dict[str, Any] | None = None
        self.last_feedback_record: dict[str, Any] | None = None
        self.last_parent_bundle_text: str | None = None
        self.debug_context: dict[str, Any] = {}
        self.consecutive_invalid_proposals = 0
        self.validator_failure_count = 0
        self.cache_hits = 0
        self.cache_misses = 0

    propose_new_texts = None

    def evaluate(
        self,
        batch: list[RouteCase],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> Any:
        bundle_text = candidate[COMPONENT_NAME]
        cache_path = self._cache_path(batch, bundle_text)
        if self.candidate_eval_cache_enabled and not capture_traces and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            outputs = cached["outputs"]
            self._update_validation_state_from_cached_outputs(outputs, bundle_text)
            self.cache_hits += 1
            return self._evaluation_batch_cls(
                outputs=outputs,
                scores=[float(score) for score in cached["scores"]],
                trajectories=None,
                objective_scores=cached.get("objective_scores"),
            )

        self.cache_misses += 1
        outputs, scores, validation = self.evaluator.evaluate(batch, bundle_text)
        if validation.ok:
            batch_score = float(summarize_outputs(outputs)["gepa_score"])
            scores = [batch_score for _ in outputs]
        objective_scores = [objective_scores_for_output(output) for output in outputs]

        if not validation.ok:
            self.validator_failure_count += 1
            self.consecutive_invalid_proposals += 1
            self.last_validator_feedback = validation.feedback()
            self.trace_logger.write(
                {
                    "event": "validator_failed",
                    "bundle_hash": validation.bundle_hash,
                    "feedback": self.last_validator_feedback,
                }
            )
        elif not capture_traces:
            self.consecutive_invalid_proposals = 0

        trajectories = None
        if capture_traces:
            trajectories = [
                {
                    "case": asdict(case),
                    "output": output,
                    "deterministic_feedback": deterministic_feedback_for_case(output),
                }
                for case, output in zip(batch, outputs, strict=False)
            ]

        if self.candidate_eval_cache_enabled and not capture_traces:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    {
                        "outputs": outputs,
                        "scores": scores,
                        "objective_scores": objective_scores,
                        "validation_feedback": validation.feedback() if not validation.ok else None,
                    },
                    ensure_ascii=False,
                    indent=2,
                    default=_json_default,
                ),
                encoding="utf-8",
            )

        return self._evaluation_batch_cls(
            outputs=outputs,
            scores=scores,
            trajectories=trajectories,
            objective_scores=objective_scores,
        )

    def _update_validation_state_from_cached_outputs(
        self,
        outputs: list[dict[str, Any]],
        bundle_text: str,
    ) -> None:
        validator_feedback = None
        for output in outputs:
            if output.get("route_method") != "invalid_proposal":
                continue
            feedback = output.get("validator_feedback")
            if isinstance(feedback, dict):
                validator_feedback = feedback
                break

        if validator_feedback is None:
            self.consecutive_invalid_proposals = 0
            return

        self.validator_failure_count += 1
        self.consecutive_invalid_proposals += 1
        self.last_validator_feedback = validator_feedback
        self.trace_logger.write(
            {
                "event": "validator_failed_cache_hit",
                "bundle_hash": _sha256_text(bundle_text),
                "feedback": validator_feedback,
            }
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: Any,
        components_to_update: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        trajectories = eval_batch.trajectories or []
        outputs = [trajectory["output"] for trajectory in trajectories]
        case_feedback = [trajectory["deterministic_feedback"] for trajectory in trajectories]
        deterministic_feedback = {
            "feedback_type": "deterministic_feedback",
            "action_type_definitions": DETERMINISTIC_ACTION_TYPE_DEFINITIONS,
            "batch_summary": summarize_outputs(outputs),
            "case_feedback": case_feedback,
            "batch_action_recommendations": aggregate_action_recommendations(case_feedback),
            "suspected_overbroad_templates": suspected_overbroad_templates(outputs),
            "top_confusion_pairs": top_confusion_pairs(outputs),
        }
        validator_feedback = self.last_validator_feedback
        llm_payload = {
            "deterministic_feedback": deterministic_feedback,
            "validator_feedback": validator_feedback,
            "parent_val_summary": self.parent_val_summary,
            "initial_summary": self.initial_summary,
        }
        llm_feedback = self.llm_feedback.generate(llm_payload)
        llm_feedback_debug = dict(self.llm_feedback.last_debug)

        feedback_text = json.dumps(
            {
                "validator_feedback": validator_feedback,
                "deterministic_feedback": deterministic_feedback,
                "llm_feedback": llm_feedback,
                "parent_val_summary": self.parent_val_summary,
                "initial_summary": self.initial_summary,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        )
        record = {
            "Inputs": {
                "task": "Improve matcher-only routing by editing the YAML Template Bundle.",
                "output_contract": OUTPUT_CONTRACT,
            },
            "Generated Outputs": {
                "current_bundle_hash": _sha256_text(candidate[COMPONENT_NAME]),
                "component": COMPONENT_NAME,
            },
            "Feedback": feedback_text,
        }
        self._write_feedback_record(
            candidate=candidate,
            components_to_update=components_to_update,
            outputs=outputs,
            deterministic_feedback=deterministic_feedback,
            validator_feedback=validator_feedback,
            llm_payload=llm_payload,
            llm_feedback=llm_feedback,
            llm_feedback_debug=llm_feedback_debug,
            reflective_dataset_record=record,
            feedback_text=feedback_text,
        )
        self.last_validator_feedback = None

        return {name: [record] for name in components_to_update if name == COMPONENT_NAME}

    def update_debug_context(self, **values: Any) -> None:
        self.debug_context.update(values)

    def _write_feedback_record(
        self,
        candidate: dict[str, str],
        components_to_update: list[str],
        outputs: list[dict[str, Any]],
        deterministic_feedback: dict[str, Any],
        validator_feedback: dict[str, Any] | None,
        llm_payload: dict[str, Any],
        llm_feedback: dict[str, Any],
        llm_feedback_debug: dict[str, Any],
        reflective_dataset_record: dict[str, Any],
        feedback_text: str,
    ) -> None:
        row = {
            "event": "feedback_record",
            **self.debug_context,
            "components_to_update": components_to_update,
            "parent_bundle_hash": _sha256_text(candidate[COMPONENT_NAME]),
            "parent_bundle_chars": len(candidate[COMPONENT_NAME]),
            "feedback_text_chars": len(feedback_text),
            "batch_case_ids": [output.get("case_id") for output in outputs],
            "incorrect_case_ids": [
                output.get("case_id")
                for output in outputs
                if not output.get("is_correct")
            ],
            "no_decision_case_ids": [
                output.get("case_id")
                for output in outputs
                if output.get("is_no_decision")
            ],
            "matcher_outputs": outputs,
            "deterministic_feedback": deterministic_feedback,
            "validator_feedback": validator_feedback,
            "llm_payload": llm_payload,
            "llm_feedback": llm_feedback,
            "llm_feedback_debug": llm_feedback_debug,
            "reflective_dataset_record": reflective_dataset_record,
        }
        self.last_feedback_record = row
        self.last_parent_bundle_text = candidate[COMPONENT_NAME]
        if self.feedback_logger is not None:
            self.feedback_logger.write(row)

    def _cache_path(self, batch: list[RouteCase], bundle_text: str) -> Path:
        case_ids = [case.case_id for case in batch]
        key = _sha256_text(
            json.dumps(
                {
                    "bundle_hash": _sha256_text(bundle_text),
                    "case_ids": case_ids,
                    "evaluator": "matcher_only_v2",
                },
                sort_keys=True,
            )
        )
        return self.cache_dir / "candidate_eval" / f"{key}.json"


class InvalidProposalStopper:
    def __init__(self, adapter: TemplateYamlGepaAdapter, max_invalid_proposals: int) -> None:
        self.adapter = adapter
        self.max_invalid_proposals = max_invalid_proposals

    def __call__(self, _state: Any) -> bool:
        return self.adapter.consecutive_invalid_proposals >= self.max_invalid_proposals


class StratifiedBatchSampler:
    def __init__(
        self,
        buckets: dict[str, list[int]],
        minibatch_size: int,
        seed: int,
    ) -> None:
        self.buckets = {name: list(ids) for name, ids in buckets.items()}
        self.minibatch_size = minibatch_size
        self.rng = random.Random(seed)
        self.freq: Counter[int] = Counter()
        self.quotas = [
            ("graph_wrong", 2),
            ("temporal_wrong", 2),
            ("openviking_wrong", 2),
            ("no_decision_cases", 2),
            ("graph_correct", 2),
            ("temporal_correct", 2),
            ("openviking_correct", 2),
            ("random_pool", 2),
        ]

    def next_minibatch_ids(self, loader: Any, _state: Any) -> list[int]:
        all_ids = list(loader.all_ids())
        if not all_ids:
            raise ValueError("Cannot sample minibatch from empty trainset.")
        selected: list[int] = []
        selected_set: set[int] = set()

        for bucket_name, quota in self.quotas:
            self._append_from_bucket(bucket_name, quota, selected, selected_set)

        while len(selected) < self.minibatch_size:
            pool = self.buckets.get("random_pool") or all_ids
            self._append_from_pool(pool, 1, selected, selected_set, allow_repeat=len(selected_set) >= len(all_ids))

        selected = selected[: self.minibatch_size]
        for data_id in selected:
            self.freq[data_id] += 1
        return selected

    def _append_from_bucket(
        self,
        bucket_name: str,
        quota: int,
        selected: list[int],
        selected_set: set[int],
    ) -> None:
        pool = self.buckets.get(bucket_name) or self.buckets.get("random_pool") or []
        self._append_from_pool(pool, quota, selected, selected_set)

    def _append_from_pool(
        self,
        pool: list[int],
        quota: int,
        selected: list[int],
        selected_set: set[int],
        allow_repeat: bool = False,
    ) -> None:
        if not pool:
            return
        candidates = list(pool)
        self.rng.shuffle(candidates)
        candidates.sort(key=lambda data_id: (self.freq[data_id], data_id))
        for data_id in candidates:
            if len(selected) >= self.minibatch_size or quota <= 0:
                return
            if data_id in selected_set and not allow_repeat:
                continue
            selected.append(data_id)
            selected_set.add(data_id)
            quota -= 1
        while quota > 0 and allow_repeat:
            data_id = min(pool, key=lambda item: (self.freq[item], item))
            selected.append(data_id)
            quota -= 1


class GepaTraceCallback:
    def __init__(
        self,
        trace_logger: JsonlLogger,
        candidate_csv: Path,
        proposal_dir: Path,
        adapter: TemplateYamlGepaAdapter,
        initial_val_summary: dict[str, Any],
    ) -> None:
        self.trace_logger = trace_logger
        self.candidate_csv = candidate_csv
        self.proposal_dir = proposal_dir
        self.proposal_diff_dir = proposal_dir.parent / "proposal_diffs"
        self.adapter = adapter
        self.val_summaries: dict[int, dict[str, Any]] = {0: initial_val_summary}
        self.candidate_csv.parent.mkdir(parents=True, exist_ok=True)
        self.proposal_dir.mkdir(parents=True, exist_ok=True)
        self.proposal_diff_dir.mkdir(parents=True, exist_ok=True)
        if not self.candidate_csv.exists():
            with self.candidate_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "iteration",
                        "candidate_idx",
                        "average_score",
                        "num_examples_evaluated",
                        "is_best_program",
                    ],
                )
                writer.writeheader()

    def on_minibatch_sampled(self, event: dict[str, Any]) -> None:
        self.adapter.update_debug_context(
            iteration=event.get("iteration"),
            minibatch_ids=event.get("minibatch_ids"),
            trainset_size=event.get("trainset_size"),
        )
        self.trace_logger.write({"event": "minibatch_sampled", **event})

    def on_candidate_selected(self, event: dict[str, Any]) -> None:
        candidate_idx = int(event["candidate_idx"])
        self.adapter.parent_val_summary = self.val_summaries.get(
            candidate_idx,
            {
                "candidate_idx": candidate_idx,
                "mean_case_score": float(event.get("score", 0.0)),
                "note": "Only aggregate score was available for this Parent.",
            },
        )
        self.adapter.update_debug_context(
            iteration=event.get("iteration"),
            parent_candidate_idx=candidate_idx,
            parent_val_score=event.get("score"),
        )
        self.trace_logger.write(
            {
                "event": "candidate_selected",
                "iteration": event["iteration"],
                "candidate_idx": candidate_idx,
                "score": event.get("score"),
            }
        )

    def on_candidate_rejected(self, event: dict[str, Any]) -> None:
        self.trace_logger.write({"event": "candidate_rejected", **event})

    def on_candidate_accepted(self, event: dict[str, Any]) -> None:
        self.trace_logger.write({"event": "candidate_accepted", **event})

    def on_reflective_dataset_built(self, event: dict[str, Any]) -> None:
        dataset = event.get("dataset") or {}
        self.trace_logger.write(
            {
                "event": "reflective_dataset_built",
                "iteration": event.get("iteration"),
                "candidate_idx": event.get("candidate_idx"),
                "components": event.get("components"),
                "component_record_counts": {
                    name: len(records)
                    for name, records in dataset.items()
                    if isinstance(records, list)
                },
            }
        )

    def on_proposal_end(self, event: dict[str, Any]) -> None:
        proposal_records = []
        iteration = event.get("iteration")
        iteration_num = int(iteration) if iteration is not None else 0
        for component, text in (event.get("new_instructions") or {}).items():
            safe_component = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(component)).strip("_") or "component"
            relative_path = Path("reports") / "proposals" / f"iter_{iteration_num:04d}_{safe_component}.txt"
            path = self.proposal_dir / relative_path.name
            path.write_text(str(text), encoding="utf-8")
            diff_summary = None
            diff_relative_path = None
            if component == COMPONENT_NAME:
                diff_summary = build_proposal_diff_summary(
                    self.adapter.last_parent_bundle_text,
                    str(text),
                    self.adapter.last_feedback_record,
                )
                diff_relative_path = (
                    Path("reports")
                    / "proposal_diffs"
                    / f"iter_{iteration_num:04d}_{safe_component}.diff_summary.json"
                )
                diff_path = self.proposal_diff_dir / diff_relative_path.name
                diff_path.write_text(
                    json.dumps(diff_summary, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8",
                )
            proposal_records.append(
                {
                    "component": component,
                    "path": str(relative_path),
                    "text_hash": _sha256_text(str(text)),
                    "text_chars": len(str(text)),
                    "diff_summary_path": str(diff_relative_path) if diff_relative_path else None,
                    "diff_summary": _trace_proposal_diff_summary(diff_summary),
                }
            )
        self.trace_logger.write(
            {
                "event": "proposal_generated",
                "iteration": event.get("iteration"),
                "proposals": proposal_records,
            }
        )

    def on_valset_evaluated(self, event: dict[str, Any]) -> None:
        outputs_by_val_id = event.get("outputs_by_val_id")
        if outputs_by_val_id:
            self.val_summaries[int(event["candidate_idx"])] = summarize_outputs(list(outputs_by_val_id.values()))
        else:
            self.val_summaries[int(event["candidate_idx"])] = {
                "candidate_idx": int(event["candidate_idx"]),
                "mean_case_score": float(event["average_score"]),
                "note": "GEPA did not expose full outputs for this val evaluation.",
            }
        row = {
            "iteration": event["iteration"],
            "candidate_idx": event["candidate_idx"],
            "average_score": event["average_score"],
            "num_examples_evaluated": event["num_examples_evaluated"],
            "is_best_program": event["is_best_program"],
        }
        with self.candidate_csv.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            writer.writerow(row)
        self.trace_logger.write({"event": "valset_evaluated", **row})


def decide_backend(
    template_candidates: list[TemplateCandidate],
    backend_candidates: list[BackendCandidate],
    template_by_id: dict[str, MemoryBackendRouteTemplate],
) -> dict[str, Any]:
    if not backend_candidates:
        return {
            "predicted_backend": "",
            "predicted_backends": [],
            "is_no_decision": True,
            "route_method": "no_template_match",
        }

    best = backend_candidates[0]
    best_template = template_by_id.get(best.best_template_id)
    second = backend_candidates[1] if len(backend_candidates) > 1 else None
    second_template = template_by_id.get(second.best_template_id) if second else None
    if best_template is None:
        return {
            "predicted_backend": "",
            "predicted_backends": [],
            "is_no_decision": True,
            "route_method": "missing_best_template",
        }

    margin = best.score - (second.score if second else 0.0)
    accept_threshold = best_template.thresholds.accept
    margin_threshold = best_template.thresholds.margin

    if best.score >= accept_threshold and margin >= margin_threshold:
        return {
            "predicted_backend": best.backend_id,
            "predicted_backends": [best.backend_id],
            "is_no_decision": False,
            "route_method": "template_embedding",
            "matched_template_id": best.best_template_id,
        }

    if (
        second is not None
        and second_template is not None
        and best.backend_id != second.backend_id
        and best.score >= accept_threshold
        and second.score >= second_template.thresholds.accept
        and margin < margin_threshold
        and second.score / max(abs(best.score), 1e-6) >= 0.85
    ):
        return {
            "predicted_backend": best.backend_id,
            "predicted_backends": [best.backend_id, second.backend_id],
            "is_no_decision": False,
            "route_method": "template_embedding_multi_backend",
            "matched_template_id": best.best_template_id,
        }

    return {
        "predicted_backend": "",
        "predicted_backends": [],
        "is_no_decision": True,
        "route_method": "no_decision",
        "matched_template_id": best.best_template_id,
    }


def build_matcher_output(
    case: RouteCase,
    decision: dict[str, Any],
    template_candidates: list[TemplateCandidate],
    backend_candidates: list[BackendCandidate],
    template_by_id: dict[str, MemoryBackendRouteTemplate],
) -> dict[str, Any]:
    expected_rank = None
    expected_score = 0.0
    for idx, candidate in enumerate(backend_candidates, start=1):
        if candidate.backend_id == case.expected_backend:
            expected_rank = idx
            expected_score = candidate.score
            break

    winning = backend_candidates[0] if backend_candidates else None
    winning_score = winning.score if winning else 0.0
    expected_backend_best_template_id = ""
    top_wrong_score = max(
        [candidate.score for candidate in backend_candidates if candidate.backend_id != case.expected_backend],
        default=0.0,
    )
    for candidate in backend_candidates:
        if candidate.backend_id == case.expected_backend:
            expected_backend_best_template_id = candidate.best_template_id
            break
    margin_to_win = expected_score - top_wrong_score

    predicted_backend = str(decision.get("predicted_backend") or "")
    is_no_decision = bool(decision.get("is_no_decision"))
    is_correct = (not is_no_decision) and predicted_backend == case.expected_backend
    matched_template_id = decision.get("matched_template_id") or (winning.best_template_id if winning else "")
    matched_template = template_by_id.get(str(matched_template_id))
    template_candidate_by_id = {candidate.template_id: candidate for candidate in template_candidates}
    expected_template_candidate = template_candidate_by_id.get(expected_backend_best_template_id)
    winning_template_candidate = template_candidate_by_id.get(winning.best_template_id) if winning else None

    return {
        "case_id": case.case_id,
        "question": case.question,
        "expected_backend": case.expected_backend,
        "predicted_backend": predicted_backend,
        "predicted_backends": decision.get("predicted_backends", []),
        "is_correct": is_correct,
        "is_no_decision": is_no_decision,
        "route_method": decision.get("route_method", ""),
        "expected_backend_rank": expected_rank,
        "expected_backend_score": round(float(expected_score), 6),
        "expected_backend_best_template_id": expected_backend_best_template_id,
        "expected_template_score_components": (
            expected_template_candidate.score_components if expected_template_candidate else {}
        ),
        "winning_backend": winning.backend_id if winning else "",
        "winning_backend_score": round(float(winning_score), 6),
        "winning_template_score_components": (
            winning_template_candidate.score_components if winning_template_candidate else {}
        ),
        "margin_to_win": round(float(margin_to_win), 6),
        "matched_template_id": matched_template_id,
        "matched_template_accept": matched_template.thresholds.accept if matched_template else None,
        "matched_template_margin": matched_template.thresholds.margin if matched_template else None,
        "top_backend_ranking": [
            {
                "backend_id": candidate.backend_id,
                "best_template_id": candidate.best_template_id,
                "score": round(float(candidate.score), 6),
            }
            for candidate in backend_candidates[:5]
        ],
        "top_template_ranking": [
            {
                "template_id": candidate.template_id,
                "backend_id": candidate.primary_backend_id,
                "score": round(float(candidate.score), 6),
                "score_components": candidate.score_components,
            }
            for candidate in template_candidates[:5]
        ],
    }


def score_case(output: dict[str, Any]) -> float:
    correct = 1.0 if output.get("is_correct") else 0.0
    top2 = 1.0 if output.get("expected_backend_rank") in (1, 2) else 0.0
    accepted = 0.0 if output.get("is_no_decision") else 1.0
    no_decision_control = accepted
    margin = float(output.get("margin_to_win") or 0.0)
    margin_score = max(0.0, min(1.0, (margin + 0.15) / 0.30))
    score = (
        0.45 * correct
        + 0.20 * top2
        + 0.15 * margin_score
        + 0.10 * accepted
        + 0.10 * no_decision_control
    )
    return round(float(score), 6)


def objective_scores_for_output(output: dict[str, Any]) -> dict[str, float]:
    return {
        "backend_correct": 1.0 if output.get("is_correct") else 0.0,
        "expected_top2": 1.0 if output.get("expected_backend_rank") in (1, 2) else 0.0,
        "template_accept": 0.0 if output.get("is_no_decision") else 1.0,
    }


def deterministic_feedback_for_case(output: dict[str, Any]) -> dict[str, Any]:
    action_recommendation = recommend_deterministic_action(output)
    diagnosis = str(action_recommendation["diagnosis"])
    action_type = str(action_recommendation["action_type"])
    return {
        "case_id": output.get("case_id"),
        "question": output.get("question"),
        "expected_backend": output.get("expected_backend"),
        "predicted_backend": output.get("predicted_backend") or NO_DECISION,
        "is_correct": output.get("is_correct"),
        "is_no_decision": output.get("is_no_decision"),
        "expected_backend_rank": output.get("expected_backend_rank"),
        "expected_backend_score": output.get("expected_backend_score"),
        "winning_backend": output.get("winning_backend"),
        "winning_backend_score": output.get("winning_backend_score"),
        "margin_to_win": output.get("margin_to_win"),
        "matched_template_id": output.get("matched_template_id"),
        "top_backend_ranking": output.get("top_backend_ranking", [])[:3],
        "top_template_ranking": output.get("top_template_ranking", [])[:3],
        "diagnosis": diagnosis,
        "suggested_action_type": action_type,
        "action_recommendation": action_recommendation,
    }


def diagnose_case(output: dict[str, Any]) -> tuple[str, str]:
    action = recommend_deterministic_action(output)
    return str(action["diagnosis"]), str(action["action_type"])


def recommend_deterministic_action(output: dict[str, Any]) -> dict[str, Any]:
    expected_backend = str(output.get("expected_backend") or "")
    predicted_backend = str(output.get("predicted_backend") or "")
    winning_backend = str(output.get("winning_backend") or predicted_backend or "")
    matched_template_id = str(output.get("matched_template_id") or "")
    expected_template_id = str(output.get("expected_backend_best_template_id") or "")
    expected_rank = output.get("expected_backend_rank")
    expected_score = float(output.get("expected_backend_score") or 0.0)
    margin_to_win = float(output.get("margin_to_win") or 0.0)
    is_no_decision = bool(output.get("is_no_decision"))

    if output.get("is_correct"):
        return _action_recommendation(
            action_type="preserve_behavior",
            diagnosis="route should be preserved",
            target_backend=expected_backend,
            target_template_id=matched_template_id or expected_template_id,
            field="none",
            direction="preserve",
            priority="high",
            rationale="This case is already correct and should anchor future Proposal changes.",
        )

    matched_accept = output.get("matched_template_accept")
    matched_margin = output.get("matched_template_margin")
    accept_gap = None
    if matched_accept is not None:
        accept_gap = float(matched_accept) - expected_score

    if is_no_decision and expected_rank == 1:
        return _action_recommendation(
            action_type="lower_expected_accept_or_margin",
            diagnosis="expected backend ranked first but was not accepted",
            target_backend=expected_backend,
            target_template_id=matched_template_id or expected_template_id,
            field="thresholds",
            direction="slightly_decrease_accept_or_margin",
            priority="medium" if accept_gap is None or accept_gap <= 0.05 else "low",
            rationale=(
                "The expected backend is already the top-ranked backend but fails acceptance; "
                f"accept_gap={round(accept_gap, 6) if accept_gap is not None else 'unknown'}, "
                f"margin_threshold={matched_margin if matched_margin is not None else 'unknown'}."
            ),
            guardrail="Use small threshold changes and preserve correct high-margin cases.",
        )

    if is_no_decision:
        return _action_recommendation(
            action_type="add_expected_backend_prototypes",
            diagnosis="templates did not confidently accept the query",
            target_backend=expected_backend,
            target_template_id=expected_template_id,
            field="query_prototypes",
            direction="add_generalized_patterns",
            priority="medium",
            rationale="No backend was accepted; improve expected backend coverage before lowering thresholds broadly.",
            guardrail="Add generalized patterns, not exact minibatch questions.",
        )

    if expected_rank is None or expected_rank > 2:
        return _action_recommendation(
            action_type="add_expected_backend_prototypes",
            diagnosis="expected backend coverage is weak",
            target_backend=expected_backend,
            target_template_id=expected_template_id,
            field="query_prototypes",
            direction="add_generalized_patterns",
            priority="high",
            rationale="The expected backend is outside the top two backends, indicating weak positive coverage.",
            guardrail="Use generalized patterns, not exact minibatch questions.",
        )

    if expected_rank <= 2 and margin_to_win > -0.08:
        return _action_recommendation(
            action_type="add_discriminative_prototypes_or_hard_negatives",
            diagnosis="expected backend is close but lacks margin",
            target_backend=expected_backend,
            target_template_id=expected_template_id,
            field="query_prototypes/hard_negatives",
            direction="add_discriminative_positive_or_negative_patterns",
            priority="medium",
            rationale="The expected backend is close to the winner; small discriminative edits may flip the route.",
            guardrail="Prefer edits that separate backend intent rather than broad threshold changes.",
        )

    return _action_recommendation(
        action_type="add_wrong_template_hard_negatives",
        diagnosis="wrong template may be overbroad",
        target_backend=winning_backend,
        target_template_id=matched_template_id,
        field="hard_negatives",
        direction="add_generalized_negative_patterns",
        priority="medium",
        rationale="The wrong template accepted the query; add hard negatives for this cross-backend pattern.",
        guardrail="Use generalized patterns, not exact minibatch questions.",
    )


def _action_recommendation(
    action_type: str,
    diagnosis: str,
    target_backend: str,
    target_template_id: str,
    field: str,
    direction: str,
    priority: str,
    rationale: str,
    guardrail: str = "",
    secondary_action_type: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "action_type": action_type,
        "definition": DETERMINISTIC_ACTION_TYPE_DEFINITIONS[action_type],
        "diagnosis": diagnosis,
        "target_backend": target_backend,
        "target_template_id": target_template_id,
        "field": field,
        "direction": direction,
        "priority": priority,
        "rationale": rationale,
    }
    if guardrail:
        row["guardrail"] = guardrail
    if secondary_action_type:
        row["secondary_action_type"] = secondary_action_type
        row["secondary_definition"] = DETERMINISTIC_ACTION_TYPE_DEFINITIONS[secondary_action_type]
    return row


def aggregate_action_recommendations(case_feedback: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], dict[str, Any]] = {}
    priority_rank = {"high": 3, "medium": 2, "low": 1}
    for row in case_feedback:
        action = row.get("action_recommendation") or {}
        action_type = str(action.get("action_type") or "")
        target_template_id = str(action.get("target_template_id") or "")
        field = str(action.get("field") or "")
        if not action_type or action_type == "preserve_behavior":
            continue
        key = (action_type, target_template_id, field)
        bucket = grouped.setdefault(
            key,
            {
                "action_type": action_type,
                "definition": DETERMINISTIC_ACTION_TYPE_DEFINITIONS.get(action_type, ""),
                "target_template_id": target_template_id,
                "target_backend": action.get("target_backend"),
                "field": field,
                "direction": action.get("direction"),
                "case_ids": [],
                "priority": action.get("priority", "low"),
                "rationale": action.get("rationale", ""),
                "guardrail": action.get("guardrail", ""),
            },
        )
        bucket["case_ids"].append(row.get("case_id"))
        if priority_rank.get(str(action.get("priority")), 0) > priority_rank.get(str(bucket.get("priority")), 0):
            bucket["priority"] = action.get("priority")
            bucket["rationale"] = action.get("rationale", "")
            bucket["guardrail"] = action.get("guardrail", "")

    recommendations = []
    for bucket in grouped.values():
        bucket["evidence_count"] = len(bucket["case_ids"])
        recommendations.append(bucket)
    recommendations.sort(
        key=lambda item: (
            priority_rank.get(str(item.get("priority")), 0),
            int(item.get("evidence_count", 0)),
            str(item.get("action_type")),
        ),
        reverse=True,
    )
    return recommendations[:limit]


def summarize_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    if not outputs:
        return {
            "count": 0,
            "gepa_score": 0.0,
            "backend_accuracy": 0.0,
            "macro_backend_recall": 0.0,
            "no_decision_rate": 0.0,
            "template_accept_rate": 0.0,
            "per_backend_recall": {},
        }
    total = len(outputs)
    correct = sum(1 for output in outputs if output.get("is_correct"))
    no_decision = sum(1 for output in outputs if output.get("is_no_decision"))
    top2 = sum(1 for output in outputs if output.get("expected_backend_rank") in (1, 2))
    margin_scores = [
        max(0.0, min(1.0, (float(output.get("margin_to_win") or 0.0) + 0.15) / 0.30))
        for output in outputs
    ]

    per_backend_recall: dict[str, float] = {}
    for backend in sorted({str(output.get("expected_backend")) for output in outputs}):
        backend_outputs = [output for output in outputs if output.get("expected_backend") == backend]
        if backend_outputs:
            per_backend_recall[backend] = round(
                sum(1 for output in backend_outputs if output.get("is_correct")) / len(backend_outputs),
                6,
            )
    macro_backend_recall = (
        sum(per_backend_recall.values()) / len(per_backend_recall) if per_backend_recall else 0.0
    )
    backend_accuracy = correct / total
    no_decision_rate = no_decision / total
    template_accept_rate = 1.0 - no_decision_rate
    expected_backend_top2_rate = top2 / total
    expected_backend_margin_score = sum(margin_scores) / len(margin_scores)
    no_decision_control_score = 1.0 - no_decision_rate
    overbroad_penalty = estimate_overbroad_penalty(outputs)
    gepa_score = (
        0.40 * backend_accuracy
        + 0.25 * macro_backend_recall
        + 0.15 * expected_backend_top2_rate
        + 0.10 * expected_backend_margin_score
        + 0.05 * template_accept_rate
        + 0.05 * no_decision_control_score
        - overbroad_penalty
    )
    return {
        "count": total,
        "gepa_score": round(float(gepa_score), 6),
        "mean_case_score": round(float(sum(float(output.get("score", 0.0)) for output in outputs) / total), 6),
        "backend_accuracy": round(float(backend_accuracy), 6),
        "macro_backend_recall": round(float(macro_backend_recall), 6),
        "expected_backend_top2_rate": round(float(expected_backend_top2_rate), 6),
        "expected_backend_margin_score": round(float(expected_backend_margin_score), 6),
        "no_decision_rate": round(float(no_decision_rate), 6),
        "template_accept_rate": round(float(template_accept_rate), 6),
        "overbroad_penalty": round(float(overbroad_penalty), 6),
        "per_backend_recall": per_backend_recall,
        "top_confusion_pairs": top_confusion_pairs(outputs),
        "suspected_overbroad_templates": suspected_overbroad_templates(outputs),
    }


def estimate_overbroad_penalty(outputs: list[dict[str, Any]]) -> float:
    wrong_template_counts = Counter(
        str(output.get("matched_template_id"))
        for output in outputs
        if not output.get("is_correct") and not output.get("is_no_decision") and output.get("matched_template_id")
    )
    if not wrong_template_counts:
        return 0.0
    max_fraction = wrong_template_counts.most_common(1)[0][1] / max(1, len(outputs))
    return max(0.0, max_fraction - 0.35) * 0.20


def top_confusion_pairs(outputs: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    counts = Counter()
    for output in outputs:
        if output.get("is_correct") or output.get("is_no_decision"):
            continue
        counts[(output.get("expected_backend"), output.get("predicted_backend"))] += 1
    return [
        {"expected_backend": expected, "predicted_backend": predicted, "count": count}
        for (expected, predicted), count in counts.most_common(limit)
    ]


def suspected_overbroad_templates(outputs: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    counts = Counter()
    for output in outputs:
        if output.get("is_correct") or output.get("is_no_decision"):
            continue
        template_id = output.get("matched_template_id")
        if template_id:
            counts[str(template_id)] += 1
    return [
        {"template_id": template_id, "wrong_absorptions": count}
        for template_id, count in counts.most_common(limit)
    ]


def build_sampling_buckets(outputs: list[dict[str, Any]]) -> dict[str, list[int]]:
    buckets: dict[str, list[int]] = defaultdict(list)
    for idx, output in enumerate(outputs):
        expected = str(output.get("expected_backend") or "")
        prefix = backend_prefix(expected)
        buckets["random_pool"].append(idx)
        if output.get("is_no_decision"):
            buckets["no_decision_cases"].append(idx)
        elif output.get("is_correct"):
            buckets[f"{prefix}_correct"].append(idx)
        else:
            buckets[f"{prefix}_wrong"].append(idx)
    for key in (
        "graph_correct",
        "graph_wrong",
        "temporal_correct",
        "temporal_wrong",
        "openviking_correct",
        "openviking_wrong",
        "no_decision_cases",
        "random_pool",
    ):
        buckets.setdefault(key, [])
    return {key: list(value) for key, value in sorted(buckets.items())}


def backend_prefix(backend: str) -> str:
    if backend.startswith("graph"):
        return "graph"
    if backend.startswith("temporal"):
        return "temporal"
    return "openviking"


def parse_template_bundle(bundle_text: str) -> dict[str, str] | list[ValidationIssue]:
    marker_re = re.compile(r"(?m)^# FILE: (?P<filename>[A-Za-z0-9_.-]+\.ya?ml)\s*$")
    matches = list(marker_re.finditer(bundle_text))
    if not matches:
        return [
            ValidationIssue(
                type="bundle_parse_error",
                message="Template Bundle must include '# FILE: <name>.yaml' markers.",
            )
        ]

    prefix = bundle_text[: matches[0].start()].strip()
    if prefix:
        return [
            ValidationIssue(
                type="bundle_parse_error",
                message="No text is allowed before the first '# FILE:' marker.",
            )
        ]

    files: dict[str, str] = {}
    issues: list[ValidationIssue] = []
    for idx, match in enumerate(matches):
        filename = match.group("filename")
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(bundle_text)
        content = bundle_text[start:end].strip()
        if not content:
            issues.append(
                ValidationIssue(
                    type="empty_template_file",
                    message=f"{filename} has no YAML content.",
                    filename=filename,
                )
            )
        if filename in files:
            issues.append(
                ValidationIssue(
                    type="duplicate_template_file",
                    message=f"{filename} appears more than once.",
                    filename=filename,
                )
            )
        files[filename] = content + "\n"
    return issues if issues else files


def read_template_bundle(template_dir: Path) -> tuple[str, dict[str, str]]:
    paths = sorted(set(template_dir.glob("*.yaml")) | set(template_dir.glob("*.yml")))
    if not paths:
        raise ValueError(f"No YAML templates found under {template_dir}")
    files = {path.name: path.read_text(encoding="utf-8") for path in paths}
    text = serialize_template_bundle(files)
    return text, files


def serialize_template_bundle(files: dict[str, str]) -> str:
    chunks = []
    for filename in sorted(files):
        chunks.append(f"# FILE: {filename}\n{files[filename].rstrip()}\n")
    return "\n".join(chunks).rstrip() + "\n"


def build_proposal_diff_summary(
    parent_bundle_text: str | None,
    proposal_bundle_text: str,
    feedback_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not parent_bundle_text:
        return {
            "status": "unavailable",
            "reason": "missing_parent_bundle",
            "proposal_bundle_hash": _sha256_text(proposal_bundle_text),
        }

    parent_files = parse_template_bundle(parent_bundle_text)
    proposal_files = parse_template_bundle(proposal_bundle_text)
    if isinstance(parent_files, list) or isinstance(proposal_files, list):
        return {
            "status": "unavailable",
            "reason": "bundle_parse_failed",
            "parent_parse_issues": [asdict(issue) for issue in parent_files] if isinstance(parent_files, list) else [],
            "proposal_parse_issues": [asdict(issue) for issue in proposal_files] if isinstance(proposal_files, list) else [],
            "parent_bundle_hash": _sha256_text(parent_bundle_text),
            "proposal_bundle_hash": _sha256_text(proposal_bundle_text),
        }

    parent_yaml, parent_issues = _load_bundle_yaml(parent_files)
    proposal_yaml, proposal_issues = _load_bundle_yaml(proposal_files)
    if parent_issues or proposal_issues:
        return {
            "status": "unavailable",
            "reason": "yaml_parse_failed",
            "parent_yaml_issues": parent_issues,
            "proposal_yaml_issues": proposal_issues,
            "parent_bundle_hash": _sha256_text(parent_bundle_text),
            "proposal_bundle_hash": _sha256_text(proposal_bundle_text),
        }

    changed_templates: list[dict[str, Any]] = []
    for filename in sorted(set(parent_yaml) | set(proposal_yaml)):
        parent_data = parent_yaml.get(filename, {})
        proposal_data = proposal_yaml.get(filename, {})
        changes = _template_allowed_field_changes(parent_data, proposal_data)
        if not changes:
            continue
        changed_templates.append(
            {
                "filename": filename,
                "template_id": proposal_data.get("template_id") or parent_data.get("template_id") or "",
                "changes": changes,
            }
        )

    executed_action_types = infer_executed_action_types(changed_templates)
    recommendation = _proposal_action_recommendation_summary(feedback_record)
    alignment = _align_recommended_and_executed_actions(
        recommendation["primary_action_types"],
        recommendation["secondary_action_types"],
        executed_action_types,
    )
    exact_question_copies = _find_exact_question_copies(
        changed_templates,
        _feedback_record_questions(feedback_record),
    )

    return {
        "status": "ok",
        "parent_bundle_hash": _sha256_text(parent_bundle_text),
        "proposal_bundle_hash": _sha256_text(proposal_bundle_text),
        "summary": _proposal_change_counts(changed_templates),
        "changed_templates": changed_templates,
        "exact_question_copies": exact_question_copies,
        "recommended_action_types": recommendation["primary_action_types"],
        "secondary_recommended_action_types": recommendation["secondary_action_types"],
        "executed_action_types": executed_action_types,
        "action_alignment": alignment,
        "llm_feedback_normalization": (
            ((feedback_record or {}).get("llm_feedback_debug") or {}).get("normalization") or {}
        ),
    }


def _trace_proposal_diff_summary(diff_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not diff_summary:
        return None
    return {
        "status": diff_summary.get("status"),
        "summary": diff_summary.get("summary"),
        "recommended_action_types": diff_summary.get("recommended_action_types"),
        "secondary_recommended_action_types": diff_summary.get("secondary_recommended_action_types"),
        "executed_action_types": diff_summary.get("executed_action_types"),
        "action_alignment": diff_summary.get("action_alignment"),
        "exact_question_copy_count": len(diff_summary.get("exact_question_copies") or []),
    }


def _load_bundle_yaml(files: dict[str, str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, str]]]:
    loaded: dict[str, dict[str, Any]] = {}
    issues: list[dict[str, str]] = []
    for filename, content in files.items():
        try:
            data = yaml.safe_load(content)
        except Exception as exc:
            issues.append({"filename": filename, "error": str(exc)})
            continue
        if not isinstance(data, dict):
            issues.append({"filename": filename, "error": "yaml_root_not_mapping"})
            continue
        loaded[filename] = data
    return loaded, issues


def _template_allowed_field_changes(parent: dict[str, Any], proposal: dict[str, Any]) -> dict[str, Any]:
    changes: dict[str, Any] = {}
    prototype_change = _string_list_change(parent.get("query_prototypes"), proposal.get("query_prototypes"))
    if prototype_change:
        changes["query_prototypes"] = prototype_change

    hard_negative_change = _hard_negative_change(parent.get("hard_negatives"), proposal.get("hard_negatives"))
    if hard_negative_change:
        changes["hard_negatives"] = hard_negative_change

    threshold_change = _threshold_change(parent.get("thresholds"), proposal.get("thresholds"))
    if threshold_change:
        changes["thresholds"] = threshold_change
    return changes


def _string_list_change(parent: Any, proposal: Any, sample_limit: int = 5) -> dict[str, Any]:
    parent_items = [str(item) for item in parent or [] if isinstance(item, str)]
    proposal_items = [str(item) for item in proposal or [] if isinstance(item, str)]
    parent_set = set(parent_items)
    proposal_set = set(proposal_items)
    added = [item for item in proposal_items if item not in parent_set]
    removed = [item for item in parent_items if item not in proposal_set]
    if not added and not removed:
        return {}
    return {
        "added_count": len(added),
        "removed_count": len(removed),
        "added_samples": added[:sample_limit],
        "removed_samples": removed[:sample_limit],
    }


def _hard_negative_change(parent: Any, proposal: Any, sample_limit: int = 5) -> dict[str, Any]:
    parent_items = [item for item in parent or [] if isinstance(item, dict)]
    proposal_items = [item for item in proposal or [] if isinstance(item, dict)]
    parent_by_key = {_stable_json(item): item for item in parent_items}
    proposal_by_key = {_stable_json(item): item for item in proposal_items}
    added_keys = [key for key in proposal_by_key if key not in parent_by_key]
    removed_keys = [key for key in parent_by_key if key not in proposal_by_key]
    if not added_keys and not removed_keys:
        return {}
    return {
        "added_count": len(added_keys),
        "removed_count": len(removed_keys),
        "added_samples": [proposal_by_key[key] for key in added_keys[:sample_limit]],
        "removed_samples": [parent_by_key[key] for key in removed_keys[:sample_limit]],
    }


def _threshold_change(parent: Any, proposal: Any) -> dict[str, Any]:
    if not isinstance(parent, dict) or not isinstance(proposal, dict):
        return {}
    changed = []
    for name in sorted(set(parent) | set(proposal)):
        parent_value = parent.get(name)
        proposal_value = proposal.get(name)
        if parent_value == proposal_value:
            continue
        parent_float = _maybe_float(parent_value)
        proposal_float = _maybe_float(proposal_value)
        direction = "changed"
        delta = None
        if parent_float is not None and proposal_float is not None:
            delta = round(proposal_float - parent_float, 6)
            direction = "increase" if delta > 0 else "decrease"
        changed.append(
            {
                "name": name,
                "parent": parent_value,
                "proposal": proposal_value,
                "delta": delta,
                "direction": direction,
            }
        )
    return {"changed": changed} if changed else {}


def _proposal_change_counts(changed_templates: list[dict[str, Any]]) -> dict[str, int]:
    counts = {
        "changed_template_count": len(changed_templates),
        "added_query_prototypes": 0,
        "removed_query_prototypes": 0,
        "added_hard_negatives": 0,
        "removed_hard_negatives": 0,
        "changed_thresholds": 0,
    }
    for template in changed_templates:
        changes = template.get("changes") or {}
        prototypes = changes.get("query_prototypes") or {}
        hard_negatives = changes.get("hard_negatives") or {}
        thresholds = changes.get("thresholds") or {}
        counts["added_query_prototypes"] += int(prototypes.get("added_count") or 0)
        counts["removed_query_prototypes"] += int(prototypes.get("removed_count") or 0)
        counts["added_hard_negatives"] += int(hard_negatives.get("added_count") or 0)
        counts["removed_hard_negatives"] += int(hard_negatives.get("removed_count") or 0)
        counts["changed_thresholds"] += len(thresholds.get("changed") or [])
    return counts


def infer_executed_action_types(changed_templates: list[dict[str, Any]]) -> list[str]:
    action_types: set[str] = set()
    for template in changed_templates:
        changes = template.get("changes") or {}
        prototypes = changes.get("query_prototypes") or {}
        hard_negatives = changes.get("hard_negatives") or {}
        thresholds = changes.get("thresholds") or {}
        if int(prototypes.get("added_count") or 0) > 0:
            action_types.add("add_expected_backend_prototypes")
        if int(prototypes.get("removed_count") or 0) > 0:
            action_types.add("narrow_or_remove_overbroad_prototypes")
        if int(hard_negatives.get("added_count") or 0) > 0:
            action_types.add("add_wrong_template_hard_negatives")
        if int(hard_negatives.get("removed_count") or 0) > 0:
            action_types.add("relax_expected_hard_negative_penalty")
        for change in thresholds.get("changed") or []:
            name = str(change.get("name") or "")
            direction = str(change.get("direction") or "")
            if direction == "increase":
                if name in {"hard_negative_margin", "hard_negative_penalty"}:
                    action_types.add("add_wrong_template_hard_negatives")
                else:
                    action_types.add("raise_wrong_accept_or_margin")
            elif direction == "decrease":
                if name in {"hard_negative_margin", "hard_negative_penalty"}:
                    action_types.add("relax_expected_hard_negative_penalty")
                else:
                    action_types.add("lower_expected_accept_or_margin")
    return sorted(action_types)


def _proposal_action_recommendation_summary(feedback_record: dict[str, Any] | None) -> dict[str, list[str]]:
    primary: set[str] = set()
    secondary: set[str] = set()
    recommendations = (
        ((feedback_record or {}).get("deterministic_feedback") or {}).get("batch_action_recommendations") or []
    )
    for recommendation in recommendations:
        if not isinstance(recommendation, dict):
            continue
        action_type = str(recommendation.get("action_type") or "")
        secondary_action_type = str(recommendation.get("secondary_action_type") or "")
        if action_type:
            primary.add(action_type)
        if secondary_action_type:
            secondary.add(secondary_action_type)
    return {
        "primary_action_types": sorted(primary),
        "secondary_action_types": sorted(secondary),
    }


ACTION_TYPE_COMPATIBILITY = {
    "preserve_behavior": {"preserve_behavior"},
    "add_expected_backend_prototypes": {"add_expected_backend_prototypes"},
    "add_wrong_template_hard_negatives": {"add_wrong_template_hard_negatives"},
    "add_discriminative_prototypes_or_hard_negatives": {
        "add_expected_backend_prototypes",
        "add_wrong_template_hard_negatives",
    },
    "lower_expected_accept_or_margin": {"lower_expected_accept_or_margin"},
    "raise_wrong_accept_or_margin": {"raise_wrong_accept_or_margin"},
    "narrow_or_remove_overbroad_prototypes": {"narrow_or_remove_overbroad_prototypes"},
    "relax_expected_hard_negative_penalty": {"relax_expected_hard_negative_penalty"},
}


def _align_recommended_and_executed_actions(
    primary_recommended: list[str],
    secondary_recommended: list[str],
    executed: list[str],
) -> dict[str, list[str]]:
    executed_set = set(executed)
    all_recommended = sorted(set(primary_recommended) | set(secondary_recommended))
    matched_primary = [
        action_type
        for action_type in primary_recommended
        if ACTION_TYPE_COMPATIBILITY.get(action_type, {action_type}) & executed_set
    ]
    matched_secondary = [
        action_type
        for action_type in secondary_recommended
        if ACTION_TYPE_COMPATIBILITY.get(action_type, {action_type}) & executed_set
    ]
    missing_primary = [
        action_type
        for action_type in primary_recommended
        if not (ACTION_TYPE_COMPATIBILITY.get(action_type, {action_type}) & executed_set)
    ]
    extra_executed = [
        action_type
        for action_type in executed
        if not any(
            action_type in ACTION_TYPE_COMPATIBILITY.get(recommended, {recommended})
            for recommended in all_recommended
        )
    ]
    return {
        "matched_primary_action_types": matched_primary,
        "matched_secondary_action_types": matched_secondary,
        "missing_primary_action_types": missing_primary,
        "extra_executed_action_types": extra_executed,
    }


def _feedback_record_questions(feedback_record: dict[str, Any] | None) -> list[str]:
    questions: list[str] = []
    for output in ((feedback_record or {}).get("matcher_outputs") or []):
        if isinstance(output, dict) and output.get("question"):
            questions.append(str(output["question"]))
    for case_feedback in (
        (((feedback_record or {}).get("deterministic_feedback") or {}).get("case_feedback") or [])
    ):
        if isinstance(case_feedback, dict) and case_feedback.get("question"):
            questions.append(str(case_feedback["question"]))
    return questions


def _find_exact_question_copies(
    changed_templates: list[dict[str, Any]],
    questions: list[str],
) -> list[dict[str, Any]]:
    normalized_questions = {_normalize_copy_check_text(question): question for question in questions}
    if not normalized_questions:
        return []
    copies: list[dict[str, Any]] = []
    for template in changed_templates:
        changes = template.get("changes") or {}
        for text in (changes.get("query_prototypes") or {}).get("added_samples") or []:
            normalized = _normalize_copy_check_text(text)
            if normalized in normalized_questions:
                copies.append(
                    {
                        "template_id": template.get("template_id"),
                        "field": "query_prototypes",
                        "text": text,
                        "matched_question": normalized_questions[normalized],
                    }
                )
        for item in (changes.get("hard_negatives") or {}).get("added_samples") or []:
            query = item.get("query") if isinstance(item, dict) else None
            normalized = _normalize_copy_check_text(query)
            if normalized in normalized_questions:
                copies.append(
                    {
                        "template_id": template.get("template_id"),
                        "field": "hard_negatives",
                        "text": query,
                        "matched_question": normalized_questions[normalized],
                    }
                )
    return copies


def _normalize_copy_check_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _maybe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def materialize_template_bundle(bundle_text: str, output_dir: Path) -> ValidationResult:
    parsed = parse_template_bundle(bundle_text)
    if isinstance(parsed, list):
        return ValidationResult(False, parsed, bundle_hash=_sha256_text(bundle_text))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in parsed.items():
        (output_dir / filename).write_text(content.rstrip() + "\n", encoding="utf-8")
    return ValidationResult(True, [], materialized_dir=output_dir, bundle_hash=_sha256_text(bundle_text))


def _without_allowed_fields(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key not in ALLOWED_TOP_LEVEL_FIELDS}


def run_eval_artifact(
    adapter: TemplateYamlGepaAdapter,
    cases: list[RouteCase],
    candidate: dict[str, str],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_batch = adapter.evaluate(cases, candidate, capture_traces=False)
    outputs = eval_batch.outputs
    summary = summarize_outputs(outputs)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "results.json").write_text(
        json.dumps(outputs, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    with (output_dir / "results.jsonl").open("w", encoding="utf-8") as f:
        for output in outputs:
            f.write(json.dumps(output, ensure_ascii=False, default=_json_default) + "\n")
    return summary


def save_candidates(result: Any, run_dir: Path) -> None:
    candidates_dir = run_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    for idx, candidate in enumerate(result.candidates):
        suffix = "_initial" if idx == 0 else ""
        cand_dir = candidates_dir / f"cand_{idx:04d}{suffix}"
        materialize_template_bundle(candidate[COMPONENT_NAME], cand_dir / "templates")
        (cand_dir / "candidate.json").write_text(
            json.dumps({"candidate_idx": idx, "val_score": result.val_aggregate_scores[idx]}, indent=2),
            encoding="utf-8",
        )


def write_selected_report(
    path: Path,
    initial_val: dict[str, Any],
    selected_val: dict[str, Any],
    initial_test: dict[str, Any],
    selected_test: dict[str, Any],
    result: Any,
    adapter: TemplateYamlGepaAdapter,
) -> None:
    def delta(new: dict[str, Any], old: dict[str, Any], key: str) -> float:
        return round(float(new.get(key, 0.0)) - float(old.get(key, 0.0)), 6)

    lines = [
        "# Selected vs Initial",
        "",
        f"Best candidate index: `{result.best_idx}`",
        f"Total candidates: `{result.num_candidates}`",
        f"Total metric calls: `{result.total_metric_calls}`",
        f"Validator failures: `{adapter.validator_failure_count}`",
        f"Candidate eval cache hits: `{adapter.cache_hits}`",
        f"Candidate eval cache misses: `{adapter.cache_misses}`",
        "",
        "| Split | Candidate | GEPA Score | Backend Accuracy | Macro Recall | No Decision Rate | Template Accept Rate |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, name, summary in (
        ("val", "Initial", initial_val),
        ("val", "Selected", selected_val),
        ("test", "Initial", initial_test),
        ("test", "Selected", selected_test),
    ):
        lines.append(
            "| {split} | {name} | {gepa_score:.6f} | {backend_accuracy:.6f} | "
            "{macro_backend_recall:.6f} | {no_decision_rate:.6f} | {template_accept_rate:.6f} |".format(
                split=split,
                name=name,
                gepa_score=float(summary.get("gepa_score", 0.0)),
                backend_accuracy=float(summary.get("backend_accuracy", 0.0)),
                macro_backend_recall=float(summary.get("macro_backend_recall", 0.0)),
                no_decision_rate=float(summary.get("no_decision_rate", 0.0)),
                template_accept_rate=float(summary.get("template_accept_rate", 0.0)),
            )
        )

    lines.extend(
        [
            "",
            "## Delta",
            "",
            f"- Val GEPA Score delta: `{delta(selected_val, initial_val, 'gepa_score')}`",
            f"- Val backend accuracy delta: `{delta(selected_val, initial_val, 'backend_accuracy')}`",
            f"- Test GEPA Score delta: `{delta(selected_test, initial_test, 'gepa_score')}`",
            f"- Test backend accuracy delta: `{delta(selected_test, initial_test, 'backend_accuracy')}`",
            "",
            "## Selected Per-Backend Recall",
            "",
            "```json",
            json.dumps(selected_test.get("per_backend_recall", {}), ensure_ascii=False, indent=2),
            "```",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_cases(path: Path, limit: int | None = None) -> list[RouteCase]:
    cases: list[RouteCase] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            cases.append(
                RouteCase(
                    case_id=str(row.get("case_id") or f"case_{len(cases)}"),
                    sample_id=str(row.get("sample_id") or ""),
                    scenario=str(row.get("scenario") or ""),
                    category=str(row.get("category") or ""),
                    question=str(row["question"]),
                    expected_backend=str(row["expected_backend"]),
                )
            )
            if limit is not None and len(cases) >= limit:
                break
    return cases


def split_summary(train: list[RouteCase], val: list[RouteCase], test: list[RouteCase]) -> dict[str, Any]:
    return {
        "train": summarize_split_cases(train),
        "val": summarize_split_cases(val),
        "test": summarize_split_cases(test),
    }


def summarize_split_cases(cases: list[RouteCase]) -> dict[str, Any]:
    return {
        "count": len(cases),
        "sample_ids": sorted({case.sample_id for case in cases if case.sample_id}),
        "expected_backend": dict(sorted(Counter(case.expected_backend for case in cases).items())),
        "scenario": dict(sorted(Counter(case.scenario for case in cases if case.scenario).items())),
        "category": dict(sorted(Counter(case.category for case in cases if case.category).items())),
    }


def create_embedding_provider(config: dict[str, Any]) -> EmbeddingProvider:
    provider_type = str(config.get("provider") or "mock")
    kwargs: dict[str, Any] = {}
    if provider_type == "mock":
        kwargs["dim"] = int(config.get("dim", 16))
    elif provider_type == "sentence-transformers":
        if config.get("model"):
            kwargs["model_name"] = config["model"]
    elif provider_type == "openai":
        normalized_config = dict(config)
        aliases = {
            "api_base": "base_url",
            "dimension": "output_dimension",
            "max_concurrent": "max_batch_size",
        }
        for old_key, new_key in aliases.items():
            if old_key in normalized_config and new_key not in normalized_config:
                normalized_config[new_key] = normalized_config[old_key]
        for key in ("model", "api_key", "base_url", "output_dimension", "max_batch_size"):
            if normalized_config.get(key) is not None:
                kwargs[key] = normalized_config[key]
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")
    return create_provider(provider_type, **kwargs)


def load_local_config(config_path: Path | None) -> dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return expand_env_vars(yaml.safe_load(f) or {})


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ANGLE_PLACEHOLDER_RE = re.compile(r"^<[^<>]+>$")


def expand_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: expand_env_vars(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(value) for value in obj]
    if isinstance(obj, str):
        return _ENV_RE.sub(lambda match: os.environ.get(match.group(1), match.group(0)), obj)
    return obj


def _is_unresolved_secret(value: str) -> bool:
    stripped = value.strip()
    lowered = stripped.lower()
    if not stripped:
        return False
    if _ENV_RE.search(stripped):
        return True
    if _ANGLE_PLACEHOLDER_RE.match(stripped):
        return True
    return any(token in lowered for token in ("fill", "placeholder", "replace", "todo", "your_", "api_key"))


SENSITIVE_KEYS = {"api_key", "auth_token", "api_key_env"}


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in config.items():
        if isinstance(value, dict):
            redacted[key] = redact_config(value)
        elif key in SENSITIVE_KEYS and isinstance(value, str) and value:
            redacted[key] = "[REDACTED]"
        else:
            redacted[key] = value
    return redacted


def resolve_args_for_resume(raw_args: argparse.Namespace) -> argparse.Namespace:
    if not raw_args.resume_run:
        return raw_args
    if len(sys.argv) > 3:
        raise ValueError("--resume-run cannot be combined with other CLI overrides.")
    run_dir = Path(raw_args.resume_run).resolve()
    snapshot_path = run_dir / "config_snapshot.yaml"
    if not snapshot_path.exists():
        raise ValueError(f"Missing config snapshot: {snapshot_path}")
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8")) or {}
    args_data = snapshot.get("args")
    if not isinstance(args_data, dict):
        raise ValueError("config_snapshot.yaml does not contain args.")
    args_data["resume_run"] = str(run_dir)
    args_data["output_dir"] = str(run_dir)
    return argparse.Namespace(**args_data)


def validate_required_args(args: argparse.Namespace) -> None:
    required = ["train", "val", "test", "template_dir", "runs_dir"]
    missing = [name for name in required if not getattr(args, name, None)]
    if missing:
        raise ValueError(f"Missing required args: {', '.join('--' + name.replace('_', '-') for name in missing)}")


def setup_run_dir(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.output_dir:
        run_dir = Path(args.output_dir).resolve()
    elif args.resume_run:
        run_dir = Path(args.resume_run).resolve()
    else:
        train_name = Path(args.train).stem
        provider = config.get("embedding", {}).get("provider", "unknown")
        model = config.get("embedding", {}).get("model", config.get("embedding", {}).get("dim", "unknown"))
        run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{train_name}_{provider}_{model}_{git_short_sha()}"
        run_dir = Path(args.runs_dir).resolve() / run_id
    if not args.resume_run and (run_dir / "gepa_state.bin").exists():
        raise ValueError(f"Run directory already contains GEPA state; use --resume-run {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ("cache", "tmp", "initial", "selected", "reports", "candidates"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    return run_dir


def git_short_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def build_reflection_lm(
    args: argparse.Namespace,
    config: dict[str, Any],
    debug_logger: JsonlLogger | None = None,
) -> Any:
    proposal_config = dict(config.get("proposal_lm", {}))
    if args.proposal_model:
        proposal_config["model"] = args.proposal_model
    if args.proposal_base_url:
        proposal_config["base_url"] = args.proposal_base_url
    if args.proposal_api_key:
        proposal_config["api_key"] = args.proposal_api_key

    if args.profile == "dry-run" and not proposal_config.get("model"):
        return NoopReflectionLM()
    if not proposal_config.get("model"):
        raise ValueError("proposal_lm.model is required outside dry-run mode.")
    return OpenAIChatLM(proposal_config, "proposal_lm", debug_logger=debug_logger)


def write_cache_manifest(path: Path, args: argparse.Namespace, embedding_config: dict[str, Any], stats: dict[str, int]) -> None:
    manifest = {
        "question_embedding_cache": bool(args.enable_question_embedding_cache),
        "template_embedding_cache": bool(args.enable_template_embedding_cache),
        "candidate_eval_cache": bool(args.enable_candidate_eval_cache),
        "embedding_config": redact_config(embedding_config),
        "stats": stats,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[A-Za-z0-9_-]*\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        obj = json.loads(stripped[start : end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError("LLM Feedback did not return a JSON object.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="dry-run", choices=["dry-run", "full"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--train", default=None)
    parser.add_argument("--val", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--template-dir", default=None)
    parser.add_argument("--matcher-config", default=None)
    parser.add_argument("--runs-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume-run", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--early-stop-rounds", type=int, default=20)
    parser.add_argument("--max-invalid-proposals", type=int, default=5)
    parser.add_argument("--enable-template-embedding-cache", dest="enable_template_embedding_cache", action="store_true", default=True)
    parser.add_argument("--disable-template-embedding-cache", dest="enable_template_embedding_cache", action="store_false")
    parser.add_argument("--enable-question-embedding-cache", dest="enable_question_embedding_cache", action="store_true", default=True)
    parser.add_argument("--disable-question-embedding-cache", dest="enable_question_embedding_cache", action="store_false")
    parser.add_argument("--enable-candidate-eval-cache", dest="enable_candidate_eval_cache", action="store_true", default=True)
    parser.add_argument("--disable-candidate-eval-cache", dest="enable_candidate_eval_cache", action="store_false")
    parser.add_argument("--parallel-val-eval", dest="parallel_val_eval", action="store_true", default=True)
    parser.add_argument("--no-parallel-val-eval", dest="parallel_val_eval", action="store_false")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-metric-calls", type=int, default=None)
    parser.add_argument("--target-accepted-candidates", type=int, default=None)
    parser.add_argument("--dry-run-train-limit", type=int, default=48)
    parser.add_argument("--dry-run-val-limit", type=int, default=24)
    parser.add_argument("--dry-run-test-limit", type=int, default=24)
    parser.add_argument("--proposal-model", default=None)
    parser.add_argument("--proposal-base-url", default=None)
    parser.add_argument("--proposal-api-key", default=None)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("echomem.templates").setLevel(logging.WARNING)
    logging.getLogger("echomem.matcher").setLevel(logging.WARNING)
    logging.getLogger("echomem.features").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    raw_args = build_arg_parser().parse_args()
    args = resolve_args_for_resume(raw_args)
    validate_required_args(args)

    config = load_local_config(Path(args.config).resolve() if args.config else None)
    run_dir = setup_run_dir(args, config)
    snapshot = {
        "args": vars(args),
        "config_redacted": redact_config(config),
        "git_sha": git_short_sha(),
    }
    (run_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    train_limit = args.dry_run_train_limit if args.profile == "dry-run" else None
    val_limit = args.dry_run_val_limit if args.profile == "dry-run" else None
    test_limit = args.dry_run_test_limit if args.profile == "dry-run" else None
    train_cases = load_cases(Path(args.train), train_limit)
    val_cases = load_cases(Path(args.val), val_limit)
    test_cases = load_cases(Path(args.test), test_limit)
    splits = split_summary(train_cases, val_cases, test_cases)
    (run_dir / "split_summary.json").write_text(
        json.dumps(splits, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    initial_bundle_text, initial_files = read_template_bundle(Path(args.template_dir))

    embedding_config = dict(config.get("embedding", {"provider": "mock", "dim": 16}))
    base_embedder = create_embedding_provider(embedding_config)
    embedding_config_hash = _sha256_text(json.dumps(redact_config(embedding_config), sort_keys=True))
    cache_stats = {"hits": 0, "misses": 0}
    query_embedder = DiskEmbeddingCacheProvider(
        base_embedder,
        run_dir / "cache" / "question_embeddings",
        namespace=f"question:{embedding_config_hash}",
        enabled=bool(args.enable_question_embedding_cache),
        stats=cache_stats,
    )

    def template_embedder_factory(bundle_hash: str) -> EmbeddingProvider:
        return DiskEmbeddingCacheProvider(
            base_embedder,
            run_dir / "cache" / "template_embeddings",
            namespace=f"template:{bundle_hash}:{embedding_config_hash}",
            enabled=bool(args.enable_template_embedding_cache),
            stats=cache_stats,
        )

    validator = TemplateBundleValidator(
        initial_bundle_text=initial_bundle_text,
        initial_files=initial_files,
        tmp_root=run_dir / "tmp",
        query_embedder=query_embedder,
        template_embedder_factory=template_embedder_factory,
        smoke_cases=train_cases,
    )
    evaluator = MatcherOnlyEvaluator(
        validator=validator,
        query_embedder=query_embedder,
        template_embedder_factory=template_embedder_factory,
        num_threads=args.num_threads,
        parallel_eval=bool(args.parallel_val_eval),
    )
    trace_logger = JsonlLogger(run_dir / "reports" / "optimization_trace.jsonl")
    feedback_logger = JsonlLogger(run_dir / "reports" / "feedback_records.jsonl")
    lm_debug_logger = JsonlLogger(run_dir / "reports" / "lm_calls.jsonl")

    initial_candidate = {COMPONENT_NAME: initial_bundle_text}
    placeholder_adapter = TemplateYamlGepaAdapter(
        evaluator=evaluator,
        llm_feedback=LLMFeedbackGenerator({"enabled": False}),
        cache_dir=run_dir / "cache",
        candidate_eval_cache_enabled=bool(args.enable_candidate_eval_cache),
        initial_summary={},
        parent_val_summary={},
        trace_logger=trace_logger,
    )
    initial_train_summary = run_eval_artifact(
        placeholder_adapter,
        train_cases,
        initial_candidate,
        run_dir / "initial" / "train_eval",
    )
    initial_val_summary = run_eval_artifact(
        placeholder_adapter,
        val_cases,
        initial_candidate,
        run_dir / "initial" / "val_eval",
    )
    initial_test_summary = run_eval_artifact(
        placeholder_adapter,
        test_cases,
        initial_candidate,
        run_dir / "initial" / "test_eval",
    )

    initial_train_results = json.loads((run_dir / "initial" / "train_eval" / "results.json").read_text(encoding="utf-8"))
    buckets = build_sampling_buckets(initial_train_results)
    (run_dir / "sampling_buckets.json").write_text(
        json.dumps(buckets, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    adapter = TemplateYamlGepaAdapter(
        evaluator=evaluator,
        llm_feedback=LLMFeedbackGenerator(config.get("llm_feedback", {}), debug_logger=lm_debug_logger),
        cache_dir=run_dir / "cache",
        candidate_eval_cache_enabled=bool(args.enable_candidate_eval_cache),
        initial_summary={
            "train": initial_train_summary,
            "val": initial_val_summary,
        },
        parent_val_summary=initial_val_summary,
        trace_logger=trace_logger,
        feedback_logger=feedback_logger,
    )

    try:
        import gepa
        from gepa.utils import NoImprovementStopper
        from gepa.utils.stop_condition import MaxTrackedCandidatesStopper
    except ImportError as exc:
        raise ImportError("GEPA is required. Install the gepa package before running this script.") from exc

    max_metric_calls = args.max_metric_calls
    if max_metric_calls is None:
        max_metric_calls = 96 if args.profile == "dry-run" else 5000
    stop_callbacks: list[Any] = [
        NoImprovementStopper(args.early_stop_rounds),
        InvalidProposalStopper(adapter, args.max_invalid_proposals),
    ]
    if args.target_accepted_candidates:
        stop_callbacks.append(MaxTrackedCandidatesStopper(args.target_accepted_candidates + 1))

    batch_sampler = StratifiedBatchSampler(
        buckets=buckets,
        minibatch_size=args.batch_size,
        seed=args.seed,
    )
    reflection_lm = build_reflection_lm(args, config, debug_logger=lm_debug_logger)
    callbacks = [
        GepaTraceCallback(
            trace_logger=trace_logger,
            candidate_csv=run_dir / "reports" / "candidate_scores.csv",
            proposal_dir=run_dir / "reports" / "proposals",
            adapter=adapter,
            initial_val_summary=initial_val_summary,
        )
    ]

    gepa_logger = FileOnlyLogger(run_dir / "reports" / "gepa_run_log.txt")
    result = gepa.optimize(
        seed_candidate=initial_candidate,
        trainset=train_cases,
        valset=val_cases,
        adapter=adapter,
        reflection_lm=reflection_lm,
        candidate_selection_strategy="current_best",
        batch_sampler=batch_sampler,
        module_selector="all",
        skip_perfect_score=True,
        perfect_score=1.0,
        reflection_prompt_template=GENERATION_PROMPT,
        max_metric_calls=max_metric_calls,
        stop_callbacks=stop_callbacks,
        logger=gepa_logger,
        run_dir=str(run_dir),
        callbacks=callbacks,
        display_progress_bar=True,
        # GEPA's per-example cache is unsafe here: our score is a batch-level
        # aggregate repeated per example, and train/val DataLoader ids overlap.
        # Use the adapter-level cache keyed by bundle hash and case ids instead.
        cache_evaluation=False,
        seed=args.seed,
        raise_on_exception=False,
    )

    (run_dir / "gepa_result.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    save_candidates(result, run_dir)

    selected_candidate = result.best_candidate
    materialize_template_bundle(selected_candidate[COMPONENT_NAME], run_dir / "selected" / "templates")
    selected_val_summary = run_eval_artifact(
        adapter,
        val_cases,
        selected_candidate,
        run_dir / "selected" / "val_eval",
    )
    selected_test_summary = run_eval_artifact(
        adapter,
        test_cases,
        selected_candidate,
        run_dir / "selected" / "test_eval",
    )
    write_selected_report(
        run_dir / "reports" / "selected_vs_initial.md",
        initial_val=initial_val_summary,
        selected_val=selected_val_summary,
        initial_test=initial_test_summary,
        selected_test=selected_test_summary,
        result=result,
        adapter=adapter,
    )
    write_cache_manifest(run_dir / "cache_manifest.json", args, embedding_config, cache_stats)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "best_candidate_idx": result.best_idx,
                "num_candidates": result.num_candidates,
                "total_metric_calls": result.total_metric_calls,
                "validator_failure_count": adapter.validator_failure_count,
                "cache_stats": cache_stats,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(json.dumps({"run_dir": str(run_dir), "best_candidate_idx": result.best_idx}, indent=2))
    return 0


OUTPUT_CONTRACT = """Proposal must be a complete Template Bundle.
- Use one '# FILE: <name>.yaml' marker before each YAML file.
- Keep the same file names and the same template order.
- Return YAML only. Do not add Markdown fences or explanation text.
- Do not return a subset of files. The Proposal is invalid unless every Parent '# FILE:' marker appears exactly once.
- Continue the output until the final Parent file has been fully emitted.
- Only edit query_prototypes, hard_negatives, and thresholds.
- Do not edit template_id, target, query_spec, backend names, status, semantic_card, intent_family, or calibration.
- Do not add or delete templates.
- Do not copy exact minibatch questions as prototypes.
"""


GENERATION_PROMPT = """You are optimizing a YAML Template Bundle for a deterministic matcher.

Goal:
Improve matcher-only routing accuracy on future samples. There is no LLM fallback. If the matcher does not accept a backend, the result is No Decision.

Output:
Return the complete Template Bundle only.

Output Contract:
{output_contract}

Allowed changes:
- query_prototypes
- hard_negatives
- thresholds

Forbidden changes:
- template_id
- target
- query_spec
- backend names
- adding or deleting templates
- copying exact training questions
- adding explanation text or Markdown fences

Decision rule:
Prefer generalized template improvements that fix the current minibatch while preserving correct behavior. Validator Feedback has highest priority. Correct cases are regression anchors.

CURRENT TEMPLATE BUNDLE START
<curr_instructions>
CURRENT TEMPLATE BUNDLE END

Feedback records:
<inputs_outputs_feedback>

Return the complete Template Bundle only.
""".format(output_contract=OUTPUT_CONTRACT)


LLM_FEEDBACK_PROMPT = """You generate high-level feedback for improving matcher YAML templates.

Use only the JSON below. Do not output YAML. Do not copy exact questions, names, dates, or event details.
Use deterministic action_recommendation as evidence, but keep your feedback general.
When naming editable fields, use only query_prototypes, hard_negatives, or thresholds.
When naming actions, use only add, rewrite, remove, increase, decrease, or preserve.

Output JSON with keys:
- feedback_type
- summary
- template_suggestions: list of objects with template_id, field, action, guidance
- preserve: list of short regression-preservation reminders

Input JSON:
<DETERMINISTIC_FEEDBACK_JSON>
"""


if __name__ == "__main__":
    raise SystemExit(main())
