"""MemRouter evaluation script.

Implements the v1.4 evaluation pipeline:
    1. Load dataset and validate schema.
    2. Run MemRouter on each query.
    3. Compute metrics and produce artifacts.

Usage:
    python scripts/eval_memrouter.py \
        --dataset data/memrouter_eval/smoke_routes.jsonl \
        --mode ci-smoke-mock \
        --embedding-provider mock \
        --llm-provider mock
"""

import argparse
import json
import logging

import numpy as np
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Add parent directory to path so we can import echomem
sys.path.insert(0, str(Path(__file__).parent.parent))

from echomem.embeddings.base import create_provider
from echomem.llm_fallback import LLMRouterConfig, create_llm_backend_router
from echomem.pipeline import MemRouterPipeline
from echomem.result import MemBackendRouteResult

logger = logging.getLogger("eval_memrouter")

# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MemRouter routing evaluation")
    parser.add_argument("--dataset", required=True, help="Path to dataset JSONL")
    parser.add_argument("--runs-dir", default="runs/memrouter", help="Base runs directory")
    parser.add_argument("--output-dir", default=None, help="Override auto-generated run directory")
    parser.add_argument("--embedding-provider", default=None, help="Embedding provider: mock, openai, sentence-transformers (overrides config)")
    parser.add_argument("--embedding-model", default=None, help="Override embedding model")
    parser.add_argument("--llm-provider", default=None, help="LLM provider: mock, openai_compatible, anthropic_compatible (overrides config)")
    parser.add_argument("--llm-model", default=None, help="Override LLM model")
    parser.add_argument("--mode", default="full_route_eval", choices=["template_eval", "full_route_eval", "ci-smoke-mock", "ci-smoke-real"])
    parser.add_argument("--config", default=None, help="Path to local config YAML (e.g., configs/memrouter_eval.local.yaml)")
    parser.add_argument("--allow-real-llm", action="store_true", help="Allow real LLM calls in template_eval mode (default: mock only)")
    parser.add_argument("--filter-backend", default=None, help="Filter by expected backend")
    parser.add_argument("--filter-scenario", default=None, help="Filter by scenario")
    parser.add_argument("--filter-benchmark", default=None, help="Filter by benchmark")
    parser.add_argument("--filter-case-id", action="append", default=None, help="Filter by case_id (repeatable)")
    return parser


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #


def load_local_config(config_path: Path) -> Dict[str, Any]:
    """Load local evaluation config YAML."""
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ANGLE_PLACEHOLDER_RE = re.compile(r"^<[^<>]+>$")
_PLACEHOLDER_TOKENS = (
    "fill",
    "placeholder",
    "replace",
    "todo",
    "your_",
    "your-",
    "api_key",
    "auth_token",
    "key>",
)


def _expand_env_in_string(value: str) -> str:
    """Expand ${ENV_VAR} placeholders using os.environ."""
    def _replacer(match: re.Match[str]) -> str:
        env_name = match.group(1)
        env_value = os.environ.get(env_name)
        if env_value is None:
            return match.group(0)
        return env_value
    return _ENV_RE.sub(_replacer, value)


def expand_env_vars(obj: Any) -> Any:
    """Recursively expand ${ENV_VAR} placeholders in config dicts/lists/strings."""
    if isinstance(obj, dict):
        return {k: expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_vars(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_in_string(obj)
    return obj


def require_resolved_secret(value: Any, field_name: str) -> None:
    """Fail early when a secret is missing or still contains a placeholder."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} is required")
    if _ENV_RE.search(value):
        raise ValueError(f"{field_name} contains an unresolved environment placeholder")
    stripped = value.strip()
    lowered = stripped.lower()
    if _ANGLE_PLACEHOLDER_RE.match(stripped):
        raise ValueError(f"{field_name} contains a placeholder value")
    if any(token in lowered for token in _PLACEHOLDER_TOKENS):
        raise ValueError(f"{field_name} contains a placeholder value")


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Merge CLI args with local config. CLI takes precedence."""
    config: Dict[str, Any] = {}
    if args.config:
        config = load_local_config(Path(args.config))
        config = expand_env_vars(config)

    # Embedding overrides (CLI takes precedence only when explicitly set)
    emb = config.get("embedding", {})
    if args.embedding_provider is not None:
        emb["provider"] = args.embedding_provider
    if args.embedding_model is not None:
        emb["model"] = args.embedding_model
    config["embedding"] = emb

    # LLM overrides
    llm = config.get("llm", {})
    if args.llm_provider is not None:
        llm["provider"] = args.llm_provider
    if args.llm_model is not None:
        llm["model"] = args.llm_model
    config["llm"] = llm

    # Apply mode-based defaults only when no provider is configured anywhere
    mode = getattr(args, "mode", "full_route_eval")
    if mode == "ci-smoke-mock":
        if not config.get("embedding", {}).get("provider"):
            config["embedding"]["provider"] = "mock"
        if not config.get("llm", {}).get("provider"):
            config["llm"]["provider"] = "mock"

    # template_eval forces mock LLM unless --allow-real-llm is explicitly set
    if mode == "template_eval" and not getattr(args, "allow_real_llm", False):
        config["llm"]["provider"] = "mock"
        logger.info("template_eval mode: forcing llm.provider=mock (use --allow-real-llm to override)")

    # template_eval requires an embedding provider (from config or CLI)
    if mode == "template_eval":
        emb_provider = config.get("embedding", {}).get("provider")
        if not emb_provider:
            raise ValueError(
                "template_eval requires embedding.provider. "
                "Provide it via --config configs/memrouter_eval.local.yaml or --embedding-provider."
            )

    return config


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #


SENSITIVE_KEYS = {"api_key", "auth_token", "api_key_env"}


def redact_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a deep copy of config with sensitive keys redacted."""
    result: Dict[str, Any] = {}
    for k, v in config.items():
        if isinstance(v, dict):
            result[k] = redact_config(v)
        elif k in SENSITIVE_KEYS and isinstance(v, str) and v:
            result[k] = _redact_value(v)
        else:
            result[k] = v
    return result


def _redact_value(value: str) -> str:
    return "[REDACTED]"


# --------------------------------------------------------------------------- #
# Run directory
# --------------------------------------------------------------------------- #


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


def generate_run_id(dataset_name: str, embed_provider: str | None, embed_model_short: str, llm_provider: str | None) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    git_sha = git_short_sha()
    # Clean provider names for filesystem safety
    ep = (embed_provider or "unknown").replace("-", "")
    lp = (llm_provider or "unknown").replace("-", "_").replace(".", "_")
    return f"{timestamp}_{dataset_name}_{ep}_{embed_model_short}_{lp}_{git_sha}"


def setup_run_dir(args: argparse.Namespace, config: Dict[str, Any]) -> Path:
    """Create run directory and return its path."""
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        runs_dir = Path(args.runs_dir)
        dataset_name = Path(args.dataset).stem
        embed_provider = config.get("embedding", {}).get("provider", args.embedding_provider)
        embed_model_short = config.get("embedding", {}).get("model_short", config.get("embedding", {}).get("model", "unknown"))
        llm_provider = config.get("llm", {}).get("provider_short", config.get("llm", {}).get("provider", args.llm_provider))
        run_id = generate_run_id(dataset_name, embed_provider, embed_model_short, llm_provider)
        run_dir = runs_dir / run_id

    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# --------------------------------------------------------------------------- #
# Dataset loading
# --------------------------------------------------------------------------- #


def load_dataset(dataset_path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


def validate_sample(sample: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a single sample. Returns (is_valid, error_message)."""
    required_top = {"case_id", "benchmark", "question", "expected", "eval_policy", "is_hard_negative"}
    for key in required_top:
        if key not in sample:
            return False, f"missing required field: {key}"

    expected = sample.get("expected", {})
    required_expected = {"primary_backend_id", "expected_routing_mode"}
    for key in required_expected:
        if key not in expected:
            return False, f"missing expected.{key}"

    eval_policy = sample.get("eval_policy", {})
    if "allow_llm_fallback" not in eval_policy:
        return False, "missing eval_policy.allow_llm_fallback"

    # Detect contradictory config
    if expected.get("expected_routing_mode") == "llm_expected" and eval_policy.get("allow_llm_fallback") is False:
        return False, "contradictory config: llm_expected with allow_llm_fallback=false"

    return True, None


# --------------------------------------------------------------------------- #
# Embedding provider factory (with field mapping)
# --------------------------------------------------------------------------- #


def create_embedding_provider(emb_config: Dict[str, Any]):
    provider_type = emb_config.get("provider", "mock")
    kwargs: Dict[str, Any] = {}

    if provider_type == "mock":
        kwargs["dim"] = emb_config.get("dimension", 16)
    elif provider_type == "openai":
        kwargs["model"] = emb_config.get("model", "text-embedding-3-small")
        kwargs["api_key"] = emb_config.get("api_key")
        require_resolved_secret(kwargs["api_key"], "embedding.api_key")
        # Map api_base -> base_url
        if "api_base" in emb_config:
            kwargs["base_url"] = emb_config["api_base"]
        # Only pass output_dimension if explicitly non-default
        model = kwargs["model"]
        requested_dim = emb_config.get("dimension")
        if requested_dim is not None:
            from echomem.embeddings.base import OPENAI_EMBEDDING_MODEL_SPECS
            spec = OPENAI_EMBEDDING_MODEL_SPECS.get(model)
            default_dim = spec.dimension if spec else None
            if default_dim is None or requested_dim != default_dim:
                kwargs["output_dimension"] = requested_dim
        # Batch size limit (e.g. DashScope limits to 10)
        batch_size = emb_config.get("max_batch_size") or emb_config.get("max_concurrent")
        if batch_size is not None:
            kwargs["max_batch_size"] = batch_size
    elif provider_type == "sentence-transformers":
        kwargs["model_name"] = emb_config.get("model", "all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedding provider: {provider_type}")

    return create_provider(provider_type, **kwargs)


# --------------------------------------------------------------------------- #
# LLM router factory (with field mapping)
# --------------------------------------------------------------------------- #


def create_llm_config(llm_config: Dict[str, Any]) -> LLMRouterConfig:
    provider = llm_config.get("provider", "mock")
    model = llm_config.get("model", "mock")

    kwargs: Dict[str, Any] = {
        "provider": provider,
        "model": model,
    }

    if provider != "mock":
        # Map auth_token -> api_key
        if "auth_token" in llm_config:
            kwargs["api_key"] = llm_config["auth_token"]
        if "api_key" in llm_config:
            kwargs["api_key"] = llm_config["api_key"]
        require_resolved_secret(kwargs.get("api_key"), "llm.api_key")
        if "base_url" in llm_config:
            kwargs["base_url"] = llm_config["base_url"]
        if "timeout_ms" in llm_config:
            kwargs["timeout_seconds"] = int(llm_config["timeout_ms"] / 1000)
        if "temperature" in llm_config:
            kwargs["temperature"] = float(llm_config["temperature"])
        if "max_tokens" in llm_config:
            kwargs["max_tokens"] = int(llm_config["max_tokens"])

    return LLMRouterConfig(**kwargs)


# --------------------------------------------------------------------------- #
# LLM preflight
# --------------------------------------------------------------------------- #

def preflight_llm_fallback(
    llm_router: Any,
    registry: Any,
) -> None:
    """Run a single-shot LLM fallback preflight to verify API connectivity.

    Raises SystemExit if the LLM call fails (auth error, timeout, etc.)
    so that a full eval is not wasted on a broken fallback path.
    """
    from echomem.llm_fallback import LLMFallbackContext
    from echomem.result import QueryHints

    dummy_context = LLMFallbackContext(
        raw_user_query="What is my favorite color?",
        normalized_user_query="what is my favorite color",
        registry=registry,
        failed_template_summary=[],
        query_hints=QueryHints(),
        fallback_reason="preflight_check",
    )
    try:
        result = llm_router.route(dummy_context)
        raw = result.model_dump()
        fallback = raw.get("fallback") or {}
        reason = fallback.get("reason", "")
        if "llm_api_error" in reason:
            raise RuntimeError(f"LLM API error during preflight: {reason}")
        if "llm_output_invalid_or_empty" in reason:
            # Preflight is allowed to hit the default fallback as long as the
            # API layer itself responded. We only guard against *API* errors.
            pass
        logger.info(
            "LLM preflight OK: backend=%s latency_ms=%s",
            result.routes[0].backend_id if result.routes else "none",
            (raw.get("debug") or {}).get("llm_fallback_meta", {}).get("latency_ms", "unknown"),
        )
    except Exception as exc:
        logger.error("LLM preflight FAILED: %s", exc)
        logger.error("Aborting evaluation. Fix LLM connectivity before running full_route_eval.")
        raise SystemExit(1) from exc


# --------------------------------------------------------------------------- #
# Pipeline runner
# --------------------------------------------------------------------------- #


def run_evaluation(cases: List[Dict[str, Any]], pipeline: MemRouterPipeline) -> List[Dict[str, Any]]:
    """Run MemRouter on each case and collect results."""
    results: List[Dict[str, Any]] = []

    for case in cases:
        case_id = case["case_id"]
        question = case["question"]

        try:
            # Measure embedding time separately (cache-aware)
            normalized = pipeline._feature_builder._normalizer.normalize(question)
            emb_ms = 0
            if normalized not in pipeline._feature_builder._embedding_cache:
                start_emb = time.perf_counter()
                emb_vec = pipeline._feature_builder._embedder.embed([normalized])[0]
                emb_ms = int((time.perf_counter() - start_emb) * 1000)
                pipeline._feature_builder._embedding_cache[normalized] = emb_vec

            # Run full pipeline (embedding comes from cache)
            start = time.perf_counter()
            route_result = pipeline.route(question)
            latency_ms = int((time.perf_counter() - start) * 1000)
        except Exception as exc:
            logger.error("Pipeline exception for case %s: %s", case_id, exc)
            results.append({
                "case_id": case_id,
                "error": True,
                "error_type": "pipeline_exception",
                "error_message": str(exc),
            })
            continue

        results.append({
            "case_id": case_id,
            "case": case,
            "route_result": route_result,
            "latency_ms": latency_ms,
            "embedding_ms": emb_ms,
        })

    return results


# --------------------------------------------------------------------------- #
# Result builders
# --------------------------------------------------------------------------- #


def _check_backend_correct(
    actual_backend: str | None,
    expected_backend: str,
    strict: bool,
    registry,
) -> bool:
    """Check backend correctness according to eval_policy."""
    if actual_backend is None:
        return False
    if strict:
        return actual_backend == expected_backend
    # Loose mode: same backend or same backend_kind
    if actual_backend == expected_backend:
        return True
    actual_entry = registry.get(actual_backend)
    expected_entry = registry.get(expected_backend)
    if actual_entry and expected_entry:
        return actual_entry.backend_kind == expected_entry.backend_kind
    return False


def build_route_results(
    run_id: str,
    cases: List[Dict[str, Any]],
    eval_results: List[Dict[str, Any]],
    pipeline: MemRouterPipeline,
) -> List[Dict[str, Any]]:
    """Build route_results.jsonl entries."""
    entries: List[Dict[str, Any]] = []
    registry = pipeline._registry
    template_index = pipeline._template_index

    for case, eval_res in zip(cases, eval_results):
        if eval_res.get("error"):
            entries.append({
                "run_id": run_id,
                "case_id": case["case_id"],
                "benchmark": case.get("benchmark", ""),
                "scenario": case.get("scenario", ""),
                "expected_routing_mode": case.get("expected", {}).get("expected_routing_mode", ""),
                "question": case["question"],
                "expected_primary_backend_id": case["expected"]["primary_backend_id"],
                "actual_primary_backend_id": None,
                "actual_route_method": None,
                "is_primary_backend_correct": None,
                "is_final_correct": None,
                "used_llm_fallback": None,
                "invalid_route_reason": None,
                "latency_ms": {"total": 0, "normalization": 0, "embedding": 0, "template_matching": 0, "decision": 0, "llm_fallback": 0},
                "token_usage": {},
                "top_templates": [],
                "top_template_scores": [],
                "raw_route_result": None,
                "error_type": eval_res.get("error_type"),
                "error_message": eval_res.get("error_message"),
            })
            continue

        route_result: MemBackendRouteResult = eval_res["route_result"]
        raw = route_result.model_dump()

        actual_backend = None
        if route_result.routes:
            actual_backend = route_result.routes[0].backend_id

        expected_backend = case["expected"]["primary_backend_id"]

        eval_policy = case.get("eval_policy", {})
        strict_backend = eval_policy.get("strict_primary_backend", True)

        # Validate for invalid route
        invalid_reason = None
        if not route_result.routes:
            invalid_reason = "routes_empty"
        else:
            seen_backends = set()
            primaries = [r for r in route_result.routes if r.role == "primary"]
            if len(primaries) != 1:
                invalid_reason = f"expected 1 primary, got {len(primaries)}"
            for r in route_result.routes:
                if r.backend_id in seen_backends:
                    invalid_reason = f"duplicate backend: {r.backend_id}"
                    break
                seen_backends.add(r.backend_id)
                if not registry.is_enabled(r.backend_id):
                    invalid_reason = f"unregistered_or_disabled_backend: {r.backend_id}"
                    break

        if invalid_reason:
            is_backend_correct = None
            is_final_correct = False
        else:
            is_backend_correct = _check_backend_correct(
                actual_backend, expected_backend, strict_backend, registry
            )
            is_final_correct = bool(is_backend_correct)
            if route_result.route_method == "llm_backend_fallback" and not eval_policy.get("allow_llm_fallback", True):
                is_final_correct = False

        # Build top_template_scores with score_components and thresholds
        top_templates = (raw.get("debug") or {}).get("top_templates", [])
        top_template_scores = []
        for t in top_templates:
            template = template_index.get(t.get("template_id", ""))
            thresholds = {}
            if template:
                thresholds = {
                    "accept": template.thresholds.accept,
                    "fallback": template.thresholds.fallback,
                    "margin": template.thresholds.margin,
                    "hard_negative_margin": template.thresholds.hard_negative_margin,
                    "hard_negative_penalty": template.thresholds.hard_negative_penalty,
                }
            score_entry = {
                "template_id": t.get("template_id"),
                "backend_id": t.get("backend_id"),
                "score": t.get("score"),
            }
            if "score_components" in t:
                score_entry.update(t["score_components"])
            score_entry["thresholds"] = thresholds
            top_template_scores.append(score_entry)

        entries.append({
            "run_id": run_id,
            "case_id": case["case_id"],
            "benchmark": case.get("benchmark", ""),
            "scenario": case.get("scenario", ""),
            "expected_routing_mode": case.get("expected", {}).get("expected_routing_mode", ""),
            "question": case["question"],
            "expected_primary_backend_id": expected_backend,
            "actual_primary_backend_id": actual_backend,
            "actual_route_method": route_result.route_method,
            "is_primary_backend_correct": is_backend_correct,
            "is_final_correct": is_final_correct,
            "used_llm_fallback": route_result.fallback.used,
            "invalid_route_reason": invalid_reason,
            "latency_ms": {
                "total": eval_res["latency_ms"] + eval_res.get("embedding_ms", 0),
                "normalization": 0,
                "embedding": eval_res.get("embedding_ms", 0),
                "template_matching": 0,
                "decision": 0,
                "llm_fallback": (
                    ((raw.get("debug") or {}).get("llm_fallback_meta") or {}).get("latency_ms", 0)
                    if route_result.fallback.used else 0
                ),
            },
            "token_usage": ((raw.get("debug") or {}).get("llm_fallback_meta") or {}).get("token_usage", {}),
            "top_templates": top_templates,
            "top_template_scores": top_template_scores,
            "raw_route_result": raw,
        })

    return entries


# --------------------------------------------------------------------------- #
# Metrics computation
# --------------------------------------------------------------------------- #


def compute_metrics(
    route_results: List[Dict[str, Any]],
    cases: List[Dict[str, Any]],
    total_original: Optional[int] = None,
) -> Dict[str, Any]:
    """Compute metrics_summary from route results.

    Args:
        route_results: List of route result entries (may include error entries).
        cases: Valid cases that were fed to the pipeline.
        total_original: Total cases loaded from dataset before filtering/validation.
            If None, defaults to len(cases).
    """
    total_cases = total_original if total_original is not None else len(cases)
    evaluated_cases = 0
    skipped_cases: Dict[str, int] = {}

    backend_correct = 0
    final_route_correct = 0
    template_route_correct = 0
    template_preferred_total = 0
    template_preferred_hit = 0
    llm_expected_total = 0
    llm_expected_hit = 0
    llm_fallback_count = 0
    llm_api_error_count = 0
    llm_default_fallback_count = 0
    invalid_route_count = 0
    fallback_violation_count = 0
    multi_backend_secondary_correct = 0
    multi_backend_secondary_total = 0

    # Grouped accumulators for detailed breakdown
    by_benchmark: Dict[str, Dict[str, Any]] = {}
    by_scenario: Dict[str, Dict[str, Any]] = {}
    by_expected_backend: Dict[str, Dict[str, Any]] = {}
    by_route_method: Dict[str, Dict[str, Any]] = {}
    template_scores: Dict[str, List[float]] = {}
    templates_topk: set[str] = set()
    templates_accepted: set[str] = set()

    def _ensure_group(groups: Dict[str, Dict[str, Any]], key: str) -> Dict[str, Any]:
        if key not in groups:
            groups[key] = {
                "count": 0,
                "backend_correct": 0,
                "final_correct": 0,
                "template_hit": 0,
            }
        return groups[key]

    for rr in route_results:
        if rr.get("error_type") == "schema_validation_error":
            skipped_cases["schema_validation_error"] = skipped_cases.get("schema_validation_error", 0) + 1
            continue
        if rr.get("error_type") == "pipeline_exception":
            skipped_cases["pipeline_exception"] = skipped_cases.get("pipeline_exception", 0) + 1
            continue

        evaluated_cases += 1

        # Find matching case for expected info
        case = next((c for c in cases if c["case_id"] == rr["case_id"]), None)
        if not case:
            continue

        # Invalid route
        if rr.get("is_primary_backend_correct") is None:
            invalid_route_count += 1
            continue

        expected_backend = case["expected"]["primary_backend_id"]
        is_backend_ok = rr["is_primary_backend_correct"]
        route_method = rr["actual_route_method"]
        routing_mode = case["expected"].get("expected_routing_mode", "template_preferred")
        eval_policy = case.get("eval_policy", {})
        benchmark = case.get("benchmark", "unknown")
        scenario = case.get("scenario", "unknown")

        # Fallback policy violation
        if route_method == "llm_backend_fallback" and not eval_policy.get("allow_llm_fallback", True):
            fallback_violation_count += 1

        # LLM fallback count
        if route_method == "llm_backend_fallback":
            llm_fallback_count += 1
            raw_result = rr.get("raw_route_result") or {}
            fallback_reason = (raw_result.get("fallback") or {}).get("reason", "")
            if "llm_api_error" in fallback_reason:
                llm_api_error_count += 1
            if "llm_output_invalid_or_empty" in fallback_reason:
                llm_default_fallback_count += 1

        # Primary backend accuracy
        if is_backend_ok:
            backend_correct += 1

        # Final route accuracy (policy violations override correctness)
        is_final_correct = bool(rr.get("is_final_correct"))
        if is_final_correct:
            final_route_correct += 1

        # Template route accuracy
        is_template = route_method in ("template_embedding", "template_embedding_multi_backend")
        if is_template and is_final_correct:
            template_route_correct += 1
        if is_template:
            for route in rr.get("raw_route_result", {}).get("routes", []):
                matched_template_id = route.get("matched_template_id")
                if matched_template_id:
                    templates_accepted.add(matched_template_id)

        # Template ideal rate
        if routing_mode == "template_preferred":
            template_preferred_total += 1
            if is_template and is_final_correct:
                template_preferred_hit += 1

        # LLM expected hit rate
        if routing_mode == "llm_expected":
            llm_expected_total += 1
            if route_method == "llm_backend_fallback" and is_final_correct:
                llm_expected_hit += 1

        # Multi-backend secondary accuracy
        expected_secondaries = case["expected"].get("secondary_backend_ids", [])
        if expected_secondaries and route_method == "template_embedding_multi_backend":
            multi_backend_secondary_total += 1
            actual_backends = [r["backend_id"] for r in rr.get("raw_route_result", {}).get("routes", [])]
            if any(sb in actual_backends for sb in expected_secondaries):
                multi_backend_secondary_correct += 1

        # Grouped stats
        bm = _ensure_group(by_benchmark, benchmark)
        sc = _ensure_group(by_scenario, scenario)
        be = _ensure_group(by_expected_backend, expected_backend)
        rm = _ensure_group(by_route_method, route_method or "unknown")

        for g in (bm, sc, be, rm):
            g["count"] += 1
            if is_backend_ok:
                g["backend_correct"] += 1
            if is_final_correct:
                g["final_correct"] += 1
            if is_template:
                g["template_hit"] += 1

        # Template coverage and score distribution
        top_templates = rr.get("top_templates", [])
        for t in top_templates:
            tid = t.get("template_id", "")
            score = t.get("score")
            if tid and score is not None:
                templates_topk.add(tid)
                template_scores.setdefault(tid, []).append(score)

    def _rate(n: int, d: int) -> Optional[float]:
        return round(n / d, 4) if d > 0 else None

    def _finalize_group(g: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "count": g["count"],
            "primary_backend_accuracy": _rate(g["backend_correct"], g["count"]),
            "final_route_accuracy": _rate(g["final_correct"], g["count"]),
            "template_hit_rate": _rate(g["template_hit"], g["count"]),
        }

    template_score_distribution: Dict[str, Any] = {}
    for tid, scores in template_scores.items():
        arr = np.array(scores, dtype=np.float32)
        template_score_distribution[tid] = {
            "count": int(len(arr)),
            "mean_score": round(float(np.mean(arr)), 4),
            "min_score": round(float(np.min(arr)), 4),
            "max_score": round(float(np.max(arr)), 4),
            "median_score": round(float(np.median(arr)), 4),
        }

    metrics: Dict[str, Any] = {
        "total_cases": total_cases,
        "evaluated_cases": evaluated_cases,
        "skipped_cases": {
            "total": sum(skipped_cases.values()),
            "by_reason": skipped_cases,
        },
        "primary_backend_accuracy": _rate(backend_correct, evaluated_cases),
        "template_route_accuracy": _rate(template_route_correct, evaluated_cases),
        "template_ideal_rate": _rate(template_preferred_hit, template_preferred_total),
        "llm_expected_hit_rate": _rate(llm_expected_hit, llm_expected_total),
        "final_route_accuracy": _rate(final_route_correct, evaluated_cases),
        "template_hit_rate": _rate(
            sum(1 for rr in route_results if rr.get("actual_route_method") in ("template_embedding", "template_embedding_multi_backend")),
            evaluated_cases,
        ),
        "llm_fallback_rate": _rate(llm_fallback_count, evaluated_cases),
        "llm_api_error_rate": _rate(llm_api_error_count, llm_fallback_count),
        "llm_default_fallback_count": llm_default_fallback_count,
        "fallback_policy_violation_rate": _rate(fallback_violation_count, evaluated_cases),
        "invalid_route_rate": _rate(invalid_route_count, evaluated_cases),
        "multi_backend_secondary_accuracy": _rate(multi_backend_secondary_correct, multi_backend_secondary_total),
        "by_benchmark": {k: _finalize_group(v) for k, v in by_benchmark.items()},
        "by_scenario": {k: _finalize_group(v) for k, v in by_scenario.items()},
        "by_expected_backend": {k: _finalize_group(v) for k, v in by_expected_backend.items()},
        "by_route_method": {k: _finalize_group(v) for k, v in by_route_method.items()},
        "template_coverage": {
            "accepted_templates": sorted(templates_accepted),
            "accepted_templates_count": len(templates_accepted),
        },
        "template_topk_coverage": {
            "topk_templates": sorted(templates_topk),
            "topk_templates_count": len(templates_topk),
        },
        "template_score_distribution": template_score_distribution,
    }

    return metrics


# --------------------------------------------------------------------------- #
# Report generation
# --------------------------------------------------------------------------- #


def generate_report(metrics: Dict[str, Any], run_id: str) -> str:
    lines = [
        f"# MemRouter 路由层测试报告",
        f"",
        f"## 1. 运行信息",
        f"- run_id: {run_id}",
        f"- total_cases: {metrics['total_cases']}",
        f"- evaluated_cases: {metrics['evaluated_cases']}",
        f"- skipped_cases: {metrics['skipped_cases']['total']}",
        f"",
        f"## 2. 总体结论",
        f"- final_route_accuracy: {metrics.get('final_route_accuracy', 'N/A')}",
        f"- primary_backend_accuracy: {metrics.get('primary_backend_accuracy', 'N/A')}",
        f"- llm_fallback_rate: {metrics.get('llm_fallback_rate', 'N/A')}",
        f"- invalid_route_rate: {metrics.get('invalid_route_rate', 'N/A')}",
        f"",
        f"## 3. 成本指标",
        f"- template_hit_rate: {metrics.get('template_hit_rate', 'N/A')}",
        f"- template_ideal_rate: {metrics.get('template_ideal_rate', 'N/A')}",
        f"- llm_expected_hit_rate: {metrics.get('llm_expected_hit_rate', 'N/A')}",
        f"- fallback_policy_violation_rate: {metrics.get('fallback_policy_violation_rate', 'N/A')}",
    ]
    return "\n".join(lines)


def generate_report(metrics: Dict[str, Any], run_id: str) -> str:
    """Generate a compact ASCII report for reliable rendering on Windows."""
    lines = [
        "# MemRouter Route-Layer Evaluation Report",
        "",
        "## 1. Run Info",
        f"- run_id: {run_id}",
        f"- total_cases: {metrics['total_cases']}",
        f"- evaluated_cases: {metrics['evaluated_cases']}",
        f"- skipped_cases: {metrics['skipped_cases']['total']}",
        "",
        "## 2. Summary",
        f"- final_route_accuracy: {metrics.get('final_route_accuracy', 'N/A')}",
        f"- primary_backend_accuracy: {metrics.get('primary_backend_accuracy', 'N/A')}",
        f"- llm_fallback_rate: {metrics.get('llm_fallback_rate', 'N/A')}",
        f"- invalid_route_rate: {metrics.get('invalid_route_rate', 'N/A')}",
        "",
        "## 3. Cost / Coverage",
        f"- template_hit_rate: {metrics.get('template_hit_rate', 'N/A')}",
        f"- template_ideal_rate: {metrics.get('template_ideal_rate', 'N/A')}",
        f"- template_route_accuracy: {metrics.get('template_route_accuracy', 'N/A')}",
        f"- llm_expected_hit_rate: {metrics.get('llm_expected_hit_rate', 'N/A')}",
        f"- fallback_policy_violation_rate: {metrics.get('fallback_policy_violation_rate', 'N/A')}",
        "",
        "## 4. Template Coverage",
        f"- accepted_template_coverage: {metrics.get('template_coverage', {})}",
        f"- topk_template_coverage: {metrics.get('template_topk_coverage', {})}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Resolve config
    config = resolve_config(args)
    emb_config = config.get("embedding", {})
    llm_config = config.get("llm", {})

    # Setup run directory
    run_dir = setup_run_dir(args, config)
    run_id = run_dir.name
    logger.info("Run directory: %s", run_dir)

    # Setup file logging
    file_handler = logging.FileHandler(run_dir / "memrouter.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    # Write run_config.yaml (redacted)
    redacted_config = redact_config(config)
    redacted_config["run_id"] = run_id
    redacted_config["eval_args"] = {
        "mode": args.mode,
        "dataset": str(args.dataset),
        "filter_backend": args.filter_backend,
        "filter_scenario": args.filter_scenario,
        "filter_case_id": args.filter_case_id,
    }
    with open(run_dir / "run_config.yaml", "w", encoding="utf-8") as f:
        yaml.dump(redacted_config, f, allow_unicode=True, sort_keys=False)

    # Load dataset
    dataset_path = Path(args.dataset)
    all_cases = load_dataset(dataset_path)
    logger.info("Loaded %d cases from %s", len(all_cases), dataset_path)

    # Validate and filter
    valid_cases: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for case in all_cases:
        is_valid, err_msg = validate_sample(case)
        if not is_valid:
            errors.append({
                "run_id": run_id,
                "error_scope": "per_case",
                "case_id": case.get("case_id", "unknown"),
                "error_type": "schema_validation_error",
                "error_message": err_msg,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })
            continue

        # Apply filters
        if args.filter_backend and case["expected"]["primary_backend_id"] != args.filter_backend:
            continue
        if args.filter_scenario and case.get("scenario") != args.filter_scenario:
            continue
        if args.filter_benchmark and case.get("benchmark") != args.filter_benchmark:
            continue
        if args.filter_case_id and case["case_id"] not in args.filter_case_id:
            continue

        valid_cases.append(case)

    logger.info("Valid cases after filtering: %d", len(valid_cases))

    # Copy dataset snapshot
    shutil.copy2(dataset_path, run_dir / "dataset_snapshot.jsonl")

    # Copy template snapshot
    template_src = Path(__file__).parent.parent / "echomem" / "templates_data"
    template_dst = run_dir / "template_snapshot"
    if template_src.exists():
        template_dst.mkdir(exist_ok=True)
        for yaml_file in template_src.glob("*.yaml"):
            shutil.copy2(yaml_file, template_dst / yaml_file.name)

    # Build pipeline. Validate LLM config first so a missing fallback key does
    # not spend time or quota initializing the embedding provider.
    llm_router_config = create_llm_config(llm_config)
    llm_router = create_llm_backend_router(llm_router_config)
    embedder = create_embedding_provider(emb_config)
    pipeline = MemRouterPipeline.with_defaults(embedder, llm_router_config=llm_router_config)

    # Preflight LLM fallback when using a real provider
    if llm_router_config.provider not in ("mock",):
        preflight_llm_fallback(llm_router, pipeline._registry)

    # Run evaluation
    logger.info("Starting evaluation on %d cases", len(valid_cases))
    eval_results = run_evaluation(valid_cases, pipeline)

    # Extract pipeline exceptions into errors.jsonl
    for er in eval_results:
        if er.get("error_type") == "pipeline_exception":
            errors.append({
                "run_id": run_id,
                "error_scope": "per_case",
                "case_id": er["case_id"],
                "error_type": "pipeline_exception",
                "error_message": er.get("error_message", ""),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            })

    # Build route_results.jsonl
    route_results = build_route_results(run_id, valid_cases, eval_results, pipeline)
    with open(run_dir / "route_results.jsonl", "w", encoding="utf-8") as f:
        for entry in route_results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Build errors.jsonl
    with open(run_dir / "errors.jsonl", "w", encoding="utf-8") as f:
        for err in errors:
            f.write(json.dumps(err, ensure_ascii=False) + "\n")

    # Build llm_fallback_calls.jsonl
    fallback_calls = []
    for rr in route_results:
        if rr.get("actual_route_method") == "llm_backend_fallback":
            raw = rr.get("raw_route_result") or {}
            fallback_info = raw.get("fallback", {})
            latency = rr.get("latency_ms", {})
            fallback_calls.append({
                "run_id": run_id,
                "case_id": rr["case_id"],
                "question": rr["question"],
                "fallback_reason": fallback_info.get("reason", ""),
                "fallback_type": fallback_info.get("type", ""),
                "top_templates_at_fallback": rr.get("top_templates", []),
                "actual_backend": rr.get("actual_primary_backend_id"),
                "latency_ms": latency.get("llm_fallback", 0),
                "token_usage": rr.get("token_usage", {}),
            })
    with open(run_dir / "llm_fallback_calls.jsonl", "w", encoding="utf-8") as f:
        for call in fallback_calls:
            f.write(json.dumps(call, ensure_ascii=False) + "\n")

    # Build metrics_summary.json (total_original = all cases loaded from dataset)
    metrics = compute_metrics(route_results, valid_cases, total_original=len(all_cases))
    with open(run_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Build report.md
    report = generate_report(metrics, run_id)
    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info("Evaluation complete. Results in %s", run_dir)

    # Exit code
    has_global_error = any(e["error_type"] == "pipeline_exception" for e in errors)
    has_invalid_route = (metrics.get("invalid_route_rate") or 0) > 0

    if has_global_error:
        return 1
    if has_invalid_route:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
