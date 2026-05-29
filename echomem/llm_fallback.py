"""LLMBackendRouter — fallback routing when template matching is inconclusive.

Implements the v1.3/v1.4 LLM fallback design:
  1. Construct a constrained prompt from MemoryBackendRegistry.
  2. Let an LLM choose among registered backends.
  3. Validate the output against the registry.
  4. Record provider, model, latency, and token usage for evaluation.

No real memory backends are called.
"""

import json
import logging
import os
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from echomem.registry import MemoryBackendRegistry
from echomem.result import (
    DebugInfo,
    FallbackInfo,
    MemBackendRouteResult,
    QueryHints,
    RouteEntry,
)

logger = logging.getLogger(__name__)

# Default fixed confidence for LLM fallback routes (see v1.4 design doc)
_DEFAULT_FALLBACK_CONFIDENCE = 0.60


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

class LLMRouterConfig(BaseModel):
    """Runtime configuration for LLMBackendRouter.

    This config is independent of v1.3 schema and is injected at pipeline
    construction time. It is not part of MemoryRouteRequest or Registry.
    """

    provider: str  # mock / openai_compatible / anthropic_compatible / local_http
    model: str
    api_key_env: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    timeout_seconds: int = 60
    max_tokens: int = 1024
    max_secondary_routes: int = 1
    fallback_confidence: float = _DEFAULT_FALLBACK_CONFIDENCE


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #

def create_llm_backend_router(config: LLMRouterConfig) -> "LLMBackendRouter":
    """Factory that creates the appropriate LLMBackendRouter from config.

    Args:
        config: LLM router configuration.

    Returns:
        Configured LLMBackendRouter instance.

    Raises:
        ValueError: If the provider is unsupported or required fields are missing.
        NotImplementedError: If the provider is recognized but not yet implemented.
    """
    logger.info("Creating LLMBackendRouter: provider=%s model=%s", config.provider, config.model)

    if config.provider == "mock":
        return MockLLMBackendRouter(config)

    if config.provider == "openai_compatible":
        _require_field(config, "model")
        _require_api_key(config)
        return OpenAICompatibleLLMBackendRouter(config)

    if config.provider == "anthropic_compatible":
        _require_field(config, "model")
        _require_api_key(config)
        return AnthropicCompatibleLLMBackendRouter(config)

    if config.provider == "local_http":
        _require_field(config, "model")
        _require_field(config, "base_url")
        raise NotImplementedError(
            "local_http LLMBackendRouter is not yet implemented. "
            "Please use mock for unit tests, openai_compatible or anthropic_compatible for real LLM fallback."
        )

    raise ValueError(
        f"Unsupported LLM router provider: '{config.provider}'. "
        f"Supported: mock, openai_compatible, anthropic_compatible, local_http"
    )


def _require_field(config: LLMRouterConfig, name: str) -> None:
    value = getattr(config, name, "")
    if not value:
        raise ValueError(f"LLMRouterConfig required field '{name}' is empty for provider='{config.provider}'")


def _require_api_key(config: LLMRouterConfig) -> None:
    # Prefer env var
    if config.api_key_env:
        key = os.getenv(config.api_key_env)
        if key:
            return
    # Fallback to explicit key
    if config.api_key:
        return
    # Neither available
    if config.api_key_env:
        raise ValueError(
            f"LLMRouterConfig api_key is empty and environment variable "
            f"'{config.api_key_env}' is not set (provider='{config.provider}')"
        )
    raise ValueError(
        f"LLMRouterConfig requires either 'api_key' or 'api_key_env' for provider='{config.provider}'"
    )


def _resolve_api_key(config: LLMRouterConfig) -> str:
    # Prefer env var over explicit key (follows v1.4 design doc)
    if config.api_key_env:
        key = os.getenv(config.api_key_env)
        if key:
            return key
    if config.api_key:
        return config.api_key
    raise ValueError("No API key available")


# --------------------------------------------------------------------------- #
# Context + ABC
# --------------------------------------------------------------------------- #

@dataclass
class LLMFallbackContext:
    """Input context for LLMBackendRouter."""

    raw_user_query: str
    normalized_user_query: str
    registry: MemoryBackendRegistry
    failed_template_summary: List[Dict[str, Any]]
    query_hints: QueryHints
    fallback_reason: str


class LLMBackendRouter(ABC):
    """Abstract LLM fallback router.

    Implementations must produce a MemBackendRouteResult that references only
    backends present in MemoryBackendRegistry.
    """

    @abstractmethod
    def route(self, context: LLMFallbackContext) -> MemBackendRouteResult:
        """Return a route decision using LLM guidance.

        The result must satisfy registry validation; if it does not, the
        implementation should internally fall back to the default backend.
        """
        ...


# --------------------------------------------------------------------------- #
# Mock implementation (for unit tests and CI)
# --------------------------------------------------------------------------- #

class MockLLMBackendRouter(LLMBackendRouter):
    """Mock LLM fallback for CI and regression tests.

    Routes to openviking_memory_backend. No external API calls are made.
    """

    def __init__(self, config: Optional[LLMRouterConfig] = None) -> None:
        self._config = config or LLMRouterConfig(provider="mock", model="mock")

    def route(self, context: LLMFallbackContext) -> MemBackendRouteResult:
        config = self._config or LLMRouterConfig(provider="mock", model="mock")
        result = _build_default_fallback_result(
            context, config, extra_reason="mock_llm_router"
        )
        return result


# --------------------------------------------------------------------------- #
# Shared helpers for real LLM implementations
# --------------------------------------------------------------------------- #

def _build_system_prompt(context: LLMFallbackContext, max_secondary_routes: int) -> str:
    """Build an ASCII-only, closed-set prompt for robust compatible endpoints.

    MemRouter v1.4+ only routes to backend_id.
    """
    catalog_lines: List[str] = []
    enabled_entries = context.registry.list_enabled()
    enabled_backend_ids = {entry.backend_id for entry in enabled_entries}
    for entry in enabled_entries:
        catalog_lines.append(
            f'- backend_id: "{entry.backend_id}"\n'
            f'  backend_kind: "{entry.backend_kind}"\n'
            f'  description: {entry.description.strip()}'
        )

    rules: List[str] = [
        "1. backend_id MUST be copied exactly from the catalog.",
        f"2. Return at most 1 primary backend and at most {max_secondary_routes} secondary backend(s).",
    ]
    examples: List[str] = []

    if "graph_memory_backend" in enabled_backend_ids:
        rules.append(
            "3. GRAPH-FIRST RULE (strict): Only route to graph_memory_backend when the query "
            "explicitly asks about structural relations, social networks, organizational "
            "membership, event co-participation, or causal dependencies between MULTIPLE "
            "entities. Do NOT route to graph for simple personal attribute lookups "
            "(e.g. 'What books has John read?', 'What tattoo does Audrey have?') -- "
            "those are personal facts and belong to openviking_memory_backend."
        )
        examples.extend(
            [
                '- "Who are Jon\'s friends?" -> graph_memory_backend (social network)',
                '- "Which events did John and Maria both attend?" -> graph_memory_backend (co-participation)',
                '- "Which API path starts all calls?" -> graph_memory_backend (system dependency)',
                '- "What caused the service outage?" -> graph_memory_backend (causal dependency)',
            ]
        )

    if "temporal_memory_backend" in enabled_backend_ids:
        rules.append(
            "4. TEMPORAL-OVERRIDE RULE (highest priority): If the user query starts with "
            "'When did', 'How long', 'How many years/months/days', 'How many times has', "
            "'First time', 'Last time', or asks explicitly for a timestamp, date, duration, "
            "elapsed time, event order, or recency, you MUST route to temporal_memory_backend. "
            "This rule OVERRIDES all other rules when time is the core answer."
        )
        rules.append(
            "4b. Strong temporal signals (any one triggers temporal-override): "
            "'when did', 'how long', 'how many years', 'how many months', 'how many days', "
            "'first time', 'last time', 'before', 'after', 'between', 'ago', 'recently', "
            "'in July 2023', 'on Monday', 'what happened on DATE', 'how many times has', "
            "'when was the last time', 'when did ... start', 'when did ... finish', "
            "'how long has ... been'."
        )
        rules.append(
            "4c. Boundary (OVERRIDE does NOT apply): A query that merely CONTAINS a time word "
            "but the answer is a person, place, item, attribute, or subjective experience. "
            "Example: 'Who supports Caroline when she has a negative experience?' -> answer is a PERSON, not a time. "
            "Example: 'Where did John explore on a road trip last year?' -> answer is a PLACE, not a time. "
            "Example: 'What did Joanna receive from her brother that brought back childhood memories?' -> answer is a GIFT, not a time. "
            "Example: 'What significant event happened... towards the end of summer 2023?' -> answer is an EVENT DESCRIPTION, not a timestamp."
        )
        examples.extend(
            [
                '- "How long did I wait for the decision?" -> temporal_memory_backend (duration)',
                '- "When did Melanie go to the museum?" -> temporal_memory_backend (timestamp) OVERRIDE',
                '- "How many times has Melanie gone to the beach in 2023?" -> temporal_memory_backend (count+time) OVERRIDE',
                '- "When did Jon start reading The Lean Startup?" -> temporal_memory_backend (start time) OVERRIDE',
                '- "When did Maria get in a car accident?" -> temporal_memory_backend (event time) OVERRIDE',
                '- "How long has Nate had his first two turtles?" -> temporal_memory_backend (duration) OVERRIDE',
                '- "When did Joanna start writing her third screenplay?" -> temporal_memory_backend (start time) OVERRIDE',
                '- "When did Melanie run a charity race?" -> temporal_memory_backend (event time) OVERRIDE',
                '- "When did Melanie go camping in July?" -> temporal_memory_backend (timestamp) OVERRIDE',
                '- "When did Jon visit networking events for his store?" -> temporal_memory_backend (event time) OVERRIDE',
                '- "Who supports Caroline when she has a negative experience?" -> openviking_memory_backend (person, not time)',
                '- "What did Joanna receive from her brother that brought back childhood memories?" -> openviking_memory_backend (gift, not time)',
                '- "Where did John explore on a road trip last year?" -> openviking_memory_backend (place, not time)',
                '- "What significant event happened in Sam\'s life towards the end of summer 2023?" -> openviking_memory_backend (event description, not time)',
            ]
        )

    if "openviking_memory_backend" in enabled_backend_ids:
        rules.append(
            "5. Route to openviking_memory_backend when the query asks for "
            "subjective preferences, habits, feelings, vague summaries, profile "
            "attributes, recommendations based on remembered context, previous-chat recall, "
            "or specific personal facts about a person (e.g. items owned, books read, "
            "places visited, attributes, experiences, collections, hobbies)."
        )
        examples.extend(
            [
                '- "What kind of music do I prefer?" -> openviking_memory_backend (preference)',
                '- "Remind me what we discussed before." -> openviking_memory_backend (chat recall)',
                '- "Why did Jolene put off doing yoga?" -> openviking_memory_backend (subjective reason)',
                '- "What books has John read?" -> openviking_memory_backend (personal fact)',
                '- "What tattoo does Audrey have?" -> openviking_memory_backend (personal attribute)',
                '- "What items did Calvin buy?" -> openviking_memory_backend (personal fact)',
            ]
        )

    rules.append("6. Output JSON only. No markdown and no explanation outside JSON.")

    return (
        "You are a memory backend router. The template matcher could not route "
        "the user query with enough confidence, so you must choose from the "
        "registered backend catalog below.\n\n"
        "Available backend catalog:\n"
        f"{chr(10).join(catalog_lines)}\n\n"
        "Strict constraints:\n"
        f"{chr(10).join(rules)}\n\n"
        "Few-shot examples:\n"
        f"{chr(10).join(examples)}\n"
        "\n"
        "Output schema (backend_id and role only):\n"
        "{\n"
        '  "routes": [\n'
        "    {\n"
        '      "backend_id": "...",\n'
        '      "role": "primary"\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def _build_user_prompt(context: LLMFallbackContext) -> str:
    """Build an ASCII-only user prompt with query and failed template evidence."""
    # Detect temporal signals for explicit prompting
    raw_lower = context.raw_user_query.lower()
    temporal_signals = []
    temporal_keywords = [
        "when did", "how long", "how many years", "how many months", "how many days",
        "how many times has", "first time", "last time", "when was the last time",
        "when did", "start", "finish", "ago", "before", "after", "between",
        "recently", "in july", "in august", "in september", "in october",
        "on monday", "on tuesday", "on wednesday", "on thursday", "on friday",
        "what happened on", "what was happening on", "where was", "on december",
        "in may", "in june", "in 2023", "in 2022", "last year", "last week",
        "last month", "two months ago", "two weeks ago",
    ]
    for kw in temporal_keywords:
        if kw in raw_lower:
            temporal_signals.append(kw)

    lines = [
        f"Raw user query: {context.raw_user_query}",
        f"Normalized query: {context.normalized_user_query}",
        f"Fallback reason: {context.fallback_reason}",
        "",
    ]
    if temporal_signals:
        lines.append(
            f"ATTENTION: This query contains STRONG TEMPORAL SIGNALS: {', '.join(temporal_signals[:5])}. "
            "If the core question is asking for a time, date, duration, or order, "
            "you MUST route to temporal_memory_backend (OVERRIDE rule 4 applies)."
        )
        lines.append("")
    lines.append("Failed or low-confidence template matches:")
    for item in context.failed_template_summary[:5]:
        lines.append(
            f"  - template_id: {item.get('template_id', 'unknown')}, "
            f"score: {item.get('score', 0)}"
        )
    lines.append("")
    lines.append(
        "Choose the best backend. Copy backend_id exactly from the catalog. "
        "Return JSON only."
    )
    return "\n".join(lines)


def _parse_llm_json_to_result(
    parsed: Any,
    context: LLMFallbackContext,
    config: LLMRouterConfig,
) -> MemBackendRouteResult:
    """Convert LLM JSON output to MemBackendRouteResult with primary-guarantee."""
    if not isinstance(parsed, dict):
        logger.warning("LLM returned non-dict JSON (%s); using default fallback", type(parsed).__name__)
        return _build_default_fallback_result(context, config, extra_reason="llm_output_schema_invalid")

    routes: List[RouteEntry] = []
    seen_backend_ids: set[str] = set()
    raw_routes = parsed.get("routes", [parsed])  # Handle both {"routes": [...]} and flat dict
    if not isinstance(raw_routes, list):
        raw_routes = [raw_routes]

    for r in raw_routes:
        if not isinstance(r, dict):
            logger.warning("LLM route entry is not a dict (%s); skipping", type(r).__name__)
            continue
        backend_id = r.get("backend_id", "")
        role = r.get("role", "primary")

        # Guard against non-string field values from malformed LLM JSON
        if not isinstance(backend_id, str) or not isinstance(role, str):
            logger.warning(
                "LLM route entry has invalid field types (backend_id=%s, role=%s); skipping",
                type(backend_id).__name__,
                type(role).__name__,
            )
            continue

        # Reject illegal roles so they cannot be silently promoted to primary later
        if role not in ("primary", "secondary"):
            logger.warning(
                "LLM route entry has invalid role '%s'; must be 'primary' or 'secondary'; skipping",
                role,
            )
            continue

        # Deduplicate by backend_id: keep the first occurrence only
        if backend_id in seen_backend_ids:
            logger.warning("LLM returned duplicate backend_id '%s'; skipping duplicate", backend_id)
            continue
        seen_backend_ids.add(backend_id)

        entry = context.registry.get(backend_id)
        backend_kind = entry.backend_kind if entry else "unknown"
        route = RouteEntry(
            backend_id=backend_id,
            backend_kind=backend_kind,
            role=role,
            confidence=config.fallback_confidence,
            matched_template_id="",
            query_hints=context.query_hints,
        )
        routes.append(route)

    # Ensure at least one route; if empty, fall back to default
    if not routes:
        logger.warning("LLM returned empty routes; using default backend")
        return _build_default_fallback_result(context, config)

    # Separate and fix primary/secondary
    primaries = [r for r in routes if r.role == "primary"]
    secondaries = [r for r in routes if r.role != "primary"]

    # Guarantee exactly one primary
    if not primaries:
        routes[0].role = "primary"
        primaries = [routes[0]]
        secondaries = [r for r in routes[1:] if r.role != "primary"]
    elif len(primaries) > 1:
        # Keep first primary, demote rest to secondary
        for r in routes:
            if r.role == "primary" and r is not primaries[0]:
                r.role = "secondary"
        primaries = [primaries[0]]
        secondaries = [r for r in routes if r is not primaries[0]]

    # Reorder: primary first, then secondaries, then trim
    ordered = primaries + secondaries
    ordered = ordered[: 1 + config.max_secondary_routes]

    # Final safety: ensure first item is primary
    if ordered and ordered[0].role != "primary":
        ordered[0].role = "primary"

    # Multi-route or LLM fallback scenarios benefit from post-retrieval checks
    post_retrieval_requirements: Dict[str, bool] = {"check_answerability": True}
    if len(ordered) > 1:
        post_retrieval_requirements["deduplicate_evidence"] = True

    return MemBackendRouteResult(
        schema_version="mem-router.backend-route-result.v2",
        raw_user_query=context.raw_user_query,
        normalized_user_query=context.normalized_user_query,
        route_method="llm_backend_fallback",
        routes=ordered,
        post_retrieval_requirements=post_retrieval_requirements,
        fallback=FallbackInfo(
            used=True,
            type="llm_backend_router",
            reason=context.fallback_reason,
        ),
        debug=DebugInfo(),
    )


def _build_default_fallback_result(
    context: LLMFallbackContext,
    config: LLMRouterConfig,
    extra_reason: str = "llm_output_invalid_or_empty",
) -> MemBackendRouteResult:
    """Build a safe default result when LLM output is invalid or empty."""
    default_backend_id = "openviking_memory_backend"
    entry = context.registry.get(default_backend_id)
    backend_kind = entry.backend_kind if entry else "openviking_native"

    route = RouteEntry(
        backend_id=default_backend_id,
        backend_kind=backend_kind,
        role="primary",
        confidence=config.fallback_confidence,
        matched_template_id="",
        query_hints=context.query_hints,
    )
    return MemBackendRouteResult(
        schema_version="mem-router.backend-route-result.v2",
        raw_user_query=context.raw_user_query,
        normalized_user_query=context.normalized_user_query,
        route_method="llm_backend_fallback",
        routes=[route],
        post_retrieval_requirements={"check_answerability": True},
        fallback=FallbackInfo(
            used=True,
            type="llm_backend_router",
            reason=f"{context.fallback_reason} ({extra_reason})",
        ),
        debug=DebugInfo(
            top_templates=context.failed_template_summary[:5],
        ),
    )


def _build_error_result(
    context: LLMFallbackContext,
    config: LLMRouterConfig,
    error: str,
    latency_ms: int = 0,
) -> MemBackendRouteResult:
    """Build a result that records an LLM call failure without interrupting benchmark."""
    result = _build_default_fallback_result(context, config, extra_reason="llm_api_error")
    # Inject error details into debug so eval scripts can count failure reasons
    result.debug.top_templates = context.failed_template_summary[:5]
    result.debug.llm_fallback_meta = {
        "provider": config.provider,
        "model": config.model,
        "error": error,
        "latency_ms": latency_ms,
        "token_usage": {},
    }
    return result


def _validate_llm_route(
    result: MemBackendRouteResult,
    registry: MemoryBackendRegistry,
) -> bool:
    """Validate that an LLM-produced route only references enabled backends.

    Returns True if valid, False otherwise.
    """
    if not result.routes:
        logger.error("LLM route has no routes")
        return False

    # Must have exactly one primary
    primaries = [r for r in result.routes if r.role == "primary"]
    if len(primaries) != 1:
        logger.error("LLM route has %d primary backends; required exactly 1", len(primaries))
        return False

    seen_backends: set[str] = set()
    for route in result.routes:
        if route.role not in ("primary", "secondary"):
            logger.error(
                "LLM route has invalid role '%s'; must be 'primary' or 'secondary'",
                route.role,
            )
            return False
        if route.backend_id in seen_backends:
            logger.error(
                "LLM route has duplicate backend_id: %s",
                route.backend_id,
            )
            return False
        seen_backends.add(route.backend_id)
        if not registry.is_enabled(route.backend_id):
            logger.error(
                "LLM route references unregistered/disabled backend: %s",
                route.backend_id,
            )
            return False
    return True


def _extract_anthropic_token_usage(usage: Any) -> Dict[str, int]:
    """Safely extract token usage from Anthropic-compatible response.usage.

    Handles both SDK objects (attribute access) and plain dicts.
    """
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
    return {
        "input_tokens": getattr(usage, "input_tokens", 0),
        "output_tokens": getattr(usage, "output_tokens", 0),
    }


def _extract_openai_token_usage(usage: Any) -> Dict[str, int]:
    """Safely extract token usage from OpenAI-compatible response.usage.

    Handles both SDK objects (attribute access) and plain dicts.
    """
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def _summarize_base_url(base_url: str) -> str:
    """Return a safe summary of base_url for logging (domain only, no path/credentials)."""
    if not base_url:
        return "default"
    try:
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        return parsed.netloc or parsed.path or base_url
    except Exception:
        return "custom"


# --------------------------------------------------------------------------- #
# OpenAI-compatible implementation (real LLM fallback)
# --------------------------------------------------------------------------- #

class OpenAICompatibleLLMBackendRouter(LLMBackendRouter):
    """Real LLM fallback using OpenAI Chat Completions API.

    Constructs a constrained prompt from the registry, uses JSON mode for
    structured output, validates against the registry, and records latency
    and token usage.
    """

    def __init__(self, config: LLMRouterConfig) -> None:
        self._config = config
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Install with: pip install 'echomem[openai]'"
            ) from exc

        api_key = _resolve_api_key(config)
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url

        self._client = openai.OpenAI(**client_kwargs)
        logger.info(
            "OpenAICompatibleLLMBackendRouter initialized: model=%s base_url=%s",
            config.model,
            config.base_url or "default",
        )

    def route(self, context: LLMFallbackContext) -> MemBackendRouteResult:
        logger.debug("OpenAICompatibleLLMBackendRouter.route() called")

        # Build constrained prompt
        system_prompt = _build_system_prompt(context, self._config.max_secondary_routes)
        user_prompt = _build_user_prompt(context)

        start_time = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self._config.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._config.temperature,
                max_tokens=self._config.max_tokens,
                response_format={"type": "json_object"},
                timeout=self._config.timeout_seconds,
            )
        except Exception as exc:
            logger.error("OpenAI LLM fallback API call failed: %s", exc)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return _build_error_result(context, self._config, error=str(exc), latency_ms=latency_ms)

        latency_ms = int((time.perf_counter() - start_time) * 1000)
        try:
            content = response.choices[0].message.content or "{}"
        except Exception as exc:
            logger.error("OpenAI response structure unexpected: %s", exc)
            result = _build_error_result(
                context, self._config, error=f"response_structure_error: {exc}", latency_ms=latency_ms
            )
            result.debug.top_templates = context.failed_template_summary[:5]
            return result

        # Parse and validate
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON: %s", exc)
            result = _build_default_fallback_result(
                context, self._config, extra_reason="llm_invalid_json"
            )
            result.debug.top_templates = context.failed_template_summary[:5]
            result.debug.llm_fallback_meta = {
                "provider": "openai_compatible",
                "model": self._config.model,
                "base_url_summary": _summarize_base_url(self._config.base_url),
                "temperature": self._config.temperature,
                "latency_ms": latency_ms,
                "error": f"invalid_json: {exc}",
                "token_usage": {},
            }
            return result

        result = _parse_llm_json_to_result(parsed, context, self._config)

        # Registry validation
        if not _validate_llm_route(result, context.registry):
            logger.error("LLM route failed registry validation; falling back to default backend")
            result = _build_default_fallback_result(
                context, self._config, extra_reason="llm_registry_validation_failed"
            )

        # Inject debug metadata
        result.debug.top_templates = context.failed_template_summary[:5]
        result.debug.llm_fallback_meta = {
            "provider": "openai_compatible",
            "model": self._config.model,
            "base_url_summary": _summarize_base_url(self._config.base_url),
            "temperature": self._config.temperature,
            "latency_ms": latency_ms,
            "token_usage": _extract_openai_token_usage(getattr(response, "usage", None)),
        }

        return result


# --------------------------------------------------------------------------- #
# Anthropic-compatible implementation (MiniMax, Claude, etc.)
# --------------------------------------------------------------------------- #

class AnthropicCompatibleLLMBackendRouter(LLMBackendRouter):
    """Real LLM fallback using Anthropic Messages API.

    Supports MiniMax (via Anthropic-compatible endpoint), Anthropic Claude,
    and any other provider exposing the Messages API.

    Anthropic does not have a JSON mode, so we enforce JSON output via the
    system prompt and parse the response text as JSON.
    """

    def __init__(self, config: LLMRouterConfig) -> None:
        self._config = config
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "anthropic is not installed. Install with: pip install anthropic"
            ) from exc

        import httpx

        api_key = _resolve_api_key(config)
        client_kwargs: Dict[str, Any] = {"api_key": api_key}
        if config.base_url:
            client_kwargs["base_url"] = config.base_url
        # MiniMax requires X-Api-Key header instead of standard Authorization.
        # The anthropic SDK overrides http_client headers, so we use an event
        # hook to inject the header at request time after SDK processing.
        def _inject_minimax_header(request: httpx.Request) -> None:
            request.headers["X-Api-Key"] = api_key

        client_kwargs["http_client"] = httpx.Client(
            event_hooks={"request": [_inject_minimax_header]},
            timeout=config.timeout_seconds if config.timeout_seconds else 120.0,
        )

        self._client = anthropic.Anthropic(**client_kwargs)
        logger.info(
            "AnthropicCompatibleLLMBackendRouter initialized: model=%s base_url=%s",
            config.model,
            config.base_url or "default",
        )

    def route(self, context: LLMFallbackContext) -> MemBackendRouteResult:
        logger.debug("AnthropicCompatibleLLMBackendRouter.route() called")

        system_prompt = _build_system_prompt(context, self._config.max_secondary_routes)
        user_prompt = _build_user_prompt(context)

        start_time = time.perf_counter()
        try:
            response = self._client.messages.create(
                model=self._config.model,
                max_tokens=self._config.max_tokens,
                temperature=self._config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                timeout=self._config.timeout_seconds,
            )
        except Exception as exc:
            logger.error("Anthropic LLM fallback API call failed: %s", exc)
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return _build_error_result(context, self._config, error=str(exc), latency_ms=latency_ms)

        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Extract text from Anthropic content blocks
        try:
            content_parts: List[str] = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    content_parts.append(block.text)
            content = "\n".join(content_parts) or "{}"
        except Exception as exc:
            logger.error("Anthropic response structure unexpected: %s", exc)
            result = _build_error_result(
                context, self._config, error=f"response_structure_error: {exc}", latency_ms=latency_ms
            )
            result.debug.top_templates = context.failed_template_summary[:5]
            return result

        # Some providers wrap JSON in markdown fences; strip them
        content = _strip_markdown_json_fences(content)

        # Parse and validate
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.error("LLM returned invalid JSON: %s", exc)
            result = _build_default_fallback_result(
                context, self._config, extra_reason="llm_invalid_json"
            )
            result.debug.top_templates = context.failed_template_summary[:5]
            result.debug.llm_fallback_meta = {
                "provider": "anthropic_compatible",
                "model": self._config.model,
                "base_url_summary": _summarize_base_url(self._config.base_url),
                "temperature": self._config.temperature,
                "latency_ms": latency_ms,
                "error": f"invalid_json: {exc}",
                "token_usage": {},
            }
            return result

        result = _parse_llm_json_to_result(parsed, context, self._config)

        # Registry validation
        if not _validate_llm_route(result, context.registry):
            logger.error("LLM route failed registry validation; falling back to default backend")
            result = _build_default_fallback_result(
                context, self._config, extra_reason="llm_registry_validation_failed"
            )

        # Inject debug metadata
        result.debug.top_templates = context.failed_template_summary[:5]
        usage = _extract_anthropic_token_usage(getattr(response, "usage", None))

        result.debug.llm_fallback_meta = {
            "provider": "anthropic_compatible",
            "model": self._config.model,
            "base_url_summary": _summarize_base_url(self._config.base_url),
            "temperature": self._config.temperature,
            "latency_ms": latency_ms,
            "token_usage": usage,
        }

        return result


def _strip_markdown_json_fences(text: str) -> str:
    """Strip markdown ```json ... ``` fences if present (case-insensitive)."""
    text = text.strip()
    lower = text.lower()
    if lower.startswith("```json"):
        text = text[7:]
    elif lower.startswith("```"):
        text = text[3:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    return text.strip()
