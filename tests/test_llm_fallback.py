"""Tests for LLM fallback router."""

import json
import os
from unittest.mock import patch

import pytest

from echomem.llm_fallback import (
    LLMFallbackContext,
    LLMRouterConfig,
    MockLLMBackendRouter,
    _build_default_fallback_result,
    _build_error_result,
    _extract_anthropic_token_usage,
    _extract_openai_token_usage,
    _build_system_prompt,
    _parse_llm_json_to_result,
    _resolve_api_key,
    _strip_markdown_json_fences,
    _validate_llm_route,
)
from echomem.registry import BackendEntry, MemoryBackendRegistry
from echomem.result import DebugInfo, FallbackInfo, MemBackendRouteResult, QueryHints, RouteEntry


def _make_registry() -> MemoryBackendRegistry:
    registry = MemoryBackendRegistry()
    registry.register(
        BackendEntry(
            backend_id="openviking_memory_backend",
            backend_kind="openviking_native",
        )
    )
    registry.register(
        BackendEntry(
            backend_id="graph_memory_backend",
            backend_kind="knowledge_graph",
        )
    )
    return registry


def _make_context(registry: MemoryBackendRegistry | None = None) -> LLMFallbackContext:
    return LLMFallbackContext(
        raw_user_query="test query",
        normalized_user_query="test query",
        registry=registry or _make_registry(),
        failed_template_summary=[],
        query_hints=QueryHints(),
        fallback_reason="test_reason",
    )


def _make_config() -> LLMRouterConfig:
    return LLMRouterConfig(provider="mock", model="mock")


# --------------------------------------------------------------------------- #
# _resolve_api_key
# --------------------------------------------------------------------------- #


class TestResolveApiKey:
    def test_env_var_priority(self) -> None:
        """api_key_env should be checked before api_key."""
        config = LLMRouterConfig(
            provider="openai_compatible",
            model="gpt-4o",
            api_key_env="TEST_API_KEY",
            api_key="explicit_key",
        )
        with patch.dict(os.environ, {"TEST_API_KEY": "from_env"}):
            key = _resolve_api_key(config)
        assert key == "from_env"

    def test_fallback_to_explicit_key(self) -> None:
        """If env var is unset, fall back to api_key."""
        config = LLMRouterConfig(
            provider="openai_compatible",
            model="gpt-4o",
            api_key_env="UNSET_VAR",
            api_key="explicit_key",
        )
        with patch.dict(os.environ, {}, clear=True):
            key = _resolve_api_key(config)
        assert key == "explicit_key"

    def test_raises_when_no_key(self) -> None:
        config = LLMRouterConfig(
            provider="openai_compatible",
            model="gpt-4o",
            api_key="",
        )
        with pytest.raises(ValueError, match="No API key"):
            _resolve_api_key(config)


# --------------------------------------------------------------------------- #
# _strip_markdown_json_fences
# --------------------------------------------------------------------------- #


class TestStripMarkdownJsonFences:
    def test_json_fence(self) -> None:
        raw = '```json\n{"a": 1}\n```'
        assert _strip_markdown_json_fences(raw) == '{"a": 1}'

    def test_plain_fence(self) -> None:
        raw = '```\n{"a": 1}\n```'
        assert _strip_markdown_json_fences(raw) == '{"a": 1}'

    def test_no_fence(self) -> None:
        raw = '{"a": 1}'
        assert _strip_markdown_json_fences(raw) == '{"a": 1}'

    def test_uppercase_json_fence(self) -> None:
        raw = '```JSON\n{"a": 1}\n```'
        assert _strip_markdown_json_fences(raw) == '{"a": 1}'

    def test_fence_with_trailing_spaces(self) -> None:
        raw = '```json\n{"a": 1}\n```   '
        assert _strip_markdown_json_fences(raw) == '{"a": 1}'


# --------------------------------------------------------------------------- #
# _parse_llm_json_to_result
# --------------------------------------------------------------------------- #


class TestParseLlmJsonToResult:
    def test_non_dict_parsed(self) -> None:
        """A JSON list or string should fall back to default instead of crashing."""
        context = _make_context()
        config = _make_config()
        result = _parse_llm_json_to_result([{"routes": []}], context, config)
        assert result.routes[0].backend_id == "openviking_memory_backend"
        assert result.fallback.used is True

    def test_empty_routes(self) -> None:
        context = _make_context()
        config = _make_config()
        result = _parse_llm_json_to_result({"routes": []}, context, config)
        assert result.routes[0].backend_id == "openviking_memory_backend"

    def test_non_dict_route_entries(self) -> None:
        """Routes containing strings/nulls should be skipped, not crash."""
        context = _make_context()
        config = _make_config()
        result = _parse_llm_json_to_result(
            {"routes": ["bad", None, {"backend_id": "openviking_memory_backend", "role": "primary"}]},
            context,
            config,
        )
        assert len(result.routes) == 1
        assert result.routes[0].backend_id == "openviking_memory_backend"
        assert result.routes[0].role == "primary"

    def test_multi_primary_demoted(self) -> None:
        context = _make_context()
        config = _make_config()
        result = _parse_llm_json_to_result(
            {
                "routes": [
                    {"backend_id": "openviking_memory_backend", "role": "primary"},
                    {"backend_id": "graph_memory_backend", "role": "primary"},
                ]
            },
            context,
            config,
        )
        roles = [r.role for r in result.routes]
        assert roles.count("primary") == 1
        assert result.routes[0].role == "primary"
        assert result.routes[0].backend_id == "openviking_memory_backend"

    def test_no_primary_promotes_first(self) -> None:
        context = _make_context()
        config = _make_config()
        result = _parse_llm_json_to_result(
            {
                "routes": [
                    {"backend_id": "graph_memory_backend", "role": "secondary"},
                    {"backend_id": "openviking_memory_backend", "role": "secondary"},
                ]
            },
            context,
            config,
        )
        assert result.routes[0].role == "primary"
        assert result.routes[0].backend_id == "graph_memory_backend"

    def test_skips_illegal_role(self) -> None:
        """Illegal role like 'backup' should be skipped, not promoted to primary."""
        context = _make_context()
        config = _make_config()
        result = _parse_llm_json_to_result(
            {
                "routes": [
                    {"backend_id": "openviking_memory_backend", "role": "backup"},
                ]
            },
            context,
            config,
        )
        # Skipped all illegal routes -> falls back to default
        assert result.routes[0].backend_id == "openviking_memory_backend"
        assert result.routes[0].role == "primary"
        assert "mock_llm_router" in result.fallback.reason or "llm_output_invalid" in result.fallback.reason

    def test_multi_route_sets_post_retrieval_requirements(self) -> None:
        context = _make_context()
        config = _make_config()
        result = _parse_llm_json_to_result(
            {
                "routes": [
                    {"backend_id": "openviking_memory_backend", "role": "primary"},
                    {"backend_id": "graph_memory_backend", "role": "secondary"},
                ]
            },
            context,
            config,
        )
        assert len(result.routes) == 2
        assert result.post_retrieval_requirements.get("check_answerability") is True
        assert result.post_retrieval_requirements.get("deduplicate_evidence") is True

    def test_single_route_sets_check_answerability(self) -> None:
        context = _make_context()
        config = _make_config()
        result = _parse_llm_json_to_result(
            {
                "routes": [
                    {"backend_id": "openviking_memory_backend", "role": "primary"},
                ]
            },
            context,
            config,
        )
        assert len(result.routes) == 1
        assert result.post_retrieval_requirements.get("check_answerability") is True
        assert result.post_retrieval_requirements.get("deduplicate_evidence") is None

    def test_deduplicates_duplicate_backend_id(self) -> None:
        """Duplicate backend_id should be deduplicated, keeping the first occurrence."""
        context = _make_context()
        config = _make_config()
        result = _parse_llm_json_to_result(
            {
                "routes": [
                    {"backend_id": "openviking_memory_backend", "role": "primary"},
                    {"backend_id": "openviking_memory_backend", "role": "secondary"},
                    {"backend_id": "graph_memory_backend", "role": "secondary"},
                ]
            },
            context,
            config,
        )
        backend_ids = [r.backend_id for r in result.routes]
        assert backend_ids == ["openviking_memory_backend", "graph_memory_backend"]
        assert len(backend_ids) == len(set(backend_ids))

# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #


class TestPromptConstruction:
    def test_system_prompt_is_closed_set_and_ascii_safe(self) -> None:
        context = _make_context()
        prompt = _build_system_prompt(context, max_secondary_routes=1)
        assert "backend_id" in prompt
        assert "role" in prompt
        assert '"reasoning"' not in prompt
        prompt.encode("ascii")

    def test_system_prompt_omits_disabled_or_unregistered_backends(self) -> None:
        context = _make_context()
        prompt = _build_system_prompt(context, max_secondary_routes=1)
        assert "temporal_memory_backend" not in prompt

    def test_default_max_tokens_leaves_room_for_json(self) -> None:
        config = LLMRouterConfig(provider="mock", model="mock")
        assert config.max_tokens == 1024


# --------------------------------------------------------------------------- #
# _build_default_fallback_result
# --------------------------------------------------------------------------- #


class TestBuildDefaultFallbackResult:
    def test_falls_back_to_default_backend(self) -> None:
        """v1.4+: default fallback uses openviking_memory_backend regardless of registry."""
        registry = MemoryBackendRegistry()
        context = _make_context(registry)
        config = _make_config()
        result = _build_default_fallback_result(context, config)
        assert result.routes[0].backend_id == "openviking_memory_backend"

    def test_has_top_templates(self) -> None:
        """Default fallback should carry failed template summary for debugging."""
        context = _make_context()
        context.failed_template_summary = [
            {"template_id": "t1", "score": 0.3},
            {"template_id": "t2", "score": 0.2},
        ]
        config = _make_config()
        result = _build_default_fallback_result(context, config)
        assert result.debug.top_templates == context.failed_template_summary[:5]

    def test_has_check_answerability(self) -> None:
        """Default fallback should set check_answerability post-retrieval requirement."""
        context = _make_context()
        config = _make_config()
        result = _build_default_fallback_result(context, config)
        assert result.post_retrieval_requirements.get("check_answerability") is True


# --------------------------------------------------------------------------- #
# _build_error_result
# --------------------------------------------------------------------------- #


class TestBuildErrorResult:
    def test_records_latency(self) -> None:
        context = _make_context()
        config = _make_config()
        result = _build_error_result(context, config, error="timeout", latency_ms=1234)
        meta = result.debug.llm_fallback_meta
        assert meta is not None
        assert meta["latency_ms"] == 1234
        assert meta["error"] == "timeout"


# --------------------------------------------------------------------------- #
# MockLLMBackendRouter
# --------------------------------------------------------------------------- #


class TestMockLLMBackendRouter:
    def test_routes_to_default(self) -> None:
        registry = _make_registry()
        context = _make_context(registry)
        router = MockLLMBackendRouter()
        result = router.route(context)
        assert result.routes[0].backend_id == "openviking_memory_backend"
        assert result.routes[0].role == "primary"
        assert result.fallback.used is True

    def test_reason_is_mock_llm_router(self) -> None:
        """Mock should not duplicate the fallback reason; it should append 'mock_llm_router'."""
        registry = _make_registry()
        context = _make_context(registry)
        router = MockLLMBackendRouter()
        result = router.route(context)
        assert "mock_llm_router" in result.fallback.reason
        # Should not have duplicated the base reason like "reason (reason)"
        assert result.fallback.reason.count("test_reason") <= 1


# --------------------------------------------------------------------------- #
# _validate_llm_route
# --------------------------------------------------------------------------- #


class TestValidateLlmRoute:
    def test_invalid_backend(self) -> None:
        registry = _make_registry()
        result = MemBackendRouteResult(
            raw_user_query="q",
            normalized_user_query="q",
            route_method="llm_backend_fallback",
            routes=[
                RouteEntry(
                    backend_id="nonexistent_backend",
                    backend_kind="unknown",
                    role="primary",
                    confidence=0.6,
                )
            ],
            fallback=FallbackInfo(used=True, type="llm", reason="test"),
            debug=DebugInfo(),
        )
        assert _validate_llm_route(result, registry) is False

    def test_valid_route(self) -> None:
        registry = _make_registry()
        result = MemBackendRouteResult(
            raw_user_query="q",
            normalized_user_query="q",
            route_method="llm_backend_fallback",
            routes=[
                RouteEntry(
                    backend_id="openviking_memory_backend",
                    backend_kind="openviking_native",
                    role="primary",
                    confidence=0.6,
                )
            ],
            fallback=FallbackInfo(used=True, type="llm", reason="test"),
            debug=DebugInfo(),
        )
        assert _validate_llm_route(result, registry) is True

    def test_multiple_primaries_invalid(self) -> None:
        registry = _make_registry()
        result = MemBackendRouteResult(
            raw_user_query="q",
            normalized_user_query="q",
            route_method="llm_backend_fallback",
            routes=[
                RouteEntry(
                    backend_id="openviking_memory_backend",
                    backend_kind="openviking_native",
                    role="primary",
                    confidence=0.6,
                ),
                RouteEntry(
                    backend_id="graph_memory_backend",
                    backend_kind="knowledge_graph",
                    role="primary",
                    confidence=0.6,
                ),
            ],
            fallback=FallbackInfo(used=True, type="llm", reason="test"),
            debug=DebugInfo(),
        )
        assert _validate_llm_route(result, registry) is False

    def test_duplicate_backend_invalid(self) -> None:
        registry = _make_registry()
        result = MemBackendRouteResult(
            raw_user_query="q",
            normalized_user_query="q",
            route_method="llm_backend_fallback",
            routes=[
                RouteEntry(
                    backend_id="openviking_memory_backend",
                    backend_kind="openviking_native",
                    role="primary",
                    confidence=0.6,
                ),
                RouteEntry(
                    backend_id="openviking_memory_backend",
                    backend_kind="openviking_native",
                    role="secondary",
                    confidence=0.6,
                ),
            ],
            fallback=FallbackInfo(used=True, type="llm", reason="test"),
            debug=DebugInfo(),
        )
        assert _validate_llm_route(result, registry) is False


# --------------------------------------------------------------------------- #
# _extract_openai_token_usage
# --------------------------------------------------------------------------- #


class TestExtractOpenaiTokenUsage:
    def test_sdk_object(self) -> None:
        class FakeUsage:
            prompt_tokens = 10
            completion_tokens = 5
            total_tokens = 15

        assert _extract_openai_token_usage(FakeUsage()) == {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    def test_plain_dict(self) -> None:
        assert _extract_openai_token_usage(
            {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
        ) == {
            "prompt_tokens": 20,
            "completion_tokens": 8,
            "total_tokens": 28,
        }

    def test_missing_fields(self) -> None:
        class FakeUsage:
            prompt_tokens = 7

        assert _extract_openai_token_usage(FakeUsage()) == {
            "prompt_tokens": 7,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    def test_none(self) -> None:
        assert _extract_openai_token_usage(None) == {}


# --------------------------------------------------------------------------- #
# _extract_anthropic_token_usage
# --------------------------------------------------------------------------- #


class TestExtractAnthropicTokenUsage:
    def test_sdk_object(self) -> None:
        class FakeUsage:
            input_tokens = 12
            output_tokens = 6

        assert _extract_anthropic_token_usage(FakeUsage()) == {
            "input_tokens": 12,
            "output_tokens": 6,
        }

    def test_plain_dict(self) -> None:
        assert _extract_anthropic_token_usage(
            {"input_tokens": 24, "output_tokens": 9}
        ) == {
            "input_tokens": 24,
            "output_tokens": 9,
        }

    def test_missing_fields(self) -> None:
        assert _extract_anthropic_token_usage({"input_tokens": 3}) == {
            "input_tokens": 3,
            "output_tokens": 0,
        }

    def test_none(self) -> None:
        assert _extract_anthropic_token_usage(None) == {}
