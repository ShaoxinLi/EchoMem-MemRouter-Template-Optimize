"""Tests for OpenVikingAdapter."""

from echomem.adapters.openviking import OpenVikingAdapter
from echomem.result import (
    MemBackendRouteResult,
    QueryHints,
    RouteEntry,
)


def test_translate_basic() -> None:
    adapter = OpenVikingAdapter()
    result = MemBackendRouteResult(
        raw_user_query="你记得我之前说过什么吗？",
        normalized_user_query="记得 我 之前 说过 什么",
        route_method="template_embedding",
        routes=[
            RouteEntry(
                backend_id="openviking_memory_backend",
                backend_kind="openviking_native",
                role="primary",
                confidence=0.82,
                matched_template_id="openviking.personal_fact_lookup.en.v2",
                query_hints=QueryHints(
                    temporal_hints=["上次"],
                    semantic_hints=["history"],
                ),
            )
        ],
    )

    params = adapter.translate(result)
    assert params["query"] == "你记得我之前说过什么吗？"
    assert params["route_method"] == "template_embedding"
    assert "query_hints" in params
    assert params["query_hints"]["temporal_hints"] == ["上次"]
    # MemRouter should not implicitly narrow OpenViking internal directory paths
    assert "target_uri" not in params


def test_translate_empty_routes() -> None:
    adapter = OpenVikingAdapter()
    result = MemBackendRouteResult(
        raw_user_query="hello",
        normalized_user_query="hello",
        route_method="llm_backend_fallback",
        routes=[],
    )
    params = adapter.translate(result)
    assert params == {}


def test_translate_rejects_non_openviking_route() -> None:
    """Adapter must reject routes not targeting OpenViking to prevent silent misuse."""
    adapter = OpenVikingAdapter()
    result = MemBackendRouteResult(
        raw_user_query="Alice 和 Bob 什么关系",
        normalized_user_query="Alice 和 Bob 什么关系",
        route_method="template_embedding",
        routes=[
            RouteEntry(
                backend_id="graph_memory_backend",
                backend_kind="knowledge_graph",
                role="primary",
                confidence=0.85,
                query_hints=QueryHints(),
            )
        ],
    )
    try:
        adapter.translate(result)
        assert False, "Expected ValueError for non-OpenViking route"
    except ValueError as exc:
        assert "graph_memory_backend" in str(exc)
        assert "openviking_memory_backend" in str(exc)


def test_default_spec_for_llm_fallback_to_openviking() -> None:
    """LLM fallback to OpenViking should still produce a conservative instruction."""
    adapter = OpenVikingAdapter()

    spec = adapter.get_default_spec("")

    assert spec is not None
    assert spec.search_mode == "search"
    assert spec.openviking is not None
    assert spec.openviking.context_type == "memory"
    assert spec.openviking.target_uri == "viking://memories"
    assert spec.openviking.skip_intent_analysis is False
    assert spec.openviking.typed_query_template is None
