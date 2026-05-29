"""End-to-end integration tests for MemRouterPipeline."""

from echomem.pipeline import MemRouterPipeline

from echomem.embeddings.base import MockEmbeddingProvider


def test_pipeline_default() -> None:
    """A non-memory query should still produce a structurally valid route."""
    embedder = MockEmbeddingProvider(dim=16)
    pipeline = MemRouterPipeline.with_defaults(embedder)

    result = pipeline.route("What is the weather like in Beijing today?")

    # Mock embeddings are deterministic but not semantic, so this test should
    # not assert a specific backend or fallback branch.
    assert len(result.routes) == 1
    assert result.schema_version == "mem-router.backend-route-result.v2"
    assert result.normalized_user_query != ""
    assert result.route_method in {
        "template_embedding",
        "template_embedding_multi_backend",
        "llm_backend_fallback",
    }


def test_pipeline_memory_recall() -> None:
    """A memory-recall query should produce a structurally valid route."""
    embedder = MockEmbeddingProvider(dim=16)
    pipeline = MemRouterPipeline.with_defaults(embedder)

    result = pipeline.route("你记得我之前说过什么吗？")

    # With the mock embedder we cannot guarantee high confidence, but we can
    # assert structural correctness of the result.
    assert len(result.routes) >= 1
    assert result.schema_version == "mem-router.backend-route-result.v2"
    assert result.normalized_user_query != ""
    # route_method should be one of the v1.3/v1.4 variants
    assert result.route_method in {
        "template_embedding",
        "template_embedding_multi_backend",
        "llm_backend_fallback",
    }
