"""Tests for TemplateMatcher."""

import numpy as np

from echomem.features import QueryFeatureBuilder
from echomem.matcher import TemplateMatcher
from echomem.normalizer import QueryNormalizer
from echomem.templates import (
    BackendRouteTemplateIndex,
    Calibration,
    HardNegative,
    IntentFamily,
    MemoryBackendRouteTemplate,
    TemplateTarget,
    Thresholds,
)

from echomem.embeddings.base import MockEmbeddingProvider


def test_match_openviking_template(mock_embedder: MockEmbeddingProvider) -> None:
    # Load builtin templates
    index = BackendRouteTemplateIndex()
    index.load_from_directory(
        __import__("pathlib").Path(__file__).parent.parent / "echomem" / "templates_data"
    )

    matcher = TemplateMatcher(embedder=mock_embedder, template_index=index)

    # Build features for a clear memory-recall query
    fb = QueryFeatureBuilder(embedder=mock_embedder, normalizer=QueryNormalizer())
    features = fb.build("你记得我之前说过什么吗？")

    template_cands, backend_cands = matcher.match(features)

    # Mock embedder cannot simulate real semantic similarity, so we only
    # assert structural correctness, not the specific ranking.
    assert len(template_cands) >= 1
    assert all(-1.0 <= c.score <= 1.0 for c in template_cands)
    assert len(backend_cands) >= 1
    assert all(-1.0 <= c.score <= 1.0 for c in backend_cands)


def test_match_with_hard_negatives(mock_embedder: MockEmbeddingProvider) -> None:
    index = BackendRouteTemplateIndex()
    index.load_from_directory(
        __import__("pathlib").Path(__file__).parent.parent / "echomem" / "templates_data"
    )
    matcher = TemplateMatcher(embedder=mock_embedder, template_index=index)

    fb = QueryFeatureBuilder(embedder=mock_embedder, normalizer=QueryNormalizer())
    # A query that is a hard negative for the OpenViking template
    features = fb.build("今天北京天气怎么样？")

    template_cands, backend_cands = matcher.match(features)

    # Should still produce candidates
    assert len(template_cands) >= 1
    # Score components should include penalty info (hard negatives exist in the template)
    assert "penalty" in template_cands[0].score_components
    # Scores are cosine similarities, must be in [-1, 1]
    assert -1.0 <= template_cands[0].score <= 1.0


def test_match_single_prototype(mock_embedder: MockEmbeddingProvider) -> None:
    """Templates with fewer than 3 prototypes must not crash on S_mean@3."""
    single_proto_template = MemoryBackendRouteTemplate(
        template_id="single.proto.v1",
        target=TemplateTarget(
            primary_backend_id="openviking_memory_backend",
        ),
        intent_family=IntentFamily(name="single_proto_test"),
        query_prototypes=["only one prototype"],
        hard_negatives=[
            HardNegative(
                query="not this route",
                confusing_with_backend="graph_memory_backend",
                reason="single hard negative evidence",
            )
        ],
        thresholds=Thresholds(
            accept=0.72,
            fallback=0.58,
            margin=0.04,
            hard_negative_margin=0.03,
            hard_negative_penalty=0.06,
        ),
        calibration=Calibration(),
    )

    index = BackendRouteTemplateIndex()
    index.add(single_proto_template)

    matcher = TemplateMatcher(embedder=mock_embedder, template_index=index)
    fb = QueryFeatureBuilder(embedder=mock_embedder, normalizer=QueryNormalizer())
    features = fb.build("test query")

    # Must not raise ValueError: kth out of bounds
    template_cands, backend_cands = matcher.match(features)
    assert len(template_cands) == 1
    assert template_cands[0].template_id == "single.proto.v1"
    components = template_cands[0].score_components
    assert components["top_matched_prototype"]["text"] == "only one prototype"
    assert components["top_hard_negative"]["query"] == "not this route"
    assert components["top_hard_negative"]["confusing_with_backend"] == "graph_memory_backend"
    assert len(backend_cands) == 1


def test_template_texts_are_normalized_before_embedding(mock_embedder: MockEmbeddingProvider) -> None:
    """Prototype and hard-negative texts should use the same normalization path as queries."""
    index = BackendRouteTemplateIndex()
    index.load_from_directory(
        __import__("pathlib").Path(__file__).parent.parent / "echomem" / "templates_data"
    )
    template = index.get("openviking.personal_fact_lookup.en.v2")
    assert template is not None

    matcher = TemplateMatcher(embedder=mock_embedder, template_index=index)
    proto_embeddings, _ = matcher._get_prototype_embeddings(template)

    expected_texts = [QueryNormalizer().normalize(text) for text in template.query_prototypes]
    expected_embeddings = mock_embedder.embed(expected_texts)
    norms = np.linalg.norm(expected_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    expected_embeddings = expected_embeddings / norms

    assert np.allclose(proto_embeddings, expected_embeddings)
