"""Tests for RouteDecision."""

from echomem.decision import RouteDecision
from echomem.matcher import BackendCandidate, TemplateCandidate
from echomem.registry import BackendEntry, MemoryBackendRegistry
from echomem.result import QueryHints
from echomem.templates import BackendRouteTemplateIndex


def make_decision() -> RouteDecision:
    reg = MemoryBackendRegistry()
    reg.register(
        BackendEntry(
            backend_id="openviking_memory_backend",
            backend_kind="openviking_native",
        )
    )
    index = BackendRouteTemplateIndex()
    index.load_from_directory(
        __import__("pathlib").Path(__file__).parent.parent / "echomem" / "templates_data"
    )
    return RouteDecision(registry=reg, template_index=index)


def test_high_confidence_match() -> None:
    dec = make_decision()
    tc = [
        TemplateCandidate(
            template_id="openviking.personal_fact_lookup.en.v2",
            primary_backend_id="openviking_memory_backend",
            score=0.82,
            score_components={},
        )
    ]
    bc = [
        BackendCandidate(
            backend_id="openviking_memory_backend",
            best_template_id="openviking.personal_fact_lookup.en.v2",
            score=0.82,
        )
    ]
    result = dec.decide("q", "q", tc, bc, QueryHints())

    assert result.route_method == "template_embedding"
    assert len(result.routes) == 1
    assert result.routes[0].confidence == 0.82
    assert not result.fallback.used


def test_weak_match() -> None:
    dec = make_decision()
    tc = [
        TemplateCandidate(
            template_id="openviking.personal_fact_lookup.en.v2",
            primary_backend_id="openviking_memory_backend",
            score=0.50,
            score_components={},
        )
    ]
    bc = [
        BackendCandidate(
            backend_id="openviking_memory_backend",
            best_template_id="openviking.personal_fact_lookup.en.v2",
            score=0.50,
        )
    ]
    result = dec.decide("q", "q", tc, bc, QueryHints())

    # 0.50 is between fallback (0.46) and accept (0.56) => LLM fallback.
    assert result.route_method == "llm_backend_fallback"
    assert result.fallback.used


def test_default_backend() -> None:
    dec = make_decision()
    tc = [
        TemplateCandidate(
            template_id="openviking.personal_fact_lookup.en.v2",
            primary_backend_id="openviking_memory_backend",
            score=0.40,
            score_components={},
        )
    ]
    bc = [
        BackendCandidate(
            backend_id="openviking_memory_backend",
            best_template_id="openviking.personal_fact_lookup.en.v2",
            score=0.40,
        )
    ]
    result = dec.decide("q", "q", tc, bc, QueryHints())

    assert result.route_method == "llm_backend_fallback"
    assert result.fallback.used


def test_multi_backend_acceptance() -> None:
    """Both top backends high-confidence with small margin -> multi-backend route."""
    reg = MemoryBackendRegistry()
    reg.register(
        BackendEntry(
            backend_id="openviking_memory_backend",
            backend_kind="openviking_native",
        )
    )
    reg.register(
        BackendEntry(
            backend_id="graph_memory_backend",
            backend_kind="knowledge_graph",
        )
    )
    index = BackendRouteTemplateIndex()
    index.load_from_directory(
        __import__("pathlib").Path(__file__).parent.parent / "echomem" / "templates_data"
    )
    dec = RouteDecision(registry=reg, template_index=index)

    tc = [
        TemplateCandidate(
            template_id="openviking.personal_fact_lookup.en.v2",
            primary_backend_id="openviking_memory_backend",
            score=0.80,
            score_components={},
        ),
        TemplateCandidate(
            template_id="graph.entity_relation.v1",
            primary_backend_id="graph_memory_backend",
            score=0.79,
            score_components={},
        ),
    ]
    bc = [
        BackendCandidate(
            backend_id="openviking_memory_backend",
            best_template_id="openviking.personal_fact_lookup.en.v2",
            score=0.80,
        ),
        BackendCandidate(
            backend_id="graph_memory_backend",
            best_template_id="graph.entity_relation.v1",
            score=0.79,
        ),
    ]
    result = dec.decide("q", "q", tc, bc, QueryHints())

    assert result.route_method == "template_embedding_multi_backend"
    assert len(result.routes) == 2
    assert result.routes[0].role == "primary"
    assert result.routes[0].backend_id == "openviking_memory_backend"
    assert result.routes[1].role == "secondary"
    assert result.routes[1].backend_id == "graph_memory_backend"
    assert result.post_retrieval_requirements.get("deduplicate_evidence") is True
    assert result.post_retrieval_requirements.get("check_answerability") is True
    assert not result.fallback.used
