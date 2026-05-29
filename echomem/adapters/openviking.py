"""OpenVikingAdapter — bridges MemBackendRouteResult to OpenViking retrieval.

In v1.4 the adapter is intentionally thin. MemRouter decides "route to OpenViking"
and supplies query hints; the adapter translates those hints into parameters that
OpenViking's own HierarchicalRetriever / IntentAnalyzer can consume.

The adapter does NOT reimplement OpenViking retrieval logic.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from echomem.query_spec import TemplateQuerySpec
from echomem.result import MemBackendRouteResult, QueryHints

logger = logging.getLogger(__name__)


class MemoryBackendAdapter(ABC):
    """Abstract adapter for a memory backend.

    Each backend (OpenViking, graph, temporal) implements this interface
    so that the dispatcher can treat them uniformly.
    """

    @property
    @abstractmethod
    def backend_id(self) -> str:
        """Return the backend_id this adapter handles."""
        ...

    @abstractmethod
    def translate(self, route_result: MemBackendRouteResult) -> Dict[str, Any]:
        """Translate a MemBackendRouteResult into backend-native query parameters.

        Returns:
            Dictionary of parameters for the backend's search API.
        """
        ...

    # Optional hooks for Template-First Hybrid query instruction generation.
    # Adapters that do not implement these will simply not contribute fallback
    # specs or runtime enrichment.

    def get_default_spec(self, intent_family: str) -> Optional[TemplateQuerySpec]:
        """Return a default TemplateQuerySpec for an intent family (fallback)."""
        return None

    def enrich_spec(
        self,
        base_spec: TemplateQuerySpec,
        hints: QueryHints,
        raw_query: str,
    ) -> TemplateQuerySpec:
        """Runtime enrichment of a base spec using query hints.

        Default implementation returns the spec unchanged.
        Subclasses may override to adjust parameters based on entities,
        temporal hints, etc.
        """
        return base_spec


class OpenVikingAdapter(MemoryBackendAdapter):
    """Adapter for OpenViking native memory backend.

    Translates MemRouter outputs into OpenViking search parameters.
    In the minimal executable version this is a parameter mapper; actual
    network or SDK calls can be added later without changing MemRouter.
    """

    BACKEND_ID = "openviking_memory_backend"

    # Fallback mapping: intent_family -> OpenViking query parameters.
    # Used when a template does NOT declare its own query_spec.
    _INTENT_QUERY_MAP: Dict[str, Dict[str, Any]] = {
        "personal_fact_lookup": {
            "context_type": "memory",
            "target_uri": "viking://memories",
            "search_mode": "search",
            "skip_intent_analysis": True,
        },
        "preference_profile": {
            "context_type": "memory",
            "target_uri": "viking://memories/preferences",
            "search_mode": "search",
            "skip_intent_analysis": True,
        },
        "aggregation_summary": {
            "context_type": "memory",
            "target_uri": "viking://memories",
            "search_mode": "search",
            "skip_intent_analysis": True,
        },
        "previous_chat_recall": {
            "context_type": "memory",
            "target_uri": "viking://memories/sessions",
            "search_mode": "search",
            "skip_intent_analysis": False,
        },
    }
    _DEFAULT_SEARCH_SPEC: Dict[str, Any] = {
        "context_type": "memory",
        "target_uri": "viking://memories",
        "search_mode": "search",
        # This conservative fallback delegates fine-grained intent handling to
        # OpenViking native search/IntentAnalyzer when MemRouter only knows
        # that the correct backend is OpenViking.
        "skip_intent_analysis": False,
    }

    def __init__(self) -> None:
        logger.info("OpenVikingAdapter initialized")

    @property
    def backend_id(self) -> str:
        return self.BACKEND_ID

    def get_default_spec(self, intent_family: str) -> Optional[TemplateQuerySpec]:
        """Return a TemplateQuerySpec from the adapter fallback mapping."""
        spec_dict = self._INTENT_QUERY_MAP.get(intent_family)
        if spec_dict is None and not intent_family:
            spec_dict = self._DEFAULT_SEARCH_SPEC
        if spec_dict is None:
            logger.debug(
                "No adapter fallback mapping for intent_family=%s", intent_family
            )
            return None

        from echomem.query_spec import OpenVikingQuerySpec, TypedQueryTemplate

        typed_query_template = None
        if intent_family:
            typed_query_template = TypedQueryTemplate(
                intent=intent_family,
                priority=spec_dict.get("priority", 1),
                target_directories=spec_dict.get("target_directories"),
            )

        ov = OpenVikingQuerySpec(
            context_type=spec_dict.get("context_type", "memory"),
            target_uri=spec_dict.get("target_uri", "viking://memories"),
            level=spec_dict.get("level"),
            skip_intent_analysis=spec_dict.get("skip_intent_analysis", False),
            typed_query_template=typed_query_template,
        )
        return TemplateQuerySpec(
            search_mode=spec_dict.get("search_mode", "find"),
            openviking=ov,
        )

    def enrich_spec(
        self,
        base_spec: TemplateQuerySpec,
        hints: QueryHints,
        raw_query: str,
    ) -> TemplateQuerySpec:
        """Runtime enrichment: tune OpenViking params based on query hints.

        Current enrichments:
            - If temporal_hints present and level does not already include L1,
              add L1 to help time-sensitive retrieval.
            - If entities present, inject them into typed_query_template query
              (if one exists) for more precise vector search.
        """
        if base_spec.openviking is None:
            return base_spec

        import copy

        spec = copy.deepcopy(base_spec)
        ov = spec.openviking
        assert ov is not None

        # Enrichment 1: temporal hints -> include L1
        if hints.temporal_hints and ov.level is not None and 1 not in ov.level:
            enriched_levels = sorted(set(ov.level + [1]))
            ov.level = enriched_levels
            logger.debug(
                "Enriched level for temporal hints: %s -> %s",
                base_spec.openviking.level,
                enriched_levels,
            )

        # Enrichment 2: entities -> inject into typed_query_template for precision
        if hints.entities and ov.typed_query_template is not None:
            original_query = ov.typed_query_template.query or raw_query
            # Append entity names to the typed query to sharpen vector search
            entity_suffix = " " + " ".join(hints.entities)
            enriched_query = original_query + entity_suffix
            ov.typed_query_template = ov.typed_query_template.model_copy(update={
                "query": enriched_query
            })
            logger.debug(
                "Enriched typed_query with entities: '%s' -> '%s'",
                original_query,
                enriched_query,
            )

        return spec

    def translate(self, route_result: MemBackendRouteResult) -> Dict[str, Any]:
        """Build OpenViking-native search parameters from a route result.

        The returned dict is designed to be passed to OpenViking's
        SearchService.search() or find() methods.
        """
        if not route_result.routes:
            logger.warning("Route result has no routes; returning empty translation")
            return {}

        # Find the route targeting this adapter's backend; fall back to routes[0] only for safety.
        route = next(
            (r for r in route_result.routes if r.backend_id == self.BACKEND_ID),
            route_result.routes[0],
        )
        if route.backend_id != self.BACKEND_ID:
            raise ValueError(
                f"OpenVikingAdapter cannot translate route for backend '{route.backend_id}'; "
                f"expected '{self.BACKEND_ID}'. Use the correct adapter for this backend."
            )
        hints: QueryHints = route.query_hints

        # Assemble the native query dict.
        # MemRouter does not decide OpenViking internal directory paths;
        # target_uri narrowing is left to OpenViking's own IntentAnalyzer.
        params: Dict[str, Any] = {
            "query": route_result.raw_user_query,
            "confidence": route.confidence,
            "route_method": route_result.route_method,
            "matched_template_id": route.matched_template_id,
        }

        # Pass hints through for OpenViking's own IntentAnalyzer to leverage
        if hints.entities or hints.temporal_hints or hints.semantic_hints:
            params["query_hints"] = {
                "entities": hints.entities,
                "temporal_hints": hints.temporal_hints,
                "semantic_hints": hints.semantic_hints,
            }

        logger.debug(
            "Translated route to OpenViking params: backend=%s",
            route.backend_id,
        )
        return params

    def execute_mock(self, route_result: MemBackendRouteResult) -> Dict[str, Any]:
        """Mock execution for testing and integration validation.

        Logs what would be sent to OpenViking without making a real call.
        """
        params = self.translate(route_result)
        logger.info(
            "[Mock OpenViking call] backend=%s params=%s",
            self.backend_id,
            params,
        )
        return {"status": "mock_ok", "params": params}
