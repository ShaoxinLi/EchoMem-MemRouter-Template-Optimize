"""MemRouterPipeline — assembles the full routing pipeline.

Provides a single entry point `route(query)` that runs through all stages:
    QueryNormalizer → QueryFeatureBuilder → TemplateMatcher → RouteDecision
"""

import logging
from pathlib import Path
from typing import Optional

from echomem.adapters.openviking import OpenVikingAdapter
from echomem.decision import RouteDecision
from echomem.embeddings.base import EmbeddingProvider
from echomem.features import QueryFeatureBuilder
from echomem.llm_fallback import LLMRouterConfig, create_llm_backend_router
from echomem.matcher import TemplateMatcher
from echomem.normalizer import QueryNormalizer
from echomem.query_instruction_builder import QueryInstructionBuilder
from echomem.registry import BackendEntry, CostProfile, MemoryBackendRegistry, QueryContract
from echomem.request import MemoryRouteRequest
from echomem.result import MemBackendRouteResult, QueryHints
from echomem.templates import BackendRouteTemplateIndex

logger = logging.getLogger(__name__)

# Builtin template directory relative to this package
_BUILTIN_TEMPLATES_DIR = Path(__file__).parent / "templates_data"


class MemRouterPipeline:
    """Main routing pipeline for EchoMem MemRouter.

    Usage:
        pipeline = MemRouterPipeline.with_defaults(embedder)
        result = pipeline.route("你记得我之前说过什么吗？")
    """

    def __init__(
        self,
        registry: MemoryBackendRegistry,
        feature_builder: QueryFeatureBuilder,
        template_index: BackendRouteTemplateIndex,
        matcher: TemplateMatcher,
        decision: RouteDecision,
        query_instruction_builder: Optional[QueryInstructionBuilder] = None,
    ) -> None:
        self._registry = registry
        self._feature_builder = feature_builder
        self._template_index = template_index
        self._matcher = matcher
        self._decision = decision
        self._query_instruction_builder = query_instruction_builder
        logger.info("MemRouterPipeline initialized")

    def route_request(self, request: MemoryRouteRequest) -> MemBackendRouteResult:
        """Route a MemoryRouteRequest through the full pipeline.

        Args:
            request: Standardized memory route request.

        Returns:
            MemBackendRouteResult containing backend route and metadata.
        """
        raw_query = request.raw_user_query
        logger.info("Routing query: %s", raw_query)

        # Use caller-provided normalized query if available, else normalize
        if request.normalized_user_query:
            normalized_query = request.normalized_user_query
        else:
            normalized_query = self._feature_builder._normalizer.normalize(raw_query)

        # Stage 1: feature extraction
        # Pass caller-provided normalized query so embedding uses the same text
        # that will appear in the result; hints are still extracted from raw_query.
        features = self._feature_builder.build(
            raw_query=raw_query,
            normalized_query=normalized_query,
        )
        logger.debug(
            "Features extracted: entities=%s temporal=%s relation=%s",
            features.entities,
            features.temporal_hints,
            features.relation_hints,
        )

        # Stage 2: template matching
        template_cands, backend_cands = self._matcher.match(features)

        # Stage 3: route decision
        query_hints = QueryHints(
            entities=features.entities,
            temporal_hints=features.temporal_hints,
            relation_hints=features.relation_hints,
        )
        result = self._decision.decide(
            raw_query=raw_query,
            normalized_query=normalized_query,
            template_candidates=template_cands,
            backend_candidates=backend_cands,
            query_hints=query_hints,
        )

        # Stage 4: query instruction generation (optional, Template-First Hybrid)
        if self._query_instruction_builder is not None:
            result.query_instructions = self._query_instruction_builder.build(
                route_result=result,
                query_hints=query_hints,
            )

        logger.info(
            "Route result: method=%s backend=%s confidence=%s instructions=%d",
            result.route_method,
            result.routes[0].backend_id if result.routes else "none",
            result.routes[0].confidence if result.routes else "none",
            len(result.query_instructions),
        )
        return result

    def route(self, raw_query: str) -> MemBackendRouteResult:
        """Convenience method that wraps a raw query into MemoryRouteRequest.

        Args:
            raw_query: Original user query string.

        Returns:
            MemBackendRouteResult containing backend route and metadata.
        """
        return self.route_request(MemoryRouteRequest(raw_user_query=raw_query))

    @classmethod
    def with_defaults(
        cls,
        embedder: EmbeddingProvider,
        template_dir: Optional[Path] = None,
        llm_router_config: Optional[LLMRouterConfig] = None,
    ) -> "MemRouterPipeline":
        """Factory that wires up the full pipeline with v1.4 defaults.

        Args:
            embedder: Embedding provider (sentence-transformers or OpenAI).
            template_dir: Optional directory of YAML templates. If None,
                uses the builtin templates shipped with the package.
            llm_router_config: Optional LLM router config. If None, uses mock
                (no real LLM calls). Pass a real config for benchmark evaluation.

        Returns:
            Configured MemRouterPipeline ready for routing.
        """
        logger.info("Building MemRouterPipeline with default v1.4 configuration")

        # 1. Registry — three logical backends for v1.4 route layer validation
        registry = MemoryBackendRegistry()

        registry.register(
            BackendEntry(
                backend_id="openviking_memory_backend",
                backend_kind="openviking_native",
                status="enabled",
                description="OpenViking native memory backend for personal semantic memory, profile, preferences, and general user context.",
                query_contract=QueryContract(
                    input_format="natural_language_with_hints",
                    supports_entities=True,
                    supports_time_range=True,
                    supports_relation_hints=False,
                ),
            )
        )

        registry.register(
            BackendEntry(
                backend_id="graph_memory_backend",
                backend_kind="knowledge_graph",
                status="enabled",
                description="Graph memory logical backend for entity relations, multi-hop queries, and co-participation.",
                query_contract=QueryContract(
                    input_format="natural_language_with_hints",
                    supports_entities=True,
                    supports_time_range=False,
                    supports_relation_hints=True,
                ),
                cost_profile=CostProfile(latency_class="unknown", token_cost_class="unknown"),
            )
        )

        registry.register(
            BackendEntry(
                backend_id="temporal_memory_backend",
                backend_kind="temporal_store",
                status="enabled",
                description="Temporal memory logical backend for timeline facts, sequence reasoning, and time-range queries. Physical execution currently falls back to OpenViking native search as the real temporal backend is not yet connected.",
                query_contract=QueryContract(
                    input_format="natural_language_with_hints",
                    supports_entities=True,
                    supports_time_range=True,
                    supports_relation_hints=False,
                ),
                cost_profile=CostProfile(latency_class="unknown", token_cost_class="unknown"),
            )
        )
        logger.info(
            "Registered %d logical backend(s), enabled=%s",
            len(registry),
            registry.enabled_backend_ids(),
        )

        # 2. Feature builder
        normalizer = QueryNormalizer()
        feature_builder = QueryFeatureBuilder(embedder=embedder, normalizer=normalizer)

        # 3. Template index
        template_index = BackendRouteTemplateIndex()
        load_dir = template_dir or _BUILTIN_TEMPLATES_DIR
        loaded = template_index.load_from_directory(load_dir)
        if loaded == 0:
            logger.warning("No templates loaded from %s; routing will always default", load_dir)

        # 4. Matcher
        matcher = TemplateMatcher(embedder=embedder, template_index=template_index)

        # 5. LLM router (mock by default, real for benchmark)
        llm_router = create_llm_backend_router(
            llm_router_config or LLMRouterConfig(provider="mock", model="mock")
        )

        # 6. Decision
        decision = RouteDecision(
            registry=registry,
            template_index=template_index,
            llm_router=llm_router,
        )

        # 7. Adapters & QueryInstructionBuilder (Template-First Hybrid)
        adapters = {
            "openviking_memory_backend": OpenVikingAdapter(),
        }
        query_instruction_builder = QueryInstructionBuilder(
            template_index=template_index,
            adapters=adapters,
        )

        return cls(
            registry=registry,
            feature_builder=feature_builder,
            template_index=template_index,
            matcher=matcher,
            decision=decision,
            query_instruction_builder=query_instruction_builder,
        )
