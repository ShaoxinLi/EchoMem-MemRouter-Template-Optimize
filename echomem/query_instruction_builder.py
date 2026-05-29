"""QueryInstructionBuilder — generates concrete BackendQueryInstructions from route results.

Implements the Template-First Hybrid design (Option 3):
    1. Template YAML provides base query_spec (primary source of truth)
    2. Adapter provides runtime enrichment (e.g. entity interpolation, hint-based tuning)
    3. If template has no query_spec, fallback to adapter default mapping
"""

import logging
from typing import Any, Dict, List, Optional

from echomem.adapters.openviking import MemoryBackendAdapter
from echomem.query_spec import (
    BackendQueryInstruction,
    GraphQuerySpec,
    OpenVikingQuerySpec,
    TemplateQuerySpec,
    TemporalQuerySpec,
    TypedQueryTemplate,
)
from echomem.result import MemBackendRouteResult, QueryHints, RouteEntry
from echomem.templates import BackendRouteTemplateIndex

logger = logging.getLogger(__name__)


class QueryInstructionBuilder:
    """Build BackendQueryInstruction list from a MemBackendRouteResult.

    Usage:
        builder = QueryInstructionBuilder(template_index, adapters={"openviking_memory_backend": ov_adapter})
        instructions = builder.build(route_result, query_hints)
    """

    def __init__(
        self,
        template_index: BackendRouteTemplateIndex,
        adapters: Optional[Dict[str, MemoryBackendAdapter]] = None,
    ) -> None:
        self._template_index = template_index
        self._adapters = adapters or {}
        logger.info(
            "QueryInstructionBuilder initialized with %d templates, %d adapters",
            len(template_index),
            len(self._adapters),
        )

    def build(
        self,
        route_result: MemBackendRouteResult,
        query_hints: QueryHints,
    ) -> List[BackendQueryInstruction]:
        """Generate query instructions for every route in the result.

        Args:
            route_result: The output of RouteDecision.
            query_hints: Extracted query hints (entities, temporal, relation).

        Returns:
            List of BackendQueryInstruction, one per route.
        """
        instructions: List[BackendQueryInstruction] = []
        for route in route_result.routes:
            inst = self._build_for_route(route, route_result, query_hints)
            if inst is not None:
                instructions.append(inst)
        logger.info(
            "Generated %d query instruction(s) for %d route(s)",
            len(instructions),
            len(route_result.routes),
        )
        return instructions

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _build_for_route(
        self,
        route: RouteEntry,
        route_result: MemBackendRouteResult,
        query_hints: QueryHints,
    ) -> Optional[BackendQueryInstruction]:
        template = (
            self._template_index.get(route.matched_template_id)
            if route.matched_template_id
            else None
        )

        # Step 1: obtain base spec (template-first, adapter fallback)
        base_spec: Optional[TemplateQuerySpec] = None
        intent_family_name = ""
        if template is not None:
            intent_family_name = template.intent_family.name
            if template.query_spec is not None:
                base_spec = template.query_spec
                logger.debug(
                    "Using template query_spec for %s", route.matched_template_id
                )

        if base_spec is None:
            adapter = self._adapters.get(route.backend_id)
            if adapter is not None and hasattr(adapter, "get_default_spec"):
                base_spec = adapter.get_default_spec(intent_family_name)
                logger.debug(
                    "Using adapter default spec for backend=%s intent=%s",
                    route.backend_id,
                    intent_family_name,
                )

        if base_spec is None:
            logger.debug(
                "No query_spec or adapter fallback for backend=%s; skipping instruction",
                route.backend_id,
            )
            return None

        # Step 2: variable interpolation for query text
        query_text = self._interpolate_query(
            base_spec.query_rewrite,
            route_result,
            query_hints,
        )
        if not query_text:
            query_text = route_result.raw_user_query

        # Step 3: adapter runtime enrichment
        adapter = self._adapters.get(route.backend_id)
        if adapter is not None and hasattr(adapter, "enrich_spec"):
            base_spec = adapter.enrich_spec(base_spec, query_hints, route_result.raw_user_query)

        # Step 4: assemble BackendQueryInstruction from backend-specific spec
        return self._assemble_instruction(route, base_spec, query_text, query_hints)

    @staticmethod
    def _interpolate_query(
        query_rewrite: Optional[str],
        route_result: MemBackendRouteResult,
        query_hints: QueryHints,
    ) -> Optional[str]:
        """Replace placeholders in query_rewrite with actual values.

        Supported placeholders:
            {raw_user_query}      → route_result.raw_user_query
            {normalized_user_query} → route_result.normalized_user_query
            {entities}            → comma-joined entity list
            {temporal_hints}      → comma-joined temporal hints
            {relation_hints}      → comma-joined relation hints
        """
        if not query_rewrite:
            return None

        text = query_rewrite
        text = text.replace("{raw_user_query}", route_result.raw_user_query)
        text = text.replace("{normalized_user_query}", route_result.normalized_user_query)
        text = text.replace("{entities}", ", ".join(query_hints.entities))
        text = text.replace("{temporal_hints}", ", ".join(query_hints.temporal_hints))
        text = text.replace("{relation_hints}", ", ".join(query_hints.relation_hints))
        return text

    @staticmethod
    def _assemble_instruction(
        route: RouteEntry,
        base_spec: TemplateQuerySpec,
        query_text: str,
        query_hints: QueryHints,
    ) -> BackendQueryInstruction:
        """Convert a TemplateQuerySpec into a BackendQueryInstruction.

        Extracts backend-specific fields (openviking / graph / temporal) and
        places them into the generic instruction structure.
        """
        # Determine which backend-specific spec to use
        backend_id = route.backend_id
        ov_spec = base_spec.openviking
        graph_spec = base_spec.graph
        temporal_spec = base_spec.temporal

        instruction = BackendQueryInstruction(
            backend_id=backend_id,
            query=query_text,
            search_mode=base_spec.search_mode,
        )

        if ov_spec is not None:
            instruction.target_uri = ov_spec.target_uri
            instruction.context_type = ov_spec.context_type
            instruction.level = ov_spec.level
            instruction.filter = ov_spec.filter
            instruction.skip_intent_analysis = ov_spec.skip_intent_analysis
            if ov_spec.typed_query_template is not None:
                # Also interpolate variables inside the typed query template
                tq = ov_spec.typed_query_template
                tq_query_template = tq.query or query_text
                interpolated_tq_query = tq_query_template.replace(
                    "{raw_user_query}", query_text
                )
                instruction.typed_query = TypedQueryTemplate(
                    query=interpolated_tq_query,
                    context_type=tq.context_type or ov_spec.context_type,
                    intent=tq.intent,
                    priority=tq.priority,
                    target_directories=tq.target_directories,
                )
            # Forward any remaining OpenViking-specific params
            instruction.extra_params.update(ov_spec.model_dump(exclude={
                "context_type", "target_uri", "level", "filter",
                "skip_intent_analysis", "typed_query_template",
            }))

        elif graph_spec is not None:
            instruction.search_mode = "graph_traversal"
            instruction.extra_params.update(graph_spec.model_dump())

        elif temporal_spec is not None:
            instruction.search_mode = "temporal_query"
            instruction.extra_params.update(temporal_spec.model_dump())

        # If no backend-specific spec was present, keep the generic instruction
        # (consumer will treat it as a basic find/search)
        return instruction
