"""RouteDecision — decide the final backend route from template matching scores.

Implements the v1.3/v1.4 decision logic:
    1. Single-backend acceptance (high-confidence, clear winner)
    2. Multi-backend acceptance (top1/top2 different backends, both high)
    3. LLMBackendRouter fallback (low confidence or ambiguous)
"""

import logging
from typing import List, Optional

from echomem.llm_fallback import LLMBackendRouter, LLMFallbackContext, MockLLMBackendRouter
from echomem.matcher import BackendCandidate, TemplateCandidate
from echomem.registry import MemoryBackendRegistry
from echomem.result import (
    DebugInfo,
    FallbackInfo,
    MemBackendRouteResult,
    QueryHints,
    RouteEntry,
)
from echomem.templates import BackendRouteTemplateIndex, MemoryBackendRouteTemplate

logger = logging.getLogger(__name__)

# Route method strings aligned with v1.3 schema
METHOD_TEMPLATE = "template_embedding"
METHOD_MULTI = "template_embedding_multi_backend"
METHOD_LLM_FALLBACK = "llm_backend_fallback"


class RouteDecision:
    """Produce a MemBackendRouteResult from template/backend candidates."""

    # Default backend used when everything else fails.
    DEFAULT_BACKEND_ID = "openviking_memory_backend"

    def __init__(
        self,
        registry: MemoryBackendRegistry,
        template_index: BackendRouteTemplateIndex,
        llm_router: Optional[LLMBackendRouter] = None,
    ) -> None:
        self._registry = registry
        self._template_index = template_index
        self._llm_router = llm_router or MockLLMBackendRouter()

        # Validate that the default backend is registered and enabled.
        if not self._registry.is_enabled(self.DEFAULT_BACKEND_ID):
            raise ValueError(
                f"Default backend '{self.DEFAULT_BACKEND_ID}' is not registered or not enabled. "
                f"Ensure it is added to MemoryBackendRegistry before constructing RouteDecision."
            )

        logger.info(
            "RouteDecision initialized with %d enabled backend(s), %d templates",
            len(registry.list_enabled()),
            len(template_index.enabled_templates()),
        )

    def decide(
        self,
        raw_query: str,
        normalized_query: str,
        template_candidates: List[TemplateCandidate],
        backend_candidates: List[BackendCandidate],
        query_hints: QueryHints,
    ) -> MemBackendRouteResult:
        """Apply decision rules and construct the route result.

        Args:
            raw_query: Original user query.
            normalized_query: Normalized query text.
            template_candidates: All enabled templates sorted by score.
            backend_candidates: Per-backend best candidates sorted by score.
            query_hints: Extracted lightweight hints.

        Returns:
            MemBackendRouteResult with one or more routes.
        """
        logger.debug(
            "Deciding route for query='%s' with %d backend candidate(s)",
            raw_query,
            len(backend_candidates),
        )

        # Filter out candidates whose backend is not registered or disabled.
        # MemRouter v1.4+ only validates backend_id; backend routing only.
        valid_backend_candidates = [
            c for c in backend_candidates
            if self._registry.is_enabled(c.backend_id)
        ]
        dropped = len(backend_candidates) - len(valid_backend_candidates)
        if dropped:
            logger.warning(
                "Dropped %d backend candidate(s) pointing to unregistered/disabled backends",
                dropped,
            )

        if not valid_backend_candidates:
            logger.info("No valid backend candidates; routing via LLM fallback")
            return self._llm_fallback(
                raw_query, normalized_query, template_candidates, query_hints,
                reason="no_enabled_backend_candidates",
            )

        best = valid_backend_candidates[0]
        second: Optional[BackendCandidate] = valid_backend_candidates[1] if len(valid_backend_candidates) > 1 else None

        # Resolve the full template to read thresholds
        best_template = self._resolve_template(best.best_template_id, template_candidates)
        if best_template is None:
            logger.warning("Could not resolve template '%s'; using LLM fallback", best.best_template_id)
            return self._llm_fallback(
                raw_query, normalized_query, template_candidates, query_hints,
                reason="could_not_resolve_best_template",
            )

        accept_thr = best_template.thresholds.accept
        fallback_thr = best_template.thresholds.fallback
        margin_thr = best_template.thresholds.margin
        margin = best.score - (second.score if second else 0.0)

        logger.debug(
            "Best backend=%s score=%.4f accept=%.4f fallback=%.4f margin=%.4f margin_thr=%.4f",
            best.backend_id,
            best.score,
            accept_thr,
            fallback_thr,
            margin,
            margin_thr,
        )

        # --- Branch 1: single-backend acceptance ---
        if best.score >= accept_thr and (second is None or margin >= margin_thr):
            logger.info("RouteDecision: single-backend -> %s", best.backend_id)
            return self._build_single_backend_result(
                raw_query,
                normalized_query,
                best,
                best_template,
                query_hints,
                template_candidates,
            )

        # --- Branch 2: multi-backend acceptance ---
        if second is not None:
            second_template = self._resolve_template(second.best_template_id, template_candidates)
            second_accept = second_template.thresholds.accept if second_template else 0.0

            score_ratio = second.score / best.score if best.score > 0 else 0.0
            if (
                best.score >= accept_thr
                and second.score >= second_accept
                and best.backend_id != second.backend_id
                and margin < margin_thr
                and score_ratio >= 0.85
            ):
                logger.info(
                    "RouteDecision: multi-backend -> %s (primary) + %s (secondary) "
                    "(ratio=%.3f)",
                    best.backend_id,
                    second.backend_id,
                    score_ratio,
                )
                return self._build_multi_backend_result(
                    raw_query,
                    normalized_query,
                    best,
                    second,
                    best_template,
                    second_template,
                    query_hints,
                    template_candidates,
                )

        # --- Branch 3: LLM fallback ---
        if best.score < fallback_thr:
            reason = "template_score_below_fallback_threshold"
        else:
            reason = "ambiguous_or_margin_too_small"
        logger.info("RouteDecision: LLM fallback (reason=%s)", reason)
        return self._llm_fallback(
            raw_query, normalized_query, template_candidates, query_hints, reason=reason
        )

    # ------------------------------------------------------------------ #
    # Result builders
    # ------------------------------------------------------------------ #

    def _build_single_backend_result(
        self,
        raw_query: str,
        normalized_query: str,
        backend: BackendCandidate,
        template: MemoryBackendRouteTemplate,
        query_hints: QueryHints,
        template_candidates: List[TemplateCandidate],
    ) -> MemBackendRouteResult:
        route = RouteEntry(
            backend_id=backend.backend_id,
            backend_kind=self._resolve_backend_kind(backend.backend_id),
            role="primary",
            confidence=round(backend.score, 4),
            matched_template_id=backend.best_template_id,
            query_hints=query_hints,
        )
        return MemBackendRouteResult(
            raw_user_query=raw_query,
            normalized_user_query=normalized_query,
            route_method=METHOD_TEMPLATE,
            routes=[route],
            fallback=FallbackInfo(used=False),
            debug=DebugInfo(
                top_templates=[
                    {
                        "template_id": c.template_id,
                        "backend_id": c.primary_backend_id,
                        "score": round(c.score, 4),
                        "score_components": c.score_components,
                    }
                    for c in template_candidates[:5]
                ]
            ),
        )

    def _build_multi_backend_result(
        self,
        raw_query: str,
        normalized_query: str,
        primary: BackendCandidate,
        secondary: BackendCandidate,
        primary_template: Optional[MemoryBackendRouteTemplate],
        secondary_template: Optional[MemoryBackendRouteTemplate],
        query_hints: QueryHints,
        template_candidates: List[TemplateCandidate],
    ) -> MemBackendRouteResult:
        primary_route = RouteEntry(
            backend_id=primary.backend_id,
            backend_kind=self._resolve_backend_kind(primary.backend_id),
            role="primary",
            confidence=round(primary.score, 4),
            matched_template_id=primary.best_template_id,
            query_hints=query_hints,
        )
        secondary_route = RouteEntry(
            backend_id=secondary.backend_id,
            backend_kind=self._resolve_backend_kind(secondary.backend_id),
            role="secondary",
            confidence=round(secondary.score, 4),
            matched_template_id=secondary.best_template_id,
            query_hints=query_hints,
        )
        return MemBackendRouteResult(
            raw_user_query=raw_query,
            normalized_user_query=normalized_query,
            route_method=METHOD_MULTI,
            routes=[primary_route, secondary_route],
            post_retrieval_requirements={"deduplicate_evidence": True, "check_answerability": True},
            fallback=FallbackInfo(used=False),
            debug=DebugInfo(
                top_templates=[
                    {
                        "template_id": c.template_id,
                        "backend_id": c.primary_backend_id,
                        "score": round(c.score, 4),
                        "score_components": c.score_components,
                    }
                    for c in template_candidates[:5]
                ]
            ),
        )

    def _llm_fallback(
        self,
        raw_query: str,
        normalized_query: str,
        template_candidates: List[TemplateCandidate],
        query_hints: QueryHints,
        reason: str,
    ) -> MemBackendRouteResult:
        context = LLMFallbackContext(
            raw_user_query=raw_query,
            normalized_user_query=normalized_query,
            registry=self._registry,
            failed_template_summary=[
                {
                    "template_id": c.template_id,
                    "backend_id": c.primary_backend_id,
                    "score": round(c.score, 4),
                    "score_components": c.score_components,
                }
                for c in template_candidates[:5]
            ],
            query_hints=query_hints,
            fallback_reason=reason,
        )
        return self._llm_router.route(context)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _resolve_template(
        self,
        template_id: str,
        candidates: List[TemplateCandidate],
    ) -> Optional[MemoryBackendRouteTemplate]:
        """Retrieve full template from the template index."""
        return self._template_index.get(template_id)

    def _resolve_backend_kind(self, backend_id: str) -> str:
        entry = self._registry.get(backend_id)
        return entry.backend_kind if entry else "unknown"
