"""TemplateMatcher — multi-prototype vector matching with hard-negative penalty.

Implements the v1.3/v1.4 scoring algorithm:
    S_pos  = 0.50 * S_max + 0.30 * S_mean@3 + 0.20 * S_centroid
    S_final = S_pos - lambda * max(0, delta_neg - M_neg)

Backend aggregation runs for all enabled templates regardless of backend.
The algorithm is kept backend-agnostic so that multi-backend expansion
requires zero code changes here.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from echomem.embeddings.base import EmbeddingProvider
from echomem.features import QueryFeatures
from echomem.normalizer import QueryNormalizer
from echomem.templates import BackendRouteTemplateIndex, MemoryBackendRouteTemplate

logger = logging.getLogger(__name__)

# Score combination weights (from v1.3/v1.4 design doc)
_WEIGHT_MAX = 0.50
_WEIGHT_MEAN3 = 0.30
_WEIGHT_CENTROID = 0.20
_TOP_K_MEAN = 3


@dataclass
class TemplateCandidate:
    """Scoring result for a single template."""

    template_id: str
    primary_backend_id: str
    score: float
    score_components: Dict[str, Any]


@dataclass
class BackendCandidate:
    """Aggregated candidate per backend (used in multi-backend scenarios)."""

    backend_id: str
    best_template_id: str
    score: float


class TemplateMatcher:
    """Match user queries against template prototypes using vector similarity."""

    def __init__(self, embedder: EmbeddingProvider, template_index: BackendRouteTemplateIndex) -> None:
        self._embedder = embedder
        self._template_index = template_index
        # Cache for prototype embeddings: {template_id: (prototype_embeddings, centroid)}
        self._proto_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        # Cache for hard-negative embeddings: {template_id: negative_embeddings}
        self._neg_cache: Dict[str, np.ndarray] = {}
        self._normalizer = QueryNormalizer()
        logger.info(
            "TemplateMatcher initialized with %d templates",
            len(template_index.enabled_templates()),
        )

    def match(self, features: QueryFeatures) -> Tuple[List[TemplateCandidate], List[BackendCandidate]]:
        """Run full template matching pipeline.

        Args:
            features: Extracted query features including embedding.

        Returns:
            (template_candidates, backend_candidates)
            template_candidates: all enabled templates sorted by final score.
            backend_candidates: per-backend best scores, sorted descending.
        """
        query_vec = features.query_embedding
        logger.debug("Starting template matching for query: %s", features.normalized_query)

        template_candidates: List[TemplateCandidate] = []
        for template in self._template_index.enabled_templates():
            score, components = self._score_template(template, query_vec)
            template_candidates.append(
                TemplateCandidate(
                    template_id=template.template_id,
                    primary_backend_id=template.target.primary_backend_id,
                    score=score,
                    score_components=components,
                )
            )

        # Sort by score descending
        template_candidates.sort(key=lambda c: c.score, reverse=True)
        logger.debug(
            "Template ranking (top3): %s",
            [(c.template_id, round(c.score, 4)) for c in template_candidates[:3]],
        )

        # Aggregate to backend candidates
        backend_candidates = self._aggregate_backends(template_candidates)
        logger.debug(
            "Backend ranking: %s",
            [(c.backend_id, round(c.score, 4)) for c in backend_candidates],
        )

        return template_candidates, backend_candidates

    def _score_template(
        self,
        template: MemoryBackendRouteTemplate,
        query_vec: np.ndarray,
    ) -> Tuple[float, Dict[str, Any]]:
        """Compute final score for a single template.

        Returns:
            (final_score, component_dict)
        """
        # --- Positive prototype scoring ---
        proto_embeddings, centroid = self._get_prototype_embeddings(template)
        sims = proto_embeddings @ query_vec
        s_max = float(np.max(sims))
        top_proto = self._top_matched_prototype(template, sims)
        # Top-K mean (guard against templates with fewer than _TOP_K_MEAN prototypes)
        k = min(_TOP_K_MEAN, len(sims))
        if k > 0:
            top_k_indices = np.argpartition(sims, -k)[-k:]
            s_mean_k = float(np.mean(sims[top_k_indices]))
        else:
            s_mean_k = 0.0
        s_centroid = float(centroid @ query_vec)

        s_pos = _WEIGHT_MAX * s_max + _WEIGHT_MEAN3 * s_mean_k + _WEIGHT_CENTROID * s_centroid

        # --- Hard-negative penalty ---
        neg_embeddings = self._get_negative_embeddings(template)
        if len(neg_embeddings) > 0:
            neg_sims = neg_embeddings @ query_vec
            s_neg = float(np.max(neg_sims))
            top_hard_negative = self._top_hard_negative(template, neg_sims)
            m_neg = s_pos - s_neg
            delta_neg = template.thresholds.hard_negative_margin
            lambda_pen = template.thresholds.hard_negative_penalty
            penalty = lambda_pen * max(0.0, delta_neg - m_neg)
            s_final = s_pos - penalty
        else:
            s_neg = 0.0
            top_hard_negative = None
            penalty = 0.0
            s_final = s_pos

        components = {
            "s_max": round(s_max, 4),
            "s_mean@3": round(s_mean_k, 4),
            "s_centroid": round(s_centroid, 4),
            "s_pos": round(s_pos, 4),
            "s_neg": round(s_neg, 4),
            "penalty": round(penalty, 4),
            "s_final": round(s_final, 4),
            "top_matched_prototype": top_proto,
            "top_hard_negative": top_hard_negative,
        }
        return s_final, components

    @staticmethod
    def _top_matched_prototype(
        template: MemoryBackendRouteTemplate,
        sims: np.ndarray,
    ) -> Dict[str, Any] | None:
        if not template.query_prototypes:
            return None
        idx = int(np.argmax(sims[: len(template.query_prototypes)]))
        return {
            "index": idx,
            "text": template.query_prototypes[idx],
            "score": round(float(sims[idx]), 4),
        }

    @staticmethod
    def _top_hard_negative(
        template: MemoryBackendRouteTemplate,
        sims: np.ndarray,
    ) -> Dict[str, Any] | None:
        if not template.hard_negatives:
            return None
        idx = int(np.argmax(sims[: len(template.hard_negatives)]))
        hard_negative = template.hard_negatives[idx]
        return {
            "index": idx,
            "query": hard_negative.query,
            "confusing_with_backend": hard_negative.confusing_with_backend,
            "reason": hard_negative.reason,
            "score": round(float(sims[idx]), 4),
        }

    def _get_prototype_embeddings(
        self, template: MemoryBackendRouteTemplate
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return cached prototype embeddings and centroid for a template."""
        if template.template_id in self._proto_cache:
            return self._proto_cache[template.template_id]

        texts = [self._normalizer.normalize(text) for text in template.query_prototypes]
        if not texts:
            # No prototypes: create zero embeddings so the template scores 0
            dim = self._embedder.dimension()
            embeddings = np.zeros((1, dim), dtype=np.float32)
            centroid = np.zeros(dim, dtype=np.float32)
            self._proto_cache[template.template_id] = (embeddings, centroid)
            return embeddings, centroid

        embeddings = self._embedder.embed(texts)
        # Normalize each row
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        # Centroid = normalized mean of all prototypes
        centroid = np.mean(embeddings, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm
        else:
            centroid = centroid  # zeros

        self._proto_cache[template.template_id] = (embeddings, centroid)
        return embeddings, centroid

    def _get_negative_embeddings(self, template: MemoryBackendRouteTemplate) -> np.ndarray:
        """Return cached hard-negative embeddings for a template."""
        if template.template_id in self._neg_cache:
            return self._neg_cache[template.template_id]

        texts = [self._normalizer.normalize(hn.query) for hn in template.hard_negatives]
        if not texts:
            self._neg_cache[template.template_id] = np.zeros((0, self._embedder.dimension()), dtype=np.float32)
            return self._neg_cache[template.template_id]

        embeddings = self._embedder.embed(texts)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        self._neg_cache[template.template_id] = embeddings
        return embeddings

    @staticmethod
    def _aggregate_backends(
        template_candidates: List[TemplateCandidate],
    ) -> List[BackendCandidate]:
        """Aggregate template candidates into per-backend best scores.

        If multiple templates map to the same backend, only the highest-scoring
        template contributes to that backend's candidate. This ensures that
        top1/top2 in RouteDecision represent distinct backends.
        """
        best_by_backend: Dict[str, TemplateCandidate] = {}
        for cand in template_candidates:
            bid = cand.primary_backend_id
            if bid not in best_by_backend or cand.score > best_by_backend[bid].score:
                best_by_backend[bid] = cand

        backend_cands = [
            BackendCandidate(
                backend_id=cand.primary_backend_id,
                best_template_id=cand.template_id,
                score=cand.score,
            )
            for cand in best_by_backend.values()
        ]
        backend_cands.sort(key=lambda c: c.score, reverse=True)
        return backend_cands
