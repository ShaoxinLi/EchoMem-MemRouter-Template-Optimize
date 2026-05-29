"""QueryFeatureBuilder — lightweight feature extraction for template matching.

Produces embedding vectors and weak hints. Does NOT classify intent or choose
backend; those decisions are left to TemplateMatcher and RouteDecision.
"""

import logging
import re
from dataclasses import dataclass
from typing import List

import numpy as np

from echomem.embeddings.base import EmbeddingProvider
from echomem.normalizer import QueryNormalizer

logger = logging.getLogger(__name__)

# Simple regex-based entity extraction (names, projects, files, products).
# This is intentionally lightweight; heavy NER can be added later without
# breaking the pipeline contract.
_ENTITY_PATTERNS = [
    re.compile(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*"),  # English names like "Jon", "Gina Smith"
    # CJK entity extraction is intentionally disabled in v1.4 MVP to avoid
    # over-extraction of common phrases. Re-enable with a curated dictionary
    # or NER model when needed.
    re.compile(r"[a-zA-Z0-9_\-\.]+\.(?:py|js|ts|go|rs|java|cpp|c|h|md|txt|yaml|json|sql)"),  # filenames
]

# Lightweight temporal hint extraction
_TEMPORAL_PATTERNS = re.compile(
    r"(上次|最近|昨天|今天|明天|上周|下周|上个月|下个月|"
    r"去年|今年|明年|刚刚|刚才|之前|以后|"
    r"\d{4}年|\d{1,2}月|\d{1,2}日|"
    r"last\s+(week|month|year)|next\s+(week|month|year)|"
    r"yesterday|today|tomorrow|recently|last\s+time)",
    re.IGNORECASE,
)

# Lightweight relation hint extraction
_RELATION_PATTERNS = re.compile(
    r"(关系|朋友|同事|共同|属于|参加|参与|关联|联系|"
    r"和.*什么关系|和.*认识|和.*一起|的.*有哪些|"
    r"friend|colleague|relationship|belong|join|participate|associate)",
    re.IGNORECASE,
)


@dataclass
class QueryFeatures:
    """Container for extracted query features."""

    normalized_query: str
    query_embedding: np.ndarray
    entities: List[str]
    temporal_hints: List[str]
    relation_hints: List[str]


class QueryFeatureBuilder:
    """Build lightweight features from a normalized user query."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        normalizer: QueryNormalizer | None = None,
    ) -> None:
        self._embedder = embedder
        self._normalizer = normalizer or QueryNormalizer()
        self._embedding_cache: dict[str, np.ndarray] = {}
        logger.info(
            "QueryFeatureBuilder initialized with embedder=%s",
            embedder.__class__.__name__,
        )

    def build(
        self,
        raw_query: str,
        normalized_query: str | None = None,
    ) -> QueryFeatures:
        """Extract all features from a raw user query.

        Args:
            raw_query: Original user query string.
            normalized_query: Optional pre-normalized query. If provided, it is
                used for embedding instead of re-normalizing raw_query. Hints
                are still extracted from raw_query to preserve casing.

        Returns:
            QueryFeatures containing normalized text, embedding, and weak hints.
        """
        logger.debug("Building features for query: %s", raw_query)

        if normalized_query is not None:
            normalized = normalized_query
        else:
            normalized = self._normalizer.normalize(raw_query)

        # Embedding (with cache)
        cached = self._embedding_cache.get(normalized)
        if cached is not None:
            embedding = cached
            logger.debug("Cache hit for normalized query")
        else:
            embedding = self._embedder.embed([normalized])[0]
            self._embedding_cache[normalized] = embedding
            logger.debug("Generated embedding shape=%s", embedding.shape)

        # Lightweight entities (extracted from raw query to preserve casing)
        entities = self._extract_entities(raw_query)
        logger.debug("Extracted entities: %s", entities)

        # Lightweight temporal hints
        temporal_hints = self._extract_temporal_hints(raw_query)
        logger.debug("Extracted temporal hints: %s", temporal_hints)

        # Lightweight relation hints
        relation_hints = self._extract_relation_hints(raw_query)
        logger.debug("Extracted relation hints: %s", relation_hints)

        return QueryFeatures(
            normalized_query=normalized,
            query_embedding=embedding,
            entities=entities,
            temporal_hints=temporal_hints,
            relation_hints=relation_hints,
        )

    @staticmethod
    def _extract_entities(text: str) -> List[str]:
        """Extract candidate entity mentions from text."""
        seen: set[str] = set()
        results: List[str] = []
        for pattern in _ENTITY_PATTERNS:
            for match in pattern.finditer(text):
                entity = match.group(0)
                if entity not in seen:
                    seen.add(entity)
                    results.append(entity)
        return results

    @staticmethod
    def _extract_relation_hints(text: str) -> List[str]:
        """Extract relation expression hints from text."""
        matches = _RELATION_PATTERNS.findall(text)
        # findall returns tuples when groups exist; flatten them
        flat: List[str] = []
        for m in matches:
            if isinstance(m, tuple):
                flat.extend(part for part in m if part)
            else:
                flat.append(m)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for item in flat:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    @staticmethod
    def _extract_temporal_hints(text: str) -> List[str]:
        """Extract temporal expression hints from text."""
        matches = _TEMPORAL_PATTERNS.findall(text)
        # findall returns tuples when groups exist; flatten them
        flat: List[str] = []
        for m in matches:
            if isinstance(m, tuple):
                flat.extend(part for part in m if part)
            else:
                flat.append(m)
        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: List[str] = []
        for item in flat:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique
