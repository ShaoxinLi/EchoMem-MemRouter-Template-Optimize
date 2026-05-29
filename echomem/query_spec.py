"""Query specification models for MemRouter backend query instructions.

Implements the Template-First Hybrid (Option 3) design: templates declare
query_spec as the primary source of truth; adapters enrich at runtime.

All models are Pydantic BaseModel for YAML deserialization and validation.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TypedQueryTemplate(BaseModel):
    """Lightweight TypedQuery for direct retrieval, inspired by OpenViking."""

    query: Optional[str] = None
    context_type: Optional[str] = None  # "memory" | "resource" | "skill"
    intent: str = ""
    priority: int = 1
    target_directories: Optional[List[str]] = None


class OpenVikingQuerySpec(BaseModel):
    """OpenViking-specific query parameters embedded in a template."""

    context_type: str = "memory"
    target_uri: str = "viking://memories"
    # Deprecated execution field for OpenViking v0.3.12 integration.
    # Keep it optional for backward-compatible YAML loading; fast path should not rely on it.
    level: Optional[List[int]] = None
    filter: Optional[Dict[str, Any]] = None
    skip_intent_analysis: bool = False
    typed_query_template: Optional[TypedQueryTemplate] = None


class GraphQuerySpec(BaseModel):
    """Graph-memory-specific query parameters embedded in a template."""

    graph_query_type: str = "entity_lookup"  # entity_relation | causal_multihop | system_dependency
    hop_depth: int = 1
    root_entity_hint: Optional[str] = None
    traversal_direction: str = "both"  # out | in | both


class TemporalQuerySpec(BaseModel):
    """Temporal-memory-specific query parameters embedded in a template."""

    temporal_query_type: str = "timeline_fact"  # timeline_fact | duration_comparison | sequence_reasoning
    time_granularity: str = "day"
    time_range_hint: Optional[str] = None


class TemplateQuerySpec(BaseModel):
    """Template-level query specification — produced when a template matches.

    This is the primary source of truth for query instructions under the
    Template-First Hybrid design. Adapters may enrich these values at runtime.
    """

    search_mode: str = "find"  # find | search | read | graph_traversal | temporal_query
    query_rewrite: Optional[str] = None  # supports variable interpolation, e.g. "用户 {entities} 相关个人事实"

    # Backend-specific specs (at most one is typically populated per template)
    openviking: Optional[OpenVikingQuerySpec] = None
    graph: Optional[GraphQuerySpec] = None
    temporal: Optional[TemporalQuerySpec] = None


class BackendQueryInstruction(BaseModel):
    """Standardized backend query instruction that a memory backend can execute directly.

    Produced by QueryInstructionBuilder (template spec + adapter enrichment).
    """

    backend_id: str
    query: str  # rewritten or raw query text
    search_mode: str = "find"  # find | search | graph_traversal | temporal_query
    target_uri: Optional[str] = None
    context_type: Optional[str] = None
    level: Optional[List[int]] = None
    filter: Optional[Dict[str, Any]] = None
    priority: int = 1
    # Backend-specific params forwarded in extra_params
    extra_params: Dict[str, Any] = Field(default_factory=dict)
    # Whether the backend should skip its own intent analysis (e.g. OpenViking IntentAnalyzer)
    skip_intent_analysis: bool = False
    # Pre-constructed TypedQuery for backends that accept it directly
    typed_query: Optional[TypedQueryTemplate] = None
