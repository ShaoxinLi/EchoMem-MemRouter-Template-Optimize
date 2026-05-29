"""Core data models for MemRouter route results and shared schemas.

All Pydantic models in this module are kept aligned with the v1.4 design doc
so that future multi-backend expansion does not require schema rewrites.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from echomem.query_spec import BackendQueryInstruction


class QueryHints(BaseModel):
    """Generic query hints passed to backend adapters.

    Adapters decide which hints to use; MemRouter only produces them.
    """

    entities: List[str] = Field(default_factory=list)
    temporal_hints: List[str] = Field(default_factory=list)
    relation_hints: List[str] = Field(default_factory=list)
    semantic_hints: List[str] = Field(default_factory=list)
    answer_shape: str = ""  # e.g. "fact", "list", "summary"


class RouteEntry(BaseModel):
    """A single backend route entry."""

    backend_id: str
    backend_kind: str
    role: str  # "primary" or "secondary"
    confidence: float
    matched_template_id: Optional[str] = None
    query_hints: QueryHints = Field(default_factory=QueryHints)
    uncertainty: Optional[Dict[str, Any]] = None


class FallbackInfo(BaseModel):
    """Fallback metadata."""

    used: bool = False
    type: str = ""
    reason: str = ""


class DebugInfo(BaseModel):
    """Debug information for evaluation and replay."""

    top_templates: List[Dict[str, Any]] = Field(default_factory=list)
    llm_fallback_meta: Optional[Dict[str, Any]] = None


class MemBackendRouteResult(BaseModel):
    """Standardized output of MemRouter.

    This schema is intentionally generic so that v1.4 (single-backend)
    and future multi-backend versions can share the same result structure.
    """

    schema_version: str = "mem-router.backend-route-result.v2"
    raw_user_query: str
    normalized_user_query: str
    route_method: str
    routes: List[RouteEntry] = Field(default_factory=list)
    post_retrieval_requirements: Dict[str, bool] = Field(default_factory=dict)
    fallback: FallbackInfo = Field(default_factory=FallbackInfo)
    debug: DebugInfo = Field(default_factory=DebugInfo)
    # NEW: concrete backend query instructions generated from template query_spec
    query_instructions: List[BackendQueryInstruction] = Field(default_factory=list)
