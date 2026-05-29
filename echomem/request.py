"""Input contract for MemRouter.

MemoryRouteRequest is the canonical entry point. Callers that have already
decided a query may need memory must construct this request explicitly.
MemRouter does not judge whether memory access is needed; it only routes
queries that have reached the memory retrieval stage.
"""

from typing import Optional

from pydantic import BaseModel, Field


class CallerContext(BaseModel):
    """Context provided by the caller about why MemRouter is being invoked."""

    caller: str = "unknown"
    reason: str = ""
    priority: str = "normal"  # normal / recall_first / latency_first
    user_id: str = ""
    session_id: str = ""


class MemoryRouteRequest(BaseModel):
    """Standardized input to the MemRouter routing pipeline.

    Fields are intentionally minimal so that the routing layer can focus on
    backend selection rather than parsing arbitrary user text.
    """

    schema_version: str = "mem-router.memory-route-request.v1"
    raw_user_query: str
    normalized_user_query: Optional[str] = None
    caller_context: CallerContext = Field(default_factory=CallerContext)
