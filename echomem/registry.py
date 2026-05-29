"""MemoryBackendRegistry — the canonical catalog of available memory backends.

All routing decisions, template targets, and adapter dispatch must reference
backends registered here. This is the single source of truth for what backends
exist and whether they are enabled.
"""

import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryContract(BaseModel):
    """Contract describing what query shapes a backend accepts."""

    input_format: str = "natural_language_with_hints"
    supports_entities: bool = True
    supports_time_range: bool = True
    supports_relation_hints: bool = False


class CostProfile(BaseModel):
    """Latency and token cost profile for observability and future scheduling."""

    latency_class: str = "medium"
    token_cost_class: str = "medium"


class BackendEntry(BaseModel):
    """A single registered memory backend."""

    backend_id: str
    backend_kind: str
    status: str = "enabled"  # "enabled" or "disabled"
    description: str = ""
    # capabilities is adapter-layer metadata only; MemRouter does NOT route by capability.
    capabilities: List[str] = Field(default_factory=list)
    query_contract: QueryContract = Field(default_factory=QueryContract)
    cost_profile: CostProfile = Field(default_factory=CostProfile)


class MemoryBackendRegistry:
    """In-memory registry of memory backends.

    Thread-safe for read-heavy workloads (typical routing pattern).
    Mutations (register/unregister) should happen at initialization time.
    """

    def __init__(self) -> None:
        self._backends: Dict[str, BackendEntry] = {}
        logger.debug("MemoryBackendRegistry initialized")

    def register(self, entry: BackendEntry) -> None:
        """Register or update a backend entry."""
        logger.info(
            "Registering backend: id=%s kind=%s status=%s",
            entry.backend_id,
            entry.backend_kind,
            entry.status,
        )
        self._backends[entry.backend_id] = entry

    def get(self, backend_id: str) -> Optional[BackendEntry]:
        """Retrieve a backend entry by ID."""
        return self._backends.get(backend_id)

    def list_enabled(self) -> List[BackendEntry]:
        """Return all backends with status == 'enabled'."""
        return [b for b in self._backends.values() if b.status == "enabled"]

    def list_all(self) -> List[BackendEntry]:
        """Return all registered backends regardless of status."""
        return list(self._backends.values())

    def is_enabled(self, backend_id: str) -> bool:
        """Check whether a backend is registered and enabled."""
        entry = self._backends.get(backend_id)
        return entry is not None and entry.status == "enabled"

    def enabled_backend_ids(self) -> List[str]:
        """Return IDs of all enabled backends."""
        return [b.backend_id for b in self.list_enabled()]

    def __len__(self) -> int:
        return len(self._backends)
