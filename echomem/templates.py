"""MemoryBackendRouteTemplate definitions and index management.

Templates are the core semantic anchors for backend routing. Each template
contains multiple prototype queries and hard negatives; the TemplateMatcher
computes similarity between the user query and these prototypes.

Schema is kept fully aligned with v1.3/v1.4 design docs so that future
multi-backend expansion does not require migration.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator

from echomem.query_spec import TemplateQuerySpec

logger = logging.getLogger(__name__)


class TemplateTarget(BaseModel):
    """Routing target for a template."""

    primary_backend_id: str
    secondary_backend_ids: List[str] = Field(default_factory=list)


class IntentFamily(BaseModel):
    """Intent cluster metadata for evaluation and maintenance."""

    name: str
    description: str = ""


class HardNegative(BaseModel):
    """A counter-example query that should NOT match this template."""

    query: str
    confusing_with_backend: str
    reason: str = ""


class Thresholds(BaseModel):
    """Scoring thresholds for RouteDecision."""

    accept: float
    fallback: float
    margin: float
    hard_negative_margin: float
    hard_negative_penalty: float

    @field_validator("accept", "fallback", "margin", "hard_negative_margin", "hard_negative_penalty")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("threshold values must be non-negative")
        return v

    @field_validator("fallback")
    @classmethod
    def _fallback_not_above_accept(cls, v: float, info) -> float:
        accept = info.data.get("accept")
        if accept is not None and v > accept:
            raise ValueError(f"fallback ({v}) must not exceed accept ({accept})")
        return v

    @field_validator("accept", "fallback")
    @classmethod
    def _not_above_one(cls, v: float) -> float:
        if v > 1.0:
            raise ValueError("threshold values must not exceed 1.0")
        return v


class Calibration(BaseModel):
    """Offline calibration metadata (not used in online path)."""

    min_positive_examples: int = 0
    min_hard_negatives: int = 0
    expected_fallback_rate: float = 0.0


class MemoryBackendRouteTemplate(BaseModel):
    """A single backend route template.

    This is the central data structure shared between v1.3 (multi-backend)
    and v1.4 (three logical backends). The schema is backend-agnostic;
    templates may target any registered backend.
    """

    schema_version: str = "mem-router.backend-route-template.v1"
    template_id: str
    version: str = "1.0"
    status: str = "enabled"  # "enabled" or "disabled"

    target: TemplateTarget
    intent_family: IntentFamily
    semantic_card: str = ""
    query_prototypes: List[str] = Field(default_factory=list)
    hard_negatives: List[HardNegative] = Field(default_factory=list)
    thresholds: Thresholds
    calibration: Calibration = Field(default_factory=Calibration)
    query_spec: Optional[TemplateQuerySpec] = None  # NEW: backend query specification


class BackendRouteTemplateIndex:
    """In-memory index of route templates.

    Responsible for loading templates from YAML files and providing
    iterable access to enabled templates.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, MemoryBackendRouteTemplate] = {}
        logger.debug("BackendRouteTemplateIndex initialized")

    def load_from_directory(self, directory: Path) -> int:
        """Load all YAML template files from a directory.

        Args:
            directory: Path to directory containing .yaml or .yml files.

        Returns:
            Number of templates loaded.
        """
        if not directory.exists():
            logger.warning("Template directory does not exist: %s", directory)
            return 0

        count = 0
        yaml_paths = sorted(set(directory.glob("*.yaml")) | set(directory.glob("*.yml")))
        for path in yaml_paths:
            try:
                template = self._load_yaml(path)
                self._templates[template.template_id] = template
                count += 1
                logger.info("Loaded template: %s from %s", template.template_id, path.name)
            except Exception:
                logger.exception("Failed to load template from %s", path)
        logger.info("Total templates loaded from %s: %d", directory, count)
        return count

    def add(self, template: MemoryBackendRouteTemplate) -> None:
        """Add a template directly (useful for programmatic construction)."""
        self._templates[template.template_id] = template
        logger.debug("Added template: %s", template.template_id)

    def get(self, template_id: str) -> Optional[MemoryBackendRouteTemplate]:
        """Retrieve a template by ID."""
        return self._templates.get(template_id)

    def enabled_templates(self) -> List[MemoryBackendRouteTemplate]:
        """Return all templates with status == 'enabled'."""
        return [t for t in self._templates.values() if t.status == "enabled"]

    def __len__(self) -> int:
        return len(self._templates)

    @staticmethod
    def _load_yaml(path: Path) -> MemoryBackendRouteTemplate:
        """Parse a single YAML file into a MemoryBackendRouteTemplate."""
        with path.open("r", encoding="utf-8") as f:
            data: Dict[str, Any] = yaml.safe_load(f)
        return MemoryBackendRouteTemplate.model_validate(data)
