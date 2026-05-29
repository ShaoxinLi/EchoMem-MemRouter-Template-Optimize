"""Tests for .yml template loading."""

import tempfile
from pathlib import Path

from echomem.templates import BackendRouteTemplateIndex


def test_load_yml_extension() -> None:
    """BackendRouteTemplateIndex should load both .yaml and .yml files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        # Write a .yml file
        yml_content = """
schema_version: "mem-router.backend-route-template.v1"
template_id: "test.yml.template.v1"
version: "1.0"
status: "enabled"
target:
  primary_backend_id: "openviking_memory_backend"
  secondary_backend_ids: []
intent_family:
  name: "yml_test"
  description: "Testing yml loading"
semantic_card: ""
query_prototypes:
  - "test prototype"
hard_negatives: []
thresholds:
  accept: 0.7
  fallback: 0.5
  margin: 0.04
  hard_negative_margin: 0.03
  hard_negative_penalty: 0.06
calibration:
  min_positive_examples: 1
  min_hard_negatives: 1
  expected_fallback_rate: 0.1
"""
        (tmp_path / "test.yml").write_text(yml_content, encoding="utf-8")

        index = BackendRouteTemplateIndex()
        count = index.load_from_directory(tmp_path)

        assert count == 1
        assert index.get("test.yml.template.v1") is not None
