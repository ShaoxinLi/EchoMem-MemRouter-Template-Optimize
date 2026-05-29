"""Tests for BackendRouteTemplateIndex."""

from pathlib import Path

from echomem.templates import BackendRouteTemplateIndex


def test_load_builtin_templates() -> None:
    index = BackendRouteTemplateIndex()
    builtin_dir = Path(__file__).parent.parent / "echomem" / "templates_data"
    count = index.load_from_directory(builtin_dir)

    assert count >= 1
    # Check a known enabled v2 template
    template = index.get("openviking.personal_fact_lookup.en.v2")
    assert template is not None
    assert template.status == "enabled"
    assert template.target.primary_backend_id == "openviking_memory_backend"
    assert len(template.query_prototypes) > 0
    assert len(template.hard_negatives) > 0
