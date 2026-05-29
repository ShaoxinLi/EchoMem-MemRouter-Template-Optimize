"""Tests for MemoryBackendRegistry."""

from echomem.registry import BackendEntry, MemoryBackendRegistry


def test_register_and_get() -> None:
    reg = MemoryBackendRegistry()
    entry = BackendEntry(backend_id="ov", backend_kind="openviking_native")
    reg.register(entry)

    assert reg.get("ov") == entry
    assert reg.is_enabled("ov")
    assert reg.enabled_backend_ids() == ["ov"]


def test_disabled_backend() -> None:
    reg = MemoryBackendRegistry()
    reg.register(BackendEntry(backend_id="ov", backend_kind="openviking_native", status="disabled"))

    assert not reg.is_enabled("ov")
    assert reg.list_enabled() == []
    assert len(reg.list_all()) == 1
