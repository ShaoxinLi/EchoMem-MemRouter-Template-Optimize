"""Tests for embedding providers."""

from types import SimpleNamespace

import pytest

from echomem.embeddings.base import OpenAIEmbeddingProvider


class _FakeEmbeddingItem:
    def __init__(self, dim: int) -> None:
        self.embedding = [0.1] * dim


class _FakeEmbeddingResponse:
    def __init__(self, dim: int) -> None:
        self.data = [_FakeEmbeddingItem(dim)]


class _FakeEmbeddingsClient:
    def __init__(self, sink: dict[str, object], dim: int) -> None:
        self._sink = sink
        self._dim = dim

    def create(self, **kwargs: object) -> _FakeEmbeddingResponse:
        self._sink.clear()
        self._sink.update(kwargs)
        return _FakeEmbeddingResponse(self._dim)


class _FakeOpenAIClient:
    def __init__(self, sink: dict[str, object], dim: int) -> None:
        self.embeddings = _FakeEmbeddingsClient(sink, dim)


def _install_fake_openai(monkeypatch: pytest.MonkeyPatch, sink: dict[str, object], dim: int) -> None:
    fake_module = SimpleNamespace(OpenAI=lambda **_: _FakeOpenAIClient(sink, dim))
    monkeypatch.setitem(__import__("sys").modules, "openai", fake_module)


def test_openai_ada_does_not_send_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: dict[str, object] = {}
    _install_fake_openai(monkeypatch, sink, dim=1536)

    provider = OpenAIEmbeddingProvider(model="text-embedding-ada-002")
    provider.embed(["hello"])

    assert sink["model"] == "text-embedding-ada-002"
    assert "dimensions" not in sink


def test_openai_explicit_output_dimension_is_sent(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: dict[str, object] = {}
    _install_fake_openai(monkeypatch, sink, dim=1024)

    provider = OpenAIEmbeddingProvider(model="text-embedding-3-large", output_dimension=1024)
    provider.embed(["hello"])

    assert sink["model"] == "text-embedding-3-large"
    assert sink["dimensions"] == 1024
    assert provider.dimension() == 1024


def test_openai_ada_rejects_explicit_output_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    sink: dict[str, object] = {}
    _install_fake_openai(monkeypatch, sink, dim=1536)

    with pytest.raises(ValueError, match="does not support"):
        OpenAIEmbeddingProvider(model="text-embedding-ada-002", output_dimension=1024)
