"""Shared fixtures for EchoMem tests."""

import pytest

from echomem.embeddings.base import MockEmbeddingProvider


@pytest.fixture
def mock_embedder() -> MockEmbeddingProvider:
    return MockEmbeddingProvider(dim=16)
