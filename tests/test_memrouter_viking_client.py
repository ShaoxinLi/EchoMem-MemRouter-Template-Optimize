"""Unit tests for MemRouterVikingClient fallback behaviour."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from echomem.agent_sdk.memrouter_viking_client import MemRouterVikingClient


def _make_instruction():
    """Return a minimal mock BackendQueryInstruction."""
    inst = MagicMock()
    inst.model_dump.return_value = {
        "query": "test query",
        "target_uri": "",
        "backend_id": "openviking_memory_backend",
    }
    return inst


class FakeHTTPClient:
    """Mock HTTP client whose execute_instruction raises a given exception."""

    def __init__(self, exc=None):
        self._exc = exc

    async def execute_instruction(self, inst_dict):
        if self._exc is not None:
            raise self._exc
        return MagicMock(memories=[], resources=[], skills=[], total=0)


class FakeClientWithoutExecuteInstruction:
    """Mock client that lacks execute_instruction (simulates old OpenViking)."""
    pass


class HTTPError(Exception):
    """Fake HTTP-like exception with status_code attribute."""
    def __init__(self, status_code, message=""):
        self.status_code = status_code
        super().__init__(message)


@pytest.mark.asyncio
async def test_fallback_on_attribute_error():
    """Missing execute_instruction should trigger fallback."""
    viking = MagicMock()
    viking.client = FakeClientWithoutExecuteInstruction()
    viking.find = AsyncMock(return_value=MagicMock(memories=[], resources=[], skills=[], total=0))

    client = MemRouterVikingClient(viking_client=viking)
    client._initialized = True
    client._pipeline = MagicMock()

    result = await client._execute_fast(_make_instruction())
    assert viking.find.called
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_fallback_on_404():
    """HTTP 404 (endpoint missing) should trigger fallback."""
    viking = MagicMock()
    viking.client = FakeHTTPClient(exc=HTTPError(404, "Not Found"))
    viking.find = AsyncMock(return_value=MagicMock(memories=[], resources=[], skills=[], total=0))

    client = MemRouterVikingClient(viking_client=viking)
    client._initialized = True
    client._pipeline = MagicMock()

    result = await client._execute_fast(_make_instruction())
    assert viking.find.called
    assert result["total"] == 0


@pytest.mark.asyncio
async def test_fallback_on_405():
    """HTTP 405 (method not allowed) should trigger fallback."""
    viking = MagicMock()
    viking.client = FakeHTTPClient(exc=HTTPError(405, "Method Not Allowed"))
    viking.find = AsyncMock(return_value=MagicMock(memories=[], resources=[], skills=[], total=0))

    client = MemRouterVikingClient(viking_client=viking)
    client._initialized = True
    client._pipeline = MagicMock()

    result = await client._execute_fast(_make_instruction())
    assert viking.find.called


@pytest.mark.asyncio
async def test_fallback_on_501():
    """HTTP 501 (not implemented) should trigger fallback."""
    viking = MagicMock()
    viking.client = FakeHTTPClient(exc=HTTPError(501, "Not Implemented"))
    viking.find = AsyncMock(return_value=MagicMock(memories=[], resources=[], skills=[], total=0))

    client = MemRouterVikingClient(viking_client=viking)
    client._initialized = True
    client._pipeline = MagicMock()

    result = await client._execute_fast(_make_instruction())
    assert viking.find.called


@pytest.mark.asyncio
async def test_no_fallback_on_400():
    """HTTP 400 (bad request) should NOT trigger fallback - real protocol error."""
    viking = MagicMock()
    viking.client = FakeHTTPClient(exc=HTTPError(400, "Bad Request"))

    client = MemRouterVikingClient(viking_client=viking)
    client._initialized = True
    client._pipeline = MagicMock()

    with pytest.raises(HTTPError):
        await client._execute_fast(_make_instruction())


@pytest.mark.asyncio
async def test_no_fallback_on_401():
    """HTTP 401 (unauthenticated) should NOT trigger fallback."""
    viking = MagicMock()
    viking.client = FakeHTTPClient(exc=HTTPError(401, "Unauthorized"))

    client = MemRouterVikingClient(viking_client=viking)
    client._initialized = True
    client._pipeline = MagicMock()

    with pytest.raises(HTTPError):
        await client._execute_fast(_make_instruction())


@pytest.mark.asyncio
async def test_no_fallback_on_403():
    """HTTP 403 (permission denied) should NOT trigger fallback."""
    viking = MagicMock()
    viking.client = FakeHTTPClient(exc=HTTPError(403, "Forbidden"))

    client = MemRouterVikingClient(viking_client=viking)
    client._initialized = True
    client._pipeline = MagicMock()

    with pytest.raises(HTTPError):
        await client._execute_fast(_make_instruction())


@pytest.mark.asyncio
async def test_no_fallback_on_422():
    """HTTP 422 (validation error) should NOT trigger fallback."""
    viking = MagicMock()
    viking.client = FakeHTTPClient(exc=HTTPError(422, "Unprocessable Entity"))

    client = MemRouterVikingClient(viking_client=viking)
    client._initialized = True
    client._pipeline = MagicMock()

    with pytest.raises(HTTPError):
        await client._execute_fast(_make_instruction())


@pytest.mark.asyncio
async def test_no_fallback_on_500():
    """HTTP 500 (internal error) should NOT trigger fallback."""
    viking = MagicMock()
    viking.client = FakeHTTPClient(exc=HTTPError(500, "Internal Server Error"))

    client = MemRouterVikingClient(viking_client=viking)
    client._initialized = True
    client._pipeline = MagicMock()

    with pytest.raises(HTTPError):
        await client._execute_fast(_make_instruction())


@pytest.mark.asyncio
async def test_no_fallback_on_openviking_error_without_status_code():
    """OpenVikingError without status_code (e.g. internal logic failure) should NOT fallback."""
    class OpenVikingError(Exception):
        pass

    viking = MagicMock()
    viking.client = FakeHTTPClient(exc=OpenVikingError("internal failure"))

    client = MemRouterVikingClient(viking_client=viking)
    client._initialized = True
    client._pipeline = MagicMock()

    with pytest.raises(OpenVikingError):
        await client._execute_fast(_make_instruction())


def test_embedding_config_supports_dense_shape():
    """Local config may use the embedding.dense shape from OpenViking-style config."""
    config = {
        "embedding": {
            "dense": {
                "provider": "openai",
                "model": "text-embedding-v3",
                "dimension": 1024,
            }
        }
    }

    embedding_config = MemRouterVikingClient._get_embedding_config(config)

    assert embedding_config["provider"] == "openai"
    assert embedding_config["model"] == "text-embedding-v3"
    assert embedding_config["dimension"] == 1024


def test_llm_router_config_supports_local_yaml_shape():
    """VikingBot E2E should use real LLM fallback when local config provides it."""
    config = {
        "llm": {
            "provider": "anthropic_compatible",
            "model": "MiniMax-M2.7",
            "base_url": "https://example.invalid/anthropic",
            "auth_token": "test-key",
            "timeout_ms": 120000,
            "max_tokens": 1024,
            "temperature": 0.0,
        }
    }

    llm_config = MemRouterVikingClient._build_llm_router_config(config)

    assert llm_config.provider == "anthropic_compatible"
    assert llm_config.model == "MiniMax-M2.7"
    assert llm_config.api_key == "test-key"
    assert llm_config.base_url == "https://example.invalid/anthropic"
    assert llm_config.timeout_seconds == 120


def test_build_memrouter_meta_contains_backend_route_and_instruction():
    """Tool output metadata should be sufficient for E2E route metric parsing."""
    route = MagicMock()
    route.backend_id = "openviking_memory_backend"
    route.backend_kind = "openviking_native"
    route.role = "primary"
    route.confidence = 0.91
    route.matched_template_id = "openviking.preference_profile.v1"

    fallback = MagicMock()
    fallback.used = False
    fallback.reason = ""

    result = MagicMock()
    result.route_method = "template_embedding"
    result.routes = [route]
    result.fallback = fallback
    result.query_instructions = [_make_instruction()]

    meta = MemRouterVikingClient._build_memrouter_meta(
        route_result=result,
        instruction=result.query_instructions[0],
    )

    assert meta["backend_id"] == "openviking_memory_backend"
    assert meta["route_method"] == "template_embedding"
    assert meta["matched_template_id"] == "openviking.preference_profile.v1"
    assert meta["query_instruction_count"] == 1
    assert meta["has_executable_instruction"] is True
    assert meta["routes"][0]["confidence"] == 0.91


def test_build_default_pipeline_prefers_config_file_over_env(tmp_path, monkeypatch):
    """Hardcoded local config should take priority over environment defaults."""
    config_path = tmp_path / "memrouter.local.yaml"
    config_path.write_text(
        "\n".join(
            [
                "embedding:",
                "  provider: mock",
                "  dimension: 7",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMROUTER_CONFIG", str(config_path))
    monkeypatch.setenv("MEMROUTER_EMBEDDING_PROVIDER", "sentence-transformers")

    pipeline = MemRouterVikingClient._build_default_pipeline()

    embedder = pipeline._feature_builder._embedder
    assert embedder.__class__.__name__ == "MockEmbeddingProvider"
    assert embedder.dimension() == 7


def test_explicit_missing_config_fails_fast(monkeypatch):
    """A typo in MEMROUTER_CONFIG should fail clearly instead of silently falling back."""
    monkeypatch.setenv("MEMROUTER_CONFIG", r"D:\missing\memrouter.local.yaml")

    with pytest.raises(FileNotFoundError):
        MemRouterVikingClient._load_local_config()
