"""MemRouterVikingClient - drop-in enhancement layer for VikingClient.

Wraps an existing VikingClient (or any object exposing the same interface)
so that MemRouter can intercept ``search()`` calls and route them through the
fast path (``skip_intent_analysis``) when template matching is confident.

All write operations (``commit``, ``add_resource``, etc.) are passed through
unchanged - MemRouter is a read-only routing layer.

Usage (VikingBot / Claw - one-line change)::

    # Before
    client = VikingClient(agent_id="shared")

    # After
    client = MemRouterVikingClient(viking_client=VikingClient(agent_id="shared"))
    await client.initialize()

    # All other code stays the same
    result = await client.search("你还记得我喜欢什么颜色吗")
    await client.commit(session_id="xxx", messages=[...])
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

from echomem.pipeline import MemRouterPipeline
from echomem.result import MemBackendRouteResult

logger = logging.getLogger(__name__)


class MemRouterVikingClient:
    """MemRouter-enhanced wrapper around VikingClient.

    Exposes the **same public interface** as ``VikingClient`` so that existing
    consumers (VikingBot, Claw, higo, etc.) require zero code changes.
    """

    def __init__(
        self,
        viking_client: Any,
        memrouter_pipeline: Optional[MemRouterPipeline] = None,
    ) -> None:
        self._viking = viking_client
        self._pipeline = memrouter_pipeline
        self._last_route_result: Optional[MemBackendRouteResult] = None
        self._initialized = False

        # Detect whether we received a raw AsyncHTTPClient or VikingClient wrapper
        self._is_raw_http = (
            hasattr(viking_client, "execute_instruction")
            and not hasattr(viking_client, "client")
        )
        self._http_client = (
            viking_client if self._is_raw_http else getattr(viking_client, "client", None)
        )

        # Route events JSONL for E2E observability
        self._route_events_path: Optional[Path] = None
        self._route_events_file: Optional[TextIO] = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    async def initialize(self) -> None:
        """Initialize the underlying VikingClient and MemRouter pipeline."""
        # Initialize VikingClient if it has async init hooks
        if hasattr(self._viking, "initialize"):
            await self._viking.initialize()
        elif hasattr(self._viking, "_initialize"):
            await self._viking._initialize()

        # Build default pipeline if caller did not supply one
        if self._pipeline is None:
            self._pipeline = self._build_default_pipeline()

        self._initialized = True
        logger.info("MemRouterVikingClient initialized (pipeline=%s)", self._pipeline)

    @property
    def last_route_result(self) -> Optional[MemBackendRouteResult]:
        """Return the most recent MemRouter route result for E2E observability."""
        return self._last_route_result

    async def close(self) -> None:
        """Close the underlying VikingClient."""
        if hasattr(self._viking, "close"):
            await self._viking.close()
        self._initialized = False
        if self._route_events_file is not None:
            self._route_events_file.close()
            self._route_events_file = None

    def _init_route_events(self) -> None:
        """Open the route events JSONL file for append.

        Path priority:
        1. ``MEMROUTER_ROUTE_EVENTS`` environment variable.
        2. ``<MEMROUTER_RUNS_DIR>/route_events.jsonl``.
        3. ``<EchoMem repo>/runs/memrouter_viking_client/route_events.jsonl``.
        """
        if self._route_events_path is not None:
            return

        explicit = os.environ.get("MEMROUTER_ROUTE_EVENTS")
        repo_root = Path(__file__).resolve().parents[2]
        runs_dir = None
        if explicit:
            self._route_events_path = Path(explicit).expanduser()
        else:
            runs_dir = os.environ.get("MEMROUTER_RUNS_DIR")
            if runs_dir:
                self._route_events_path = Path(runs_dir) / "route_events.jsonl"
            else:
                self._route_events_path = (
                    repo_root / "runs" / "memrouter_viking_client" / "route_events.jsonl"
                )

        self._route_events_path.parent.mkdir(parents=True, exist_ok=True)
        self._route_events_file = open(
            self._route_events_path, "a", encoding="utf-8"
        )
        logger.info("MemRouter route events: %s", self._route_events_path)

    def _append_route_event(self, event: Dict[str, Any]) -> None:
        """Append a single route event to the JSONL file."""
        if self._route_events_file is None:
            return
        try:
            self._route_events_file.write(json.dumps(event, ensure_ascii=False) + "\n")
            self._route_events_file.flush()
        except Exception as exc:
            logger.warning("Failed to write route event: %s", exc)

    @staticmethod
    def _build_route_event(
        query: str,
        route_result: Optional[MemBackendRouteResult] = None,
        instruction: Optional[Any] = None,
        execution_path: str = "unknown",
        latency_ms: int = 0,
        error: str = "",
    ) -> Dict[str, Any]:
        """Build a route event dict for JSONL observability."""
        from datetime import datetime, timezone

        primary_route = route_result.routes[0] if route_result and route_result.routes else None
        routes_meta: List[Dict[str, Any]] = []
        if route_result is not None:
            routes_meta = [
                {
                    "backend_id": r.backend_id,
                    "backend_kind": r.backend_kind,
                    "role": r.role,
                    "confidence": r.confidence,
                    "matched_template_id": r.matched_template_id,
                }
                for r in route_result.routes
            ]

        inst_dict: Optional[Dict[str, Any]] = None
        if instruction is not None:
            try:
                inst_dict = instruction.model_dump()
            except Exception:
                inst_dict = {"backend_id": getattr(instruction, "backend_id", "unknown")}

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "route_method": route_result.route_method if route_result else "unknown",
            "backend_id": primary_route.backend_id if primary_route else "",
            "matched_template_id": primary_route.matched_template_id if primary_route else "",
            "confidence": primary_route.confidence if primary_route else None,
            "routes": routes_meta,
            "fallback_used": bool(route_result.fallback.used) if route_result else False,
            "fallback_reason": route_result.fallback.reason if route_result else "",
            "execution_path": execution_path,
            "skip_intent_analysis": bool(instruction.skip_intent_analysis) if instruction else False,
            "search_mode": getattr(instruction, "search_mode", "") if instruction else "",
            "instruction": inst_dict,
            "latency_ms": latency_ms,
            "error": error,
        }

    async def __aenter__(self) -> MemRouterVikingClient:
        await self.initialize()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------ #
    # Enhanced search - the only method MemRouter intercepts
    # ------------------------------------------------------------------ #

    async def search(
        self,
        query: str,
        target_uri: Optional[str] = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Semantic search with MemRouter fast-path optimisation.

        Flow:
            1. MemRouter ``route()`` → ``BackendQueryInstruction``
            2. If ``skip_intent_analysis=True`` → call OpenViking
               ``execute_instruction()`` (zero VLM tokens).
            3. Otherwise → fall back to the native ``VikingClient.search()``.
        """
        if not self._initialized:
            raise RuntimeError(
                "MemRouterVikingClient not initialized. Call initialize() first."
            )

        self._init_route_events()
        started = time.perf_counter()
        execution_path = "unknown"
        error = ""
        result: Dict[str, Any] = {}

        route_result = self._pipeline.route(query)
        self._last_route_result = route_result
        instructions = route_result.query_instructions

        if not instructions:
            logger.debug(
                "MemRouter produced no instructions for query='%s'; "
                "falling back to VikingClient.search",
                query,
            )
            execution_path = "no_instructions_fallback"
            try:
                if self._is_raw_http:
                    result = await self._viking.search(
                        query, target_uri=target_uri or "", limit=limit, **kwargs
                    )
                else:
                    result = await self._viking.search(query, target_uri=target_uri)
                result = self._normalize_search_result(
                    result,
                    query=query,
                    target_uri=target_uri or "",
                    route_result=route_result,
                )
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                self._append_route_event(
                    self._build_route_event(
                        query=query,
                        route_result=route_result,
                        execution_path=execution_path,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        error=error,
                    )
                )
            return result

        inst = instructions[0]  # primary backend
        logger.info(
            "MemRouter route: backend=%s template=%s skip_ia=%s search_mode=%s "
            "confidence=%.3f",
            inst.backend_id,
            route_result.routes[0].matched_template_id if route_result.routes else "none",
            inst.skip_intent_analysis,
            inst.search_mode,
            route_result.routes[0].confidence if route_result.routes else 0.0,
        )

        try:
            # Fast path - only for OpenViking when MemRouter is confident
            if inst.backend_id == "openviking_memory_backend" and inst.skip_intent_analysis:
                execution_path = "fast_path"
                result = await self._execute_fast(
                    inst, route_result=route_result, limit=limit
                )
            elif self._is_raw_http:
                execution_path = "native_search_http"
                result = await self._viking.search(
                    query, target_uri=target_uri or "", limit=limit, **kwargs
                )
                result = self._normalize_search_result(
                    result,
                    query=query,
                    target_uri=target_uri or "",
                    route_result=route_result,
                    instruction=inst,
                )
            else:
                execution_path = "native_search"
                result = await self._viking.search(query, target_uri=target_uri)
                result = self._normalize_search_result(
                    result,
                    query=query,
                    target_uri=target_uri or "",
                    route_result=route_result,
                    instruction=inst,
                )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._append_route_event(
                self._build_route_event(
                    query=query,
                    route_result=route_result,
                    instruction=inst,
                    execution_path=execution_path,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    error=error,
                )
            )
        return result

    async def _execute_fast(
        self,
        instruction: "BackendQueryInstruction",
        route_result: Optional[MemBackendRouteResult] = None,
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Call OpenViking ``execute_instruction()`` bypassing IntentAnalyzer."""
        inst_dict = instruction.model_dump()
        inst_dict["limit"] = limit

        try:
            if self._http_client and hasattr(self._http_client, "execute_instruction"):
                result = await self._http_client.execute_instruction(inst_dict)
            else:
                raise AttributeError("No HTTP client with execute_instruction available")
        except Exception as exc:
            # Fallback only when the endpoint is missing or unsupported.
            # Real retrieval errors (500, timeout during search) are re-raised.
            status_code = getattr(exc, "status_code", None)
            is_unsupported = (
                isinstance(exc, AttributeError)
                or status_code in {404, 405, 501}
            )
            if is_unsupported:
                logger.warning(
                    "OpenViking execute_instruction() unavailable (%s: %s); "
                    "falling back to native find/search",
                    type(exc).__name__,
                    exc,
                )
                return await self._fallback_find(
                    query=instruction.query,
                    target_uri=instruction.target_uri or "",
                    limit=limit,
                )
            logger.error(
                "OpenViking execute_instruction() failed (%s: %s); "
                "not falling back to avoid masking real errors",
                type(exc).__name__,
                exc,
            )
            raise

        return self._normalize_search_result(
            result,
            instruction,
            route_result=route_result or self._last_route_result,
        )

    async def _fallback_find(
        self,
        query: str,
        target_uri: str = "",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """Fallback to native find/search when execute_instruction unavailable."""
        if self._is_raw_http:
            result = await self._viking.find(
                query=query, target_uri=target_uri, limit=limit
            )
        else:
            # VikingClient.find only accepts query and target_uri
            result = await self._viking.find(query=query, target_uri=target_uri)
        return self._normalize_search_result(result, query=query, target_uri=target_uri)

    def _normalize_search_result(
        self,
        result: Any,
        instruction: Optional["BackendQueryInstruction"] = None,
        query: Optional[str] = None,
        target_uri: Optional[str] = None,
        route_result: Optional[MemBackendRouteResult] = None,
    ) -> Dict[str, Any]:
        """Convert any FindResult / dict into VikingBot-compatible dict."""
        if isinstance(result, dict):
            # Already normalized (e.g. VikingClient.search returns this)
            base = dict(result)
            if "_memrouter_meta" not in base:
                meta = self._build_memrouter_meta(
                    route_result=route_result,
                    instruction=instruction,
                )
                if meta:
                    base["_memrouter_meta"] = meta
            return base

        query = query or (instruction.query if instruction else "")
        target_uri = target_uri or (instruction.target_uri if instruction else "")
        meta = self._build_memrouter_meta(
            route_result=route_result,
            instruction=instruction,
        )

        return {
            "memories": [
                self._matched_context_to_dict(m)
                for m in getattr(result, "memories", [])
            ],
            "resources": [
                self._matched_context_to_dict(r)
                for r in getattr(result, "resources", [])
            ],
            "skills": [
                self._matched_context_to_dict(s)
                for s in getattr(result, "skills", [])
            ],
            "total": getattr(result, "total", 0),
            "query": query,
            "target_uri": target_uri,
            **({"_memrouter_meta": meta} if meta else {}),
        }

    @staticmethod
    def _build_memrouter_meta(
        route_result: Optional[MemBackendRouteResult] = None,
        instruction: Optional["BackendQueryInstruction"] = None,
    ) -> Dict[str, Any]:
        """Build compact metadata for E2E evaluation and VikingBot tool output."""
        if route_result is None and instruction is None:
            return {}

        routes = []
        if route_result is not None:
            routes = [
                {
                    "backend_id": r.backend_id,
                    "backend_kind": r.backend_kind,
                    "role": r.role,
                    "confidence": r.confidence,
                    "matched_template_id": r.matched_template_id,
                }
                for r in route_result.routes
            ]

        primary = routes[0] if routes else {}
        instruction_count = len(route_result.query_instructions) if route_result else (1 if instruction else 0)
        instruction_payload = instruction.model_dump() if instruction is not None else None

        return {
            "route_method": route_result.route_method if route_result else "unknown",
            "backend_id": primary.get(
                "backend_id",
                instruction.backend_id if instruction else "openviking_memory_backend",
            ),
            "matched_template_id": primary.get("matched_template_id", ""),
            "confidence": primary.get("confidence"),
            "routes": routes,
            "fallback_used": bool(route_result.fallback.used) if route_result else False,
            "fallback_reason": route_result.fallback.reason if route_result else "",
            "query_instruction_count": instruction_count,
            "has_executable_instruction": instruction is not None,
            "skip_intent_analysis": bool(instruction.skip_intent_analysis) if instruction else False,
            "search_mode": instruction.search_mode if instruction else "",
            "instruction": instruction_payload,
        }

    # ------------------------------------------------------------------ #
    # Pass-through methods - MemRouter does not participate in writes
    # ------------------------------------------------------------------ #

    async def find(
        self,
        query: str,
        target_uri: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Pass-through to ``VikingClient.find()``."""
        return await self._viking.find(query, target_uri=target_uri, **kwargs)

    async def commit(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        user_id: Optional[str] = None,
    ) -> Any:
        """Pass-through to ``VikingClient.commit()``."""
        return await self._viking.commit(session_id, messages, user_id=user_id)

    async def read_content(self, uri: str, level: str = "abstract") -> str:
        """Pass-through to ``VikingClient.read_content()``."""
        return await self._viking.read_content(uri, level=level)

    async def add_resource(
        self, local_path: str, desc: str
    ) -> Optional[Dict[str, Any]]:
        """Pass-through to ``VikingClient.add_resource()``."""
        return await self._viking.add_resource(local_path, desc)

    async def list_resources(
        self,
        path: Optional[str] = None,
        recursive: bool = False,
    ) -> List[Dict[str, Any]]:
        """Pass-through to ``VikingClient.list_resources()``."""
        return await self._viking.list_resources(path=path, recursive=recursive)

    async def read_user_profile(self, user_id: str) -> str:
        """Pass-through to ``VikingClient.read_user_profile()``."""
        return await self._viking.read_user_profile(user_id)

    async def search_user_memory(self, query: str, user_id: str) -> List[Any]:
        """Pass-through to ``VikingClient.search_user_memory()``."""
        return await self._viking.search_user_memory(query, user_id)

    async def search_memory(
        self,
        query: str,
        user_ids: Any,
        agent_user_id: str,
        limit: int = 10,
    ) -> Dict[str, List[Any]]:
        """Pass-through to ``VikingClient.search_memory()``."""
        return await self._viking.search_memory(
            query, user_ids, agent_user_id, limit=limit
        )

    async def search_experiences(self, query: str, limit: int = 5) -> List[Any]:
        """Pass-through to ``VikingClient.search_experiences()``."""
        return await self._viking.search_experiences(query, limit=limit)

    async def grep(
        self,
        uri: str,
        pattern: str,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Pass-through to ``VikingClient.grep()``."""
        return await self._viking.grep(uri, pattern, **kwargs)

    async def glob(self, pattern: str, uri: Optional[str] = None) -> Dict[str, Any]:
        """Pass-through to ``VikingClient.glob()``."""
        return await self._viking.glob(pattern, uri=uri)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_default_pipeline() -> MemRouterPipeline:
        """Build a MemRouterPipeline with config-file-first defaults.

        VikingBot runs inside the OpenViking runtime, so optional EchoMem
        dependencies such as sentence-transformers may not always be installed.
        Local configuration is preferred because E2E runs often use hardcoded
        OpenAI-compatible embedding credentials. Environment variables remain
        as a fallback for CI and quick smoke tests.
        """
        from echomem.embeddings.base import (
            create_provider,
            get_openai_model_default_dimension,
        )

        config, config_path = MemRouterVikingClient._load_local_config()
        MemRouterVikingClient._configure_run_logging(config)
        embedding_config = MemRouterVikingClient._get_embedding_config(config)
        llm_router_config = MemRouterVikingClient._build_llm_router_config(config)

        provider_type = (
            embedding_config.get("provider")
            or os.environ.get("MEMROUTER_EMBEDDING_PROVIDER")
            or "sentence-transformers"
        )
        logger.info(
            "Building MemRouter pipeline (embedding_provider=%s, config=%s)",
            provider_type,
            str(config_path) if config_path else "env/default",
        )

        if provider_type == "mock":
            embedder = create_provider(
                "mock",
                dim=int(
                    embedding_config.get("dimension")
                    or embedding_config.get("dim")
                    or os.environ.get("MEMROUTER_EMBEDDING_DIM", "32")
                ),
            )
        elif provider_type == "openai":
            model = (
                embedding_config.get("model")
                or os.environ.get("MEMROUTER_EMBEDDING_MODEL")
                or "text-embedding-v3"
            )
            raw_dim = (
                embedding_config.get("dimension")
                or embedding_config.get("output_dimension")
                or os.environ.get("MEMROUTER_EMBEDDING_DIM")
            )
            output_dimension = None
            if raw_dim:
                requested_dim = int(raw_dim)
                try:
                    default_dim = get_openai_model_default_dimension(model)
                except ValueError:
                    default_dim = None
                # Do not send the OpenAI-compatible "dimensions" parameter
                # when the requested dimension is the model default; some
                # compatible endpoints reject it even though the default output
                # shape is already correct.
                if default_dim is None or requested_dim != default_dim:
                    output_dimension = requested_dim

            max_batch_raw = (
                embedding_config.get("max_batch_size")
                or embedding_config.get("max_concurrent")
                or os.environ.get("MEMROUTER_EMBEDDING_MAX_BATCH")
            )
            embedder = create_provider(
                "openai",
                model=model,
                api_key=(
                    embedding_config.get("api_key")
                    or os.environ.get("MEMROUTER_EMBEDDING_API_KEY")
                ),
                base_url=(
                    embedding_config.get("api_base")
                    or embedding_config.get("base_url")
                    or os.environ.get("MEMROUTER_EMBEDDING_API_BASE")
                ),
                output_dimension=output_dimension,
                max_batch_size=int(max_batch_raw) if max_batch_raw else None,
            )
        else:
            embedder = create_provider(
                "sentence-transformers",
                model_name=(
                    embedding_config.get("model_name")
                    or embedding_config.get("model")
                    or os.environ.get("MEMROUTER_ST_MODEL")
                    or "sentence-transformers/all-MiniLM-L6-v2"
                ),
            )

        return MemRouterPipeline.with_defaults(
            embedder=embedder,
            llm_router_config=llm_router_config,
        )

    @staticmethod
    def _build_llm_router_config(config: Dict[str, Any]):
        """Build LLMRouterConfig from local YAML for real VikingBot E2E fallback.

        If no ``llm`` section is present, MemRouter keeps the mock fallback used
        by CI and lightweight smoke tests.
        """
        from echomem.llm_fallback import LLMRouterConfig

        llm = config.get("llm") if isinstance(config, dict) else None
        if not isinstance(llm, dict):
            return LLMRouterConfig(provider="mock", model="mock")

        provider = llm.get("provider") or "mock"
        if provider == "mock":
            return LLMRouterConfig(provider="mock", model=llm.get("model") or "mock")

        timeout_seconds = llm.get("timeout_seconds")
        if timeout_seconds is None and llm.get("timeout_ms") is not None:
            timeout_seconds = max(1, int(llm["timeout_ms"]) // 1000)

        return LLMRouterConfig(
            provider=provider,
            model=llm.get("model") or "",
            api_key=llm.get("api_key") or llm.get("auth_token") or "",
            api_key_env=llm.get("api_key_env") or "",
            base_url=llm.get("base_url") or llm.get("api_base") or "",
            temperature=float(llm.get("temperature", 0.0)),
            timeout_seconds=int(timeout_seconds or 60),
            max_tokens=int(llm.get("max_tokens", 1024)),
            max_secondary_routes=int(llm.get("max_secondary_routes", 1)),
            fallback_confidence=float(llm.get("fallback_confidence", 0.60)),
        )

    @staticmethod
    def _load_local_config() -> tuple[Dict[str, Any], Optional[Path]]:
        """Load MemRouter local config without logging sensitive values.

        Priority:
            1. ``MEMROUTER_CONFIG`` if provided.
            2. ``<EchoMem repo>/configs/memrouter_eval.local.yaml``.
            3. ``<cwd>/configs/memrouter_eval.local.yaml``.

        The returned dict may contain hardcoded credentials, so callers should
        never log it directly.
        """
        import yaml

        repo_root = Path(__file__).resolve().parents[2]
        explicit = os.environ.get("MEMROUTER_CONFIG")
        candidates: List[Path] = []

        if explicit:
            candidates.append(Path(explicit).expanduser())
        else:
            candidates.extend(
                [
                    repo_root / "configs" / "memrouter_eval.local.yaml",
                    Path.cwd() / "configs" / "memrouter_eval.local.yaml",
                ]
            )

        for candidate in candidates:
            path = candidate if candidate.is_absolute() else Path.cwd() / candidate
            if not path.exists():
                continue
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(f"MemRouter config must be a YAML mapping: {path}")
            return data, path

        if explicit:
            raise FileNotFoundError(f"MEMROUTER_CONFIG does not exist: {explicit}")

        return {}, None

    @staticmethod
    def _get_embedding_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Return the embedding config section.

        Supports both shapes used in discussion:

        - ``embedding: {provider, model, api_base, api_key, ...}``
        - ``embedding: {dense: {provider, model, api_base, api_key, ...}}``
        """
        embedding = config.get("embedding")
        if not isinstance(embedding, dict):
            return {}
        dense = embedding.get("dense")
        if isinstance(dense, dict):
            return dense
        return embedding

    @staticmethod
    def _configure_run_logging(config: Dict[str, Any]) -> None:
        """Write EchoMem runtime logs to a local runs directory.

        The default location is ``<EchoMem repo>/runs/memrouter_viking_client``.
        It can be overridden by ``logging.runs_dir`` in the local config or by
        ``MEMROUTER_RUNS_DIR``. Sensitive config values are never logged here.
        """
        import datetime as _dt

        echomem_logger = logging.getLogger("echomem")
        for handler in echomem_logger.handlers:
            if getattr(handler, "_memrouter_runs_handler", False):
                return

        repo_root = Path(__file__).resolve().parents[2]
        logging_config = config.get("logging") if isinstance(config, dict) else None
        configured_runs_dir = None
        if isinstance(logging_config, dict):
            configured_runs_dir = logging_config.get("runs_dir")

        runs_dir = Path(
            configured_runs_dir
            or os.environ.get("MEMROUTER_RUNS_DIR")
            or (repo_root / "runs" / "memrouter_viking_client")
        )
        runs_dir.mkdir(parents=True, exist_ok=True)

        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = runs_dir / f"memrouter_viking_client_{stamp}_pid{os.getpid()}.log"

        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler._memrouter_runs_handler = True  # type: ignore[attr-defined]
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)s [%(name)s] %(message)s"
            )
        )
        echomem_logger.addHandler(handler)
        echomem_logger.setLevel(logging.INFO)
        logger.info("MemRouter runtime log file: %s", log_path)

    @staticmethod
    def _matched_context_to_dict(matched_context: Any) -> Dict[str, Any]:
        """Convert a MatchedContext to a plain dict (VikingClient-compatible)."""
        return {
            "uri": getattr(matched_context, "uri", ""),
            "context_type": str(getattr(matched_context, "context_type", "")),
            "abstract": getattr(matched_context, "abstract", ""),
            "overview": getattr(matched_context, "overview", None),
            "category": getattr(matched_context, "category", ""),
            "score": getattr(matched_context, "score", 0.0),
            "match_reason": getattr(matched_context, "match_reason", ""),
            "relations": [
                {
                    "from_uri": getattr(r, "from_uri", ""),
                    "to_uri": getattr(r, "to_uri", ""),
                    "relation_type": getattr(r, "relation_type", ""),
                    "reason": getattr(r, "reason", ""),
                }
                for r in getattr(matched_context, "relations", [])
            ],
        }
