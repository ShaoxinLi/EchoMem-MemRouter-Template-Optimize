"""Embedding provider abstraction.

Uses Strategy pattern so that sentence-transformers (local) and OpenAI API
can be swapped without changing upstream code.

OpenAI model registration follows the same registry pattern as OpenViking's
LOCAL_DENSE_MODEL_SPECS: a central model-spec dict with lookup helpers.
"""

import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts.

        Args:
            texts: List of input strings.

        Returns:
            2-D numpy array of shape (len(texts), embedding_dim).
        """
        ...

    @abstractmethod
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...


class SentenceTransformersProvider(EmbeddingProvider):
    """Local embedding provider using sentence-transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        logger.info("Initializing SentenceTransformersProvider with model=%s", model_name)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install with: pip install 'echomem[local]'"
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._dim: int = self._model.get_sentence_embedding_dimension()
        logger.info("Loaded model '%s' with dimension=%d", model_name, self._dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        logger.debug("Embedding %d texts with sentence-transformers", len(texts))
        embeddings = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        # Ensure 2-D
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        return embeddings.astype(np.float32)

    def dimension(self) -> int:
        return self._dim


# --------------------------------------------------------------------------- #
# OpenAI embedding model registry (aligned with OpenViking's spec pattern)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class OpenAIEmbeddingModelSpec:
    """Specification for a supported OpenAI embedding model."""

    model_name: str
    dimension: int
    max_input_tokens: int = 8192
    supports_dimensions: bool = False


# Central registry of known OpenAI embedding models.
# New models are added here; no code changes needed elsewhere.
OPENAI_EMBEDDING_MODEL_SPECS: Dict[str, OpenAIEmbeddingModelSpec] = {
    "text-embedding-3-small": OpenAIEmbeddingModelSpec(
        model_name="text-embedding-3-small",
        dimension=1536,
        max_input_tokens=8192,
        supports_dimensions=True,
    ),
    "text-embedding-3-large": OpenAIEmbeddingModelSpec(
        model_name="text-embedding-3-large",
        dimension=3072,
        max_input_tokens=8192,
        supports_dimensions=True,
    ),
    "text-embedding-v3": OpenAIEmbeddingModelSpec(
        model_name="text-embedding-v3",
        dimension=1024,
        max_input_tokens=8192,
        supports_dimensions=True,
    ),
    "text-embedding-ada-002": OpenAIEmbeddingModelSpec(
        model_name="text-embedding-ada-002",
        dimension=1536,
        max_input_tokens=8191,
        supports_dimensions=False,
    ),
}


def get_openai_model_spec(model_name: str) -> OpenAIEmbeddingModelSpec:
    """Look up a registered OpenAI embedding model spec by name.

    Args:
        model_name: The model identifier (e.g. "text-embedding-3-small").

    Returns:
        The model spec.

    Raises:
        ValueError: If the model is not registered.
    """
    try:
        return OPENAI_EMBEDDING_MODEL_SPECS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown OpenAI embedding model '{model_name}'. "
            f"Supported models: {list(OPENAI_EMBEDDING_MODEL_SPECS.keys())}"
        ) from exc


def get_openai_model_default_dimension(model_name: str) -> int:
    """Return the default dimension for a registered OpenAI model."""
    return get_openai_model_spec(model_name).dimension


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI API embedding provider.

    Model metadata (dimension, max_input_tokens) is resolved through the
    OPENAI_EMBEDDING_MODEL_SPECS registry, avoiding hidden probe API calls.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        output_dimension: int | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        logger.info("Initializing OpenAIEmbeddingProvider with model=%s", model)
        try:
            import openai
        except ImportError as exc:
            raise ImportError(
                "openai is not installed. Install with: pip install 'echomem[openai]'"
            ) from exc

        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._max_batch_size = max_batch_size

        spec = OPENAI_EMBEDDING_MODEL_SPECS.get(model)

        if output_dimension is not None and spec is not None and not spec.supports_dimensions:
            raise ValueError(
                f"Model '{model}' does not support the OpenAI 'dimensions' parameter. "
                "Use its registered default dimension or choose a text-embedding-3 model."
            )

        # Expected output dimension is used for local validation and zero vectors.
        # API dimensions is only sent when the caller explicitly requests a
        # non-default output dimension on a model that supports it.
        self._expected_dimension: int | None = output_dimension or (spec.dimension if spec else None)
        self._api_dimensions: int | None = output_dimension

        if self._expected_dimension:
            logger.info("Resolved expected_dimension=%d for model=%s", self._expected_dimension, model)
        else:
            logger.warning(
                "Model '%s' is not registered in OPENAI_EMBEDDING_MODEL_SPECS. "
                "Pass 'output_dimension' at init to avoid a probe API call.",
                model,
            )

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            dim = self._expected_dimension or 1536
            return np.zeros((0, dim), dtype=np.float32)

        logger.debug("Embedding %d texts with OpenAI model=%s", len(texts), self._model)

        # Chunk if provider imposes a max batch size (e.g. DashScope = 10)
        max_batch = self._max_batch_size
        if max_batch is not None and len(texts) > max_batch:
            chunks = [texts[i : i + max_batch] for i in range(0, len(texts), max_batch)]
            logger.debug("Split into %d chunks (max_batch_size=%d)", len(chunks), max_batch)
            batches = [self._embed_batch(chunk) for chunk in chunks]
            embeddings = np.concatenate(batches, axis=0)
        else:
            embeddings = self._embed_batch(texts)

        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms
        return embeddings

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        kwargs: Dict[str, Any] = {"input": texts, "model": self._model}
        if self._api_dimensions is not None:
            kwargs["dimensions"] = self._api_dimensions

        response = self._client.embeddings.create(**kwargs)
        embeddings = np.array([item.embedding for item in response.data], dtype=np.float32)

        # Validate actual dimension matches expectation
        actual_dim = embeddings.shape[1]
        if self._expected_dimension is not None and actual_dim != self._expected_dimension:
            logger.warning(
                "Embedding dimension mismatch: expected %d, got %d. "
                "Model '%s' may not support the 'dimensions' parameter.",
                self._expected_dimension,
                actual_dim,
                self._model,
            )
            self._expected_dimension = actual_dim

        return embeddings

    def dimension(self) -> int:
        if self._expected_dimension is None:
            logger.warning(
                "OpenAI dimension unknown for model=%s; triggering probe embedding. "
                "Register the model in OPENAI_EMBEDDING_MODEL_SPECS or pass 'output_dimension' at init.",
                self._model,
            )
            self.embed(["probe"])
        return self._expected_dimension  # type: ignore[return-value]


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock embedder for CI and smoke tests.

    Embeddings are based on a simple hash so that similar texts
    get somewhat similar vectors (cosine similarity in [-1, 1]).
    """

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        embeddings = []
        for t in texts:
            # Stable seed across processes and Python runs
            seed = int.from_bytes(
                hashlib.sha256(t.encode("utf-8")).digest()[:4],
                "little",
            )
            rng = np.random.default_rng(seed=seed)
            vec = rng.random(self._dim).astype(np.float32) - 0.5
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings, dtype=np.float32)

    def dimension(self) -> int:
        return self._dim


def create_provider(provider_type: str, **kwargs: object) -> EmbeddingProvider:
    """Factory for creating embedding providers.

    Args:
        provider_type: "mock", "sentence-transformers" or "openai".
        **kwargs: Provider-specific arguments.

    Returns:
        Configured EmbeddingProvider instance.
    """
    logger.info("Creating embedding provider: type=%s", provider_type)
    if provider_type == "mock":
        return MockEmbeddingProvider(**kwargs)  # type: ignore[arg-type]
    if provider_type == "sentence-transformers":
        return SentenceTransformersProvider(**kwargs)  # type: ignore[arg-type]
    if provider_type == "openai":
        return OpenAIEmbeddingProvider(**kwargs)  # type: ignore[arg-type]
    raise ValueError(f"Unknown embedding provider: {provider_type}")
