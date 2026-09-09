"""
Traditional RAG — Embedding Service
====================================
Generates dense vector embeddings using NVIDIA NIM embedding models via the OpenAI SDK.
Supports batch embedding for passages and single query embedding with input_type parameter.
"""

import time
from typing import List
from flask import current_app
from openai import (
    OpenAI,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APIStatusError,
)
from app.utils.logger import get_logger, log_performance
from app.utils.exceptions import (
    LLMServiceError,
    LLMAuthenticationError,
    LLMRateLimitError,
)

logger = get_logger(__name__)


class EmbeddingService:
    """
    Client for generating text embeddings using NVIDIA NIM embedding models.

    Uses the OpenAI Python SDK pointing to NVIDIA's base URL:
    https://integrate.api.nvidia.com/v1
    Model: nvidia/nv-embedqa-e5-v5 (or configured EMBEDDING_MODEL)
    """

    BATCH_SIZE = 16  # NVIDIA NIM supports batching

    def __init__(self, app=None):
        self.app = app
        self._client = None

    def _get_config(self):
        """Extract embedding configuration from current Flask app context."""
        app = self.app or current_app
        return {
            "api_key": app.config.get("NVIDIA_API_KEY", ""),
            "api_base_url": app.config.get("NVIDIA_API_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            "model": app.config.get("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5"),
            "timeout": app.config.get("REQUEST_TIMEOUT", 120),
            "max_retries": app.config.get("MAX_RETRIES", 3),
        }

    def _get_client(self):
        """Lazily initialize and return the OpenAI client."""
        if self._client is None:
            config = self._get_config()

            if not config["api_key"] or config["api_key"] == "your_nvidia_api_key_here":
                raise LLMAuthenticationError(
                    "NVIDIA API key is not configured. Set NVIDIA_API_KEY in your .env file."
                )

            self._client = OpenAI(
                base_url=config["api_base_url"],
                api_key=config["api_key"],
                timeout=config["timeout"],
            )
            logger.debug(f"OpenAI client initialized for embeddings: {config['api_base_url']}")

        return self._client

    @staticmethod
    def _generate_fallback_embedding(text: str, dim: int = 384) -> List[float]:
        """
        Generate a deterministic 384-dimensional feature vector based on token hashing.
        Used as a resilient offline fallback when NVIDIA API key is not yet set.
        """
        import hashlib
        import numpy as np
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        if not words:
            vec[0] = 1.0
            return vec.tolist()
        for w in words:
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest(), 16)
            idx = h % dim
            vec[idx] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec.tolist()

    @log_performance
    def embed_texts(self, texts: List[str], input_type: str = "passage") -> List[List[float]]:
        """
        Generate embeddings for a list of text strings in batches.

        Args:
            texts: List of strings to embed.
            input_type: "passage" for document chunks or "query" for user queries.

        Returns:
            List of embedding vectors (each vector is a list of floats).
        """
        if not texts:
            return []

        config = self._get_config()

        # If API key is missing, seamlessly fall back to local feature embeddings
        if not config["api_key"] or config["api_key"] == "your_nvidia_api_key_here":
            logger.warning("NVIDIA_API_KEY not configured. Generating deterministic fallback embeddings.")
            return [self._generate_fallback_embedding(t) for t in texts]

        try:
            client = self._get_client()
        except LLMAuthenticationError:
            logger.warning("NVIDIA authentication failed. Using fallback embeddings.")
            return [self._generate_fallback_embedding(t) for t in texts]

        model = config["model"]
        max_retries = config["max_retries"]

        all_embeddings: List[List[float]] = []

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            # Replace empty strings with a single space to avoid API validation errors
            cleaned_batch = [t.strip() if t and t.strip() else " " for t in batch]

            attempt = 0
            while attempt < max_retries:
                try:
                    attempt += 1
                    logger.debug(
                        f"Embedding batch {i // self.BATCH_SIZE + 1}/{(len(texts) - 1) // self.BATCH_SIZE + 1} "
                        f"({len(batch)} items) using model {model}"
                    )

                    extra_body = {}
                    if input_type:
                        extra_body["input_type"] = input_type

                    response = client.embeddings.create(
                        model=model,
                        input=cleaned_batch,
                        encoding_format="float",
                        extra_body=extra_body if extra_body else None,
                    )

                    # Extract vectors ordered by index
                    batch_vectors = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_vectors)
                    break

                except AuthenticationError as e:
                    logger.error(f"NVIDIA API Authentication Error: {e}")
                    raise LLMAuthenticationError(f"NVIDIA API authentication failed: {e}")

                except RateLimitError as e:
                    logger.warning(f"Rate limited by NVIDIA API (attempt {attempt}/{max_retries}): {e}")
                    if attempt >= max_retries:
                        raise LLMRateLimitError(f"Embedding rate limit exceeded: {e}")
                    time.sleep(2 ** attempt)

                except (APIConnectionError, APIStatusError) as e:
                    logger.warning(f"NVIDIA API error (attempt {attempt}/{max_retries}): {e}")
                    if attempt >= max_retries:
                        raise LLMServiceError(f"NVIDIA Embedding API error after {max_retries} attempts: {e}")
                    time.sleep(2 ** attempt)

                except Exception as e:
                    logger.error(f"Unexpected error during embedding generation: {e}", exc_info=True)
                    raise LLMServiceError(f"Embedding generation failed: {e}")

        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Generate embedding for a single user search query.

        Args:
            query: User's search query string.

        Returns:
            Vector of floats representing the query embedding.
        """
        embeddings = self.embed_texts([query], input_type="query")
        if not embeddings:
            raise LLMServiceError("Failed to generate embedding for query.")
        return embeddings[0]
