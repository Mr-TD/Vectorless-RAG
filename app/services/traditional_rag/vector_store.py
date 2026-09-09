"""
Traditional RAG — In-Memory Vector Store
==========================================
Stores chunks and dense vectors using numpy for fast cosine similarity search.
Supports saving and loading index files to disk without external DB dependencies.
"""

import os
import json
from typing import List, Tuple, Optional
import numpy as np
from app.models.document import Chunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """
    Lightweight, fast in-memory vector store using numpy.

    Computes cosine similarity between query vectors and stored chunk vectors.
    Persists to disk using compressed numpy archive (.npz) and chunk metadata (.json).
    """

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.embeddings: Optional[np.ndarray] = None  # Shape: (N, D)

    def count(self) -> int:
        """Return the number of stored vectors."""
        return len(self.chunks)

    def add(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """
        Add chunks and their corresponding embedding vectors to the store.

        Args:
            chunks: List of Chunk objects.
            embeddings: List of float vectors of identical dimension.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})."
            )

        if not chunks:
            return

        new_embeddings = np.array(embeddings, dtype=np.float32)

        if self.embeddings is None or len(self.chunks) == 0:
            self.embeddings = new_embeddings
            self.chunks = list(chunks)
        else:
            if new_embeddings.shape[1] != self.embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dimension mismatch: store has {self.embeddings.shape[1]}, "
                    f"new embeddings have {new_embeddings.shape[1]}"
                )
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
            self.chunks.extend(chunks)

        logger.debug(f"Added {len(chunks)} vectors to VectorStore (total: {len(self.chunks)})")

    def search(self, query_vector: List[float], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Search for the top-K chunks most similar to the query vector.

        Computes cosine similarity:
            cos_sim(A, B) = (A . B) / (||A|| * ||B||)

        Args:
            query_vector: Query embedding vector (list of floats).
            top_k: Number of nearest chunks to retrieve.

        Returns:
            List of tuples (Chunk, similarity_score) sorted descending by score.
        """
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = q.reshape(1, -1)

        # Compute L2 norms
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        q_norm = np.where(q_norm == 0, 1e-10, q_norm)

        emb_norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        emb_norms = np.where(emb_norms == 0, 1e-10, emb_norms)

        # Normalize to unit vectors
        q_unit = q / q_norm
        emb_unit = self.embeddings / emb_norms

        # Dot product of unit vectors equals cosine similarity
        similarities = np.dot(emb_unit, q_unit.T).flatten()

        # Handle top_k larger than total chunks
        k = min(top_k, len(self.chunks))

        # Retrieve top K indices
        top_indices = np.argpartition(similarities, -k)[-k:]
        # Sort top K in descending order
        sorted_top_indices = top_indices[np.argsort(-similarities[top_indices])]

        results: List[Tuple[Chunk, float]] = []
        for idx in sorted_top_indices:
            score = float(similarities[idx])
            results.append((self.chunks[idx], score))

        return results

    def save(self, base_path: str) -> None:
        """
        Persist vectors and chunk metadata to disk.

        Creates:
            <base_path>.npz   (compressed embeddings array)
            <base_path>.json  (chunk metadata list)
        """
        os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)

        if self.embeddings is not None and len(self.chunks) > 0:
            np.savez_compressed(f"{base_path}.npz", embeddings=self.embeddings)
        else:
            np.savez_compressed(f"{base_path}.npz", embeddings=np.empty((0, 0), dtype=np.float32))

        chunks_data = [chunk.to_dict() for chunk in self.chunks]
        with open(f"{base_path}.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        logger.info(f"VectorStore saved to {base_path}.npz and {base_path}.json ({len(self.chunks)} items)")

    def load(self, base_path: str) -> bool:
        """
        Load vectors and chunk metadata from disk.

        Args:
            base_path: Base path without extension.

        Returns:
            True if loaded successfully, False if files do not exist.
        """
        npz_file = f"{base_path}.npz"
        json_file = f"{base_path}.json"

        if not os.path.exists(npz_file) or not os.path.exists(json_file):
            return False

        with np.load(npz_file) as data:
            self.embeddings = data["embeddings"]

        with open(json_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
            self.chunks = [Chunk.from_dict(c) for c in chunks_data]

        logger.info(f"VectorStore loaded from {base_path} ({len(self.chunks)} items)")
        return True
