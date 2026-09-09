"""
Traditional RAG — Retrieval & Query Service
============================================
Orchestrates the complete Traditional RAG pipeline:
1. Indexing: Text Chunking -> Dense Embeddings (NVIDIA NIM) -> Vector Store
2. Querying: Query Embedding -> Cosine Similarity Search -> Top-K Retrieval -> LLM Synthesis
"""

import os
import time
from typing import List, Dict, Any, Tuple
from flask import current_app
from app.models.document import Document, Chunk, TraditionalQueryResult
from app.services.document_service import DocumentService
from app.services.llm_service import LLMService
from app.services.traditional_rag.chunker import Chunker
from app.services.traditional_rag.embedding_service import EmbeddingService
from app.services.traditional_rag.vector_store import VectorStore
from app.utils.logger import get_logger, log_performance
from app.utils.exceptions import IndexBuildError, DocumentNotFoundError

logger = get_logger(__name__)


class TraditionalRetrievalService:
    """
    Traditional RAG pipeline for document querying.

    Unlike Vectorless RAG which navigates hierarchical tables of contents,
    Traditional RAG:
    - Splits text into fixed overlapping chunks
    - Maps chunks into dense semantic vector space
    - Performs nearest-neighbor cosine similarity search
    - Stitches retrieved chunks together as prompt context for the LLM
    """

    def __init__(self, app=None):
        self.app = app
        self.doc_service = DocumentService(app)
        self.chunker = Chunker(app=app)
        self.embedding_service = EmbeddingService(app)
        self.llm_service = LLMService(app)

    def _get_app_config(self):
        app = self.app or (current_app._get_current_object() if current_app else None)
        folder = app.config.get("TRADITIONAL_INDEX_FOLDER") if app else "data/traditional_indexes"
        top_k = int(app.config.get("TOP_K_CHUNKS", 5)) if app else 5
        model = app.config.get("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5") if app else "nvidia/nv-embedqa-e5-v5"
        return folder, top_k, model

    def _get_index_base_path(self, doc_id: str) -> str:
        folder, _, _ = self._get_app_config()
        return os.path.join(folder, doc_id)

    def has_index(self, doc_id: str) -> bool:
        """Check if a traditional vector index exists for this document."""
        base_path = self._get_index_base_path(doc_id)
        return os.path.exists(f"{base_path}.npz") and os.path.exists(f"{base_path}.json")

    @log_performance
    def build_index(self, doc_id: str) -> Dict[str, Any]:
        """
        Build a vector index for the document:
        1. Load document text
        2. Chunk text with overlap
        3. Embed all chunks via NVIDIA NIM embedding API
        4. Save vectors and chunk metadata to disk

        Args:
            doc_id: Unique document identifier.

        Returns:
            Dict containing build statistics.
        """
        doc = self.doc_service.get_document(doc_id)
        logger.info(f"Building traditional vector index for '{doc.filename}' ({doc.id})")

        # Step 1: Chunk document
        chunks = self.chunker.chunk_document(doc)
        if not chunks:
            raise IndexBuildError(f"No textual content extracted from document '{doc.filename}' to chunk.")

        # Step 2: Embed all chunks
        texts = [chunk.text for chunk in chunks]
        folder, _, model_name = self._get_app_config()
        logger.info(f"Generating {len(texts)} embeddings using model '{model_name}'...")

        try:
            embeddings = self.embedding_service.embed_texts(texts, input_type="passage")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for document {doc_id}: {e}", exc_info=True)
            raise IndexBuildError(f"Embedding generation failed: {e}")

        # Step 3: Populate vector store
        store = VectorStore()
        store.add(chunks, embeddings)

        # Step 4: Persist index
        base_path = self._get_index_base_path(doc_id)
        store.save(base_path)

        logger.info(f"Traditional RAG index created for '{doc.filename}' with {len(chunks)} chunks.")
        return {
            "document_id": doc_id,
            "filename": doc.filename,
            "chunk_count": len(chunks),
            "embedding_model": model_name,
            "status": "indexed",
        }

    @log_performance
    def answer_query(self, query: str, doc_id: str) -> TraditionalQueryResult:
        """
        Answer a user question using Traditional Vector RAG:
        1. Embed the query
        2. Retrieve Top-K chunks by cosine similarity
        3. Build context from chunks
        4. Ask LLM to answer using only that context

        Args:
            query: Natural language question.
            doc_id: Target document ID.

        Returns:
            TraditionalQueryResult dataclass instance with answer, scores, and trace.
        """
        start_time = time.time()
        logger.info(f"[Traditional RAG] Querying document {doc_id}: '{query[:100]}'")

        if not self.has_index(doc_id):
            raise DocumentNotFoundError(
                f"No traditional vector index found for document '{doc_id}'. "
                f"Please build the traditional index first."
            )

        doc = self.doc_service.get_document(doc_id)
        base_path = self._get_index_base_path(doc_id)
        store = VectorStore()
        store.load(base_path)

        _, top_k, _ = self._get_app_config()

        # Step 1: Embed the search query
        query_vector = self.embedding_service.embed_query(query)

        # Step 2: Cosine similarity search
        search_results: List[Tuple[Chunk, float]] = store.search(query_vector, top_k=top_k)

        # Format retrieved chunks for response and tracing
        retrieved_chunks = []
        context_parts = []

        for chunk, score in search_results:
            retrieved_chunks.append({
                "chunk_id": chunk.id,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "score": round(float(score), 4),
                "text": chunk.text,
                "start_pos": chunk.start_pos,
                "end_pos": chunk.end_pos,
            })
            context_parts.append(
                f"[Chunk {chunk.chunk_index + 1} - Page {chunk.page_number} - Similarity: {round(score, 3)}]:\n"
                f"{chunk.text}"
            )

        context_str = "\n\n---\n\n".join(context_parts)

        # Step 3: LLM Answer Synthesis
        answer, confidence = self._synthesize_answer(query, context_str, doc.filename)

        elapsed_ms = (time.time() - start_time) * 1000

        result = TraditionalQueryResult(
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            context_used=context_str,
            confidence=confidence,
            query=query,
            document_id=doc_id,
            timing_ms=elapsed_ms,
        )

        logger.info(
            f"[Traditional RAG] Finished query in {elapsed_ms:.1f}ms. "
            f"Retrieved {len(retrieved_chunks)} chunks, confidence: {confidence}"
        )
        return result

    def _synthesize_answer(self, query: str, context: str, filename: str) -> Tuple[str, str]:
        """Synthesize answer using retrieved chunk context."""
        system_prompt = """You are a precise question-answering assistant implementing Traditional Vector RAG.
You answer questions based strictly on the provided retrieved text chunks.

RULES:
1. Answer the question using ONLY the retrieved chunks below.
2. If the context does not contain enough information, state that clearly.
3. Reference which chunk or page informed your answer when applicable.
4. End your answer with a confidence line: CONFIDENCE: <HIGH/MEDIUM/LOW>
5. Format your output as:
   ANSWER: <detailed answer>
   CONFIDENCE: <HIGH/MEDIUM/LOW>"""

        user_prompt = f"""QUESTION: {query}

RETRIEVED CONTEXT CHUNKS (from "{filename}"):
==================================================
{context}
==================================================

Answer the question based strictly on the retrieved chunks above."""

        try:
            response = self.llm_service.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
            )

            answer, confidence = self._parse_answer(response)
            return answer, confidence

        except Exception as e:
            logger.error(f"Answer synthesis failed in Traditional RAG: {e}")
            return (
                f"Retrieved relevant chunks but encountered an error synthesizing the answer: {e}",
                "low",
            )

    def _parse_answer(self, response: str) -> Tuple[str, str]:
        """Extract answer text and confidence from LLM response."""
        response = response.strip()
        answer = response
        confidence = "medium"

        if "CONFIDENCE:" in response.upper():
            parts = response.upper().split("CONFIDENCE:")
            answer = response[:response.upper().rfind("CONFIDENCE:")].strip()
            conf_text = parts[-1].strip().lower()

            if "high" in conf_text:
                confidence = "high"
            elif "low" in conf_text:
                confidence = "low"
            else:
                confidence = "medium"

        if answer.upper().startswith("ANSWER:"):
            answer = answer[7:].strip()

        return answer, confidence
