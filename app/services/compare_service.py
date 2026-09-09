"""
Comparison Service — Traditional RAG vs Vectorless RAG
======================================================
Executes both Traditional RAG and Vectorless RAG against the same query and document.
Collects granular performance metrics, retrieval traces, and generates a side-by-side comparison.
"""

import time
from typing import Dict, Any, Optional
from flask import current_app
from app.models.document import ComparisonResult, TraditionalQueryResult, QueryResult
from app.services.traditional_rag import TraditionalRetrievalService
from app.services.vectorless_rag import IndexService, QueryService
from app.services.document_service import DocumentService
from app.utils.logger import get_logger, log_performance
from app.utils.exceptions import DocumentNotFoundError

logger = get_logger(__name__)


class CompareService:
    """
    Orchestrates execution of both Traditional Vector RAG and Vectorless RAG
    for side-by-side inspection, benchmarking, and architectural comparison.
    """

    def __init__(self, app=None):
        self.app = app
        self.doc_service = DocumentService(app)
        self.traditional_service = TraditionalRetrievalService(app)
        self.vectorless_index_service = IndexService(app)
        self.vectorless_query_service = QueryService(app)

    def check_indexes_status(self, doc_id: str) -> Dict[str, bool]:
        """
        Check whether traditional and vectorless indexes exist for a document.

        Args:
            doc_id: Unique document identifier.

        Returns:
            Dict indicating whether each index is built.
        """
        return {
            "traditional_indexed": self.traditional_service.has_index(doc_id),
            "vectorless_indexed": self.vectorless_index_service.has_index(doc_id),
        }

    @log_performance
    def compare(self, query: str, doc_id: str) -> ComparisonResult:
        """
        Run both RAG pipelines on the exact same question and document.

        Args:
            query: Natural language question.
            doc_id: Target document ID.

        Returns:
            ComparisonResult with full traces from both systems.
        """
        # Ensure document exists
        doc = self.doc_service.get_document(doc_id)
        logger.info(f"Comparing RAG methods on document '{doc.filename}' for query: '{query}'")

        statuses = self.check_indexes_status(doc_id)
        if not statuses["traditional_indexed"]:
            raise DocumentNotFoundError(
                f"Traditional index not found for '{doc.filename}'. Please build the traditional index first."
            )
        if not statuses["vectorless_indexed"]:
            raise DocumentNotFoundError(
                f"Vectorless index not found for '{doc.filename}'. Please build the vectorless index first."
            )

        # 1. Execute Traditional RAG
        t0 = time.time()
        trad_result = self.traditional_service.answer_query(query, doc_id)
        trad_time_ms = (time.time() - t0) * 1000

        # 2. Execute Vectorless RAG
        t1 = time.time()
        vless_result = self.vectorless_query_service.answer_query(query, doc_id)
        vless_time_ms = (time.time() - t1) * 1000

        result = ComparisonResult(
            query=query,
            document_id=doc_id,
            traditional=trad_result.to_dict(),
            vectorless=vless_result.to_dict(),
            traditional_time_ms=trad_time_ms,
            vectorless_time_ms=vless_time_ms,
        )

        logger.info(
            f"Comparison completed: Traditional took {trad_time_ms:.1f}ms, "
            f"Vectorless took {vless_time_ms:.1f}ms."
        )
        return result
