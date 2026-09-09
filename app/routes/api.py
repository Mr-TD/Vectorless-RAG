"""
Vectorless RAG — REST API Routes
====================================
JSON API endpoints for document management, indexing, and querying.

All API responses follow a consistent structure:
{
    "success": true/false,
    "data": { ... },        // on success
    "error": { ... }        // on failure
}
"""

from flask import Blueprint, request, jsonify, current_app
from app.services.document_service import DocumentService
from app.services.vectorless_rag import IndexService, QueryService
from app.services.traditional_rag import TraditionalRetrievalService
from app.services.compare_service import CompareService
from app.services.llm_service import LLMService
from app.utils.validators import validate_query, validate_document_id
from app.utils.exceptions import VectorlessRAGError
from app.utils.logger import get_logger

logger = get_logger(__name__)

api_bp = Blueprint("api", __name__)


# ================================================================
# Helper: Consistent JSON response
# ================================================================

def success_response(data=None, message="", status_code=200):
    """Build a successful JSON response."""
    body = {"success": True}
    if data is not None:
        body["data"] = data
    if message:
        body["message"] = message
    return jsonify(body), status_code


def error_response(message, error_type="Error", status_code=400):
    """Build an error JSON response."""
    return jsonify({
        "success": False,
        "error": {"type": error_type, "message": message}
    }), status_code


# ================================================================
# Health Check
# ================================================================

@api_bp.route("/health", methods=["GET"])
def health_check():
    """
    Check application health and NVIDIA API connectivity.

    Returns:
        JSON with status and latency information.
    """
    llm = LLMService()
    health = llm.health_check()
    status_code = 200 if health["status"] == "healthy" else 503
    return jsonify({"success": True, "data": health}), status_code


# ================================================================
# Document Management
# ================================================================

@api_bp.route("/upload", methods=["POST"])
def upload_document():
    """
    Upload and ingest a document.

    Accepts a file via multipart form upload. Validates the file type,
    saves it, and extracts text content page by page.

    Returns:
        JSON with the new document's metadata.
    """
    if "file" not in request.files:
        return error_response("No file field in the request.", "ValidationError")

    file = request.files["file"]

    try:
        doc_service = DocumentService()
        doc = doc_service.ingest(file)

        logger.info(f"Document uploaded: {doc.filename} (ID: {doc.id})")

        return success_response(
            data=doc.to_dict(),
            message=f"Document '{doc.filename}' uploaded successfully. {doc.page_count} pages extracted.",
            status_code=201,
        )

    except VectorlessRAGError as e:
        return error_response(str(e), e.__class__.__name__, e.status_code)
    except Exception as e:
        logger.error(f"Upload failed: {e}", exc_info=True)
        return error_response(f"Upload failed: {e}", "InternalError", 500)


@api_bp.route("/documents", methods=["GET"])
def list_documents():
    """
    List all ingested documents.

    Returns:
        JSON array of document summaries.
    """
    try:
        doc_service = DocumentService()
        documents = doc_service.list_documents()
        return success_response(data={"documents": documents, "count": len(documents)})
    except Exception as e:
        logger.error(f"List documents failed: {e}", exc_info=True)
        return error_response(str(e), "InternalError", 500)


@api_bp.route("/documents/<doc_id>", methods=["GET"])
def get_document(doc_id):
    """
    Get detailed information about a specific document.

    Returns:
        JSON with document metadata and tree structure (if indexed).
    """
    try:
        doc_id = validate_document_id(doc_id)
        doc_service = DocumentService()
        doc = doc_service.get_document(doc_id)

        data = doc.to_dict()

        # Include indexing info
        index_service = IndexService()
        trad_service = TraditionalRetrievalService()
        has_vless = index_service.has_index(doc_id)
        has_trad = trad_service.has_index(doc_id)

        data["has_vectorless_index"] = has_vless
        data["has_traditional_index"] = has_trad

        if has_vless:
            tree = index_service.get_index(doc_id)
            data["tree"] = tree.to_dict()
            data["tree_node_count"] = tree.get_node_count()

        return success_response(data=data)

    except VectorlessRAGError as e:
        return error_response(str(e), e.__class__.__name__, e.status_code)
    except Exception as e:
        logger.error(f"Get document failed: {e}", exc_info=True)
        return error_response(str(e), "InternalError", 500)


@api_bp.route("/documents/<doc_id>", methods=["DELETE"])
def delete_document(doc_id):
    """
    Delete a document and its index.

    Returns:
        JSON confirmation of deletion.
    """
    try:
        doc_id = validate_document_id(doc_id)
        doc_service = DocumentService()
        doc_service.delete_document(doc_id)
        return success_response(message=f"Document '{doc_id}' deleted successfully.")

    except VectorlessRAGError as e:
        return error_response(str(e), e.__class__.__name__, e.status_code)
    except Exception as e:
        logger.error(f"Delete document failed: {e}", exc_info=True)
        return error_response(str(e), "InternalError", 500)


# ================================================================
# Indexing
# ================================================================

@api_bp.route("/index/<doc_id>", methods=["POST"])
def build_index(doc_id):
    """
    Build the hierarchical tree index for a document.

    This is the core Vectorless RAG indexing step. The LLM analyzes
    the document structure and creates a navigable tree.

    Returns:
        JSON with the tree structure and node count.
    """
    try:
        doc_id = validate_document_id(doc_id)
        index_service = IndexService()
        tree = index_service.build_index(doc_id)

        return success_response(
            data={
                "document_id": doc_id,
                "tree": tree.to_dict(),
                "node_count": tree.get_node_count(),
            },
            message="Index built successfully.",
            status_code=201,
        )

    except VectorlessRAGError as e:
        return error_response(str(e), e.__class__.__name__, e.status_code)
    except Exception as e:
        logger.error(f"Index build failed: {e}", exc_info=True)
        return error_response(str(e), "InternalError", 500)


# ================================================================
# Querying
# ================================================================

@api_bp.route("/query", methods=["POST"])
def query_document():
    """
    Submit a query against an indexed document.

    The system traverses the document tree using LLM reasoning
    to find the most relevant section, then synthesizes an answer.

    Request body (JSON):
    {
        "query": "What is the total revenue?",
        "document_id": "abc-123-..."
    }

    Returns:
        JSON with the answer, sources, traversal path, and confidence.
    """
    try:
        data = request.get_json()
        if not data:
            return error_response("Request body must be valid JSON.", "ValidationError")

        # Validate inputs
        query = validate_query(data.get("query", ""))
        doc_id = validate_document_id(data.get("document_id", ""))

        # Process the query
        query_service = QueryService()
        result = query_service.answer_query(query, doc_id)

        return success_response(data=result.to_dict())

    except VectorlessRAGError as e:
        return error_response(str(e), e.__class__.__name__, e.status_code)
    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        return error_response(str(e), "InternalError", 500)


# ================================================================
# Traditional RAG Indexing & Querying
# ================================================================

@api_bp.route("/traditional/index/<doc_id>", methods=["POST"])
def build_traditional_index(doc_id):
    """
    Build the vector embedding index for a document.
    Chunks the document text and computes dense embeddings via NVIDIA NIM.
    """
    try:
        doc_id = validate_document_id(doc_id)
        trad_service = TraditionalRetrievalService()
        info = trad_service.build_index(doc_id)
        return success_response(
            data=info,
            message=f"Traditional vector index built successfully ({info['chunk_count']} chunks).",
            status_code=201,
        )
    except VectorlessRAGError as e:
        return error_response(str(e), e.__class__.__name__, e.status_code)
    except Exception as e:
        logger.error(f"Traditional index build failed: {e}", exc_info=True)
        return error_response(str(e), "InternalError", 500)


@api_bp.route("/traditional/query", methods=["POST"])
def query_traditional():
    """
    Query a document using Traditional RAG (Embedding + Cosine Similarity).
    """
    try:
        data = request.get_json()
        if not data:
            return error_response("Request body must be valid JSON.", "ValidationError")

        query = validate_query(data.get("query", ""))
        doc_id = validate_document_id(data.get("document_id", ""))

        trad_service = TraditionalRetrievalService()
        result = trad_service.answer_query(query, doc_id)
        return success_response(data=result.to_dict())
    except VectorlessRAGError as e:
        return error_response(str(e), e.__class__.__name__, e.status_code)
    except Exception as e:
        logger.error(f"Traditional query failed: {e}", exc_info=True)
        return error_response(str(e), "InternalError", 500)


# ================================================================
# Comparison Endpoints
# ================================================================

@api_bp.route("/compare", methods=["POST"])
def compare_query():
    """
    Execute both Traditional RAG and Vectorless RAG on the same query
    and return side-by-side performance, answers, and retrieval traces.
    """
    try:
        data = request.get_json()
        if not data:
            return error_response("Request body must be valid JSON.", "ValidationError")

        query = validate_query(data.get("query", ""))
        doc_id = validate_document_id(data.get("document_id", ""))

        compare_service = CompareService()
        comparison = compare_service.compare(query, doc_id)
        return success_response(data=comparison.to_dict())
    except VectorlessRAGError as e:
        return error_response(str(e), e.__class__.__name__, e.status_code)
    except Exception as e:
        logger.error(f"Comparison failed: {e}", exc_info=True)
        return error_response(str(e), "InternalError", 500)


@api_bp.route("/indexes/status/<doc_id>", methods=["GET"])
def get_indexes_status(doc_id):
    """Get the indexing status for both Traditional and Vectorless pipelines."""
    try:
        doc_id = validate_document_id(doc_id)
        compare_service = CompareService()
        statuses = compare_service.check_indexes_status(doc_id)
        return success_response(data=statuses)
    except VectorlessRAGError as e:
        return error_response(str(e), e.__class__.__name__, e.status_code)
    except Exception as e:
        logger.error(f"Status check failed: {e}", exc_info=True)
        return error_response(str(e), "InternalError", 500)

