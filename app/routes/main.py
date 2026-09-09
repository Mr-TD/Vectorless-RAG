"""
Vectorless RAG — Page Routes
================================
Serves the HTML pages for the web UI.
"""

from flask import Blueprint, render_template, current_app
from app.services.document_service import DocumentService
from app.services.vectorless_rag import IndexService
from app.services.traditional_rag import TraditionalRetrievalService
from app.utils.logger import get_logger

logger = get_logger(__name__)

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    """Home page — Document upload interface."""
    return render_template("index.html")


@main_bp.route("/documents")
def documents():
    """Document management page — list all documents."""
    try:
        doc_service = DocumentService()
        docs = doc_service.list_documents()
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        docs = []

    return render_template("documents.html", documents=docs)


@main_bp.route("/query/<doc_id>")
def query(doc_id):
    """Query interface for a specific document."""
    try:
        doc_service = DocumentService()
        doc = doc_service.get_document(doc_id)
        doc_data = doc.to_dict()

        # Check index statuses
        index_service = IndexService()
        traditional_service = TraditionalRetrievalService()
        doc_data["is_indexed"] = index_service.has_index(doc_id)
        doc_data["vectorless_indexed"] = doc_data["is_indexed"]
        doc_data["traditional_indexed"] = traditional_service.has_index(doc_id)

    except Exception as e:
        logger.error(f"Error loading query page: {e}")
        doc_data = {
            "id": doc_id,
            "filename": "Unknown",
            "is_indexed": False,
            "vectorless_indexed": False,
            "traditional_indexed": False,
            "error": str(e),
        }

    return render_template("query.html", document=doc_data)


@main_bp.route("/compare/<doc_id>")
def compare(doc_id):
    """Side-by-side comparison interface for Traditional vs Vectorless RAG."""
    try:
        doc_service = DocumentService()
        doc = doc_service.get_document(doc_id)
        doc_data = doc.to_dict()

        index_service = IndexService()
        traditional_service = TraditionalRetrievalService()

        doc_data["vectorless_indexed"] = index_service.has_index(doc_id)
        doc_data["traditional_indexed"] = traditional_service.has_index(doc_id)

    except Exception as e:
        logger.error(f"Error loading compare page: {e}")
        doc_data = {
            "id": doc_id,
            "filename": "Unknown",
            "vectorless_indexed": False,
            "traditional_indexed": False,
            "error": str(e),
        }

    return render_template("compare.html", document=doc_data)
