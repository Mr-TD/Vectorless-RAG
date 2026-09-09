"""
Vectorless RAG — Document Service
=====================================
Handles document ingestion: file upload, validation, text extraction,
and metadata management.

Supports:
    - PDF files (via PyPDF2)
    - Plain text files (.txt, .md)

The extracted page-level content is stored as a Document object
and persisted as JSON metadata for later use by the IndexService.
"""

import os
import json
from flask import current_app
from PyPDF2 import PdfReader
from app.models.document import Document
from app.utils.logger import get_logger, log_performance
from app.utils.exceptions import DocumentParseError, DocumentNotFoundError, ValidationError
from app.utils.validators import validate_file_upload

logger = get_logger(__name__)


class DocumentService:
    """
    Manages document ingestion, storage, and retrieval.

    Documents are stored on disk with their extracted text content
    persisted as JSON metadata alongside the original file.
    """

    def __init__(self, app=None):
        self.app = app

    def _get_upload_folder(self):
        app = self.app or current_app
        return app.config["UPLOAD_FOLDER"]

    def _get_metadata_path(self, doc_id):
        """Get the path to the document's metadata JSON file."""
        return os.path.join(self._get_upload_folder(), f"{doc_id}.json")

    @log_performance
    def ingest(self, file):
        """
        Ingest an uploaded file: validate, save, extract text.

        Args:
            file: The werkzeug FileStorage object from the request.

        Returns:
            A Document object with extracted page content.

        Raises:
            ValidationError: If the file is invalid.
            DocumentParseError: If the file content cannot be extracted.
        """
        # Validate the upload
        filename, file_ext = validate_file_upload(file)
        logger.info(f"Ingesting document: {filename} (type: .{file_ext})")

        # Create the Document object
        doc = Document(filename=filename, file_type=file_ext)

        # Save the original file to disk
        upload_folder = self._get_upload_folder()
        filepath = os.path.join(upload_folder, f"{doc.id}_{filename}")
        file.save(filepath)
        doc.filepath = filepath
        doc.file_size = os.path.getsize(filepath)

        logger.info(f"File saved: {filepath} ({doc.file_size} bytes)")

        # Extract text content
        try:
            if file_ext == "pdf":
                doc.pages = self._extract_pdf(filepath)
            elif file_ext in ("txt", "md"):
                doc.pages = self._extract_text(filepath)
            else:
                raise DocumentParseError(f"Unsupported file type: .{file_ext}")

            doc.page_count = len(doc.pages)
            doc.status = "uploaded"

            if doc.page_count == 0:
                raise DocumentParseError("Document appears to be empty — no text content found.")

            logger.info(f"Extracted {doc.page_count} pages from {filename}")

        except DocumentParseError:
            doc.status = "error"
            doc.error_message = "Failed to extract text content."
            self._save_metadata(doc)
            raise

        except Exception as e:
            doc.status = "error"
            doc.error_message = str(e)
            self._save_metadata(doc)
            raise DocumentParseError(f"Error extracting content from {filename}: {e}")

        # Save metadata
        self._save_metadata(doc)
        return doc

    def get_document(self, doc_id):
        """
        Retrieve a document by its ID.

        Args:
            doc_id: The document's unique identifier.

        Returns:
            A Document object.

        Raises:
            DocumentNotFoundError: If the document does not exist.
        """
        metadata_path = self._get_metadata_path(doc_id)
        if not os.path.exists(metadata_path):
            raise DocumentNotFoundError(f"Document '{doc_id}' not found.")

        with open(metadata_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return Document.from_dict(data)

    def list_documents(self):
        """
        List all ingested documents.

        Returns:
            List of Document summary dictionaries (without page content).
        """
        upload_folder = self._get_upload_folder()
        documents = []

        if not os.path.exists(upload_folder):
            return documents

        for fname in os.listdir(upload_folder):
            if fname.endswith(".json"):
                try:
                    fpath = os.path.join(upload_folder, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    doc = Document.from_dict(data)
                    documents.append(doc.to_dict())
                except Exception as e:
                    logger.warning(f"Failed to read metadata file {fname}: {e}")

        # Sort by upload time (newest first)
        documents.sort(key=lambda d: d.get("upload_time", ""), reverse=True)
        return documents

    def delete_document(self, doc_id):
        """
        Delete a document and its metadata.

        Args:
            doc_id: The document's unique identifier.

        Raises:
            DocumentNotFoundError: If the document does not exist.
        """
        doc = self.get_document(doc_id)

        # Delete the original file
        if doc.filepath and os.path.exists(doc.filepath):
            os.remove(doc.filepath)
            logger.info(f"Deleted file: {doc.filepath}")

        # Delete the metadata file
        metadata_path = self._get_metadata_path(doc_id)
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
            logger.info(f"Deleted metadata: {metadata_path}")

        # Delete the index if it exists
        app = self.app or current_app
        index_path = os.path.join(app.config["INDEX_FOLDER"], f"{doc_id}.json")
        if os.path.exists(index_path):
            os.remove(index_path)
            logger.info(f"Deleted index: {index_path}")

    def update_status(self, doc_id, status, error_message=""):
        """
        Update a document's processing status.

        Args:
            doc_id: The document's unique identifier.
            status: New status string (uploaded/indexing/indexed/error).
            error_message: Optional error message.
        """
        doc = self.get_document(doc_id)
        doc.status = status
        doc.error_message = error_message
        self._save_metadata(doc)
        logger.info(f"Document {doc_id} status updated to: {status}")

    # ================================================================
    # Private: Text Extraction Methods
    # ================================================================

    def _extract_pdf(self, filepath):
        """
        Extract text from a PDF file, page by page.

        Preserves the page structure — each entry in the returned list
        corresponds to one PDF page.

        Args:
            filepath: Path to the PDF file.

        Returns:
            List of strings, one per page.

        Raises:
            DocumentParseError: If the PDF cannot be read.
        """
        try:
            reader = PdfReader(filepath)
            pages = []

            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    pages.append(text.strip())
                else:
                    logger.debug(f"Page {i+1}: No extractable text (may be scanned/image).")
                    pages.append(f"[Page {i+1}: No extractable text]")

            if not pages:
                raise DocumentParseError("PDF contains no extractable text pages.")

            return pages

        except DocumentParseError:
            raise
        except Exception as e:
            raise DocumentParseError(f"Failed to read PDF file: {e}")

    def _extract_text(self, filepath):
        """
        Extract text from a plain text file (.txt, .md).

        Splits content into logical pages/sections based on:
        1. Page break markers (\\f, --- separators)
        2. Or fixed line-count chunks if no markers found

        Args:
            filepath: Path to the text file.

        Returns:
            List of strings, one per logical section.

        Raises:
            DocumentParseError: If the file cannot be read.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                raise DocumentParseError("Text file is empty.")

            # Try splitting by form feed characters (page breaks)
            if "\f" in content:
                pages = [p.strip() for p in content.split("\f") if p.strip()]
                if pages:
                    return pages

            # Try splitting by markdown-style horizontal rules
            import re
            sections = re.split(r"\n-{3,}\n|\n={3,}\n|\n#{1,2}\s", content)
            sections = [s.strip() for s in sections if s.strip()]

            if len(sections) > 1:
                return sections

            # Fallback: split into chunks of ~50 lines
            lines = content.split("\n")
            chunk_size = 50
            pages = []
            for i in range(0, len(lines), chunk_size):
                chunk = "\n".join(lines[i:i + chunk_size]).strip()
                if chunk:
                    pages.append(chunk)

            return pages if pages else [content]

        except DocumentParseError:
            raise
        except UnicodeDecodeError:
            raise DocumentParseError("File encoding not supported. Please use UTF-8.")
        except Exception as e:
            raise DocumentParseError(f"Failed to read text file: {e}")

    def _save_metadata(self, doc):
        """Save document metadata to a JSON file."""
        metadata_path = self._get_metadata_path(doc.id)
        with open(metadata_path, "w", encoding="utf-8") as f:
            f.write(doc.to_json())
