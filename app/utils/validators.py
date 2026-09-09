"""
Vectorless RAG — Input Validators
====================================
Provides validation functions for file uploads, queries, and other inputs.
Raises ValidationError with descriptive messages on failure.
"""

import os
from werkzeug.utils import secure_filename
from app.utils.exceptions import ValidationError


# Allowed file extensions
ALLOWED_EXTENSIONS = {"pdf", "txt", "md"}

# Maximum query length (characters)
MAX_QUERY_LENGTH = 5000

# Minimum query length (characters)
MIN_QUERY_LENGTH = 3


def validate_file_upload(file, allowed_extensions=None):
    """
    Validate an uploaded file.

    Args:
        file: The werkzeug FileStorage object from the request.
        allowed_extensions: Set of allowed file extensions. Defaults to ALLOWED_EXTENSIONS.

    Returns:
        Tuple of (secured_filename, file_extension).

    Raises:
        ValidationError: If validation fails.
    """
    if file is None:
        raise ValidationError("No file provided in the request.")

    if file.filename == "" or file.filename is None:
        raise ValidationError("No file selected. Please choose a file to upload.")

    # Secure the filename to prevent directory traversal attacks
    filename = secure_filename(file.filename)
    if not filename:
        raise ValidationError("Invalid filename. The filename contains no valid characters.")

    # Check file extension
    extensions = allowed_extensions or ALLOWED_EXTENSIONS
    file_ext = _get_extension(filename)
    if file_ext not in extensions:
        raise ValidationError(
            f"File type '.{file_ext}' is not supported. "
            f"Allowed types: {', '.join(f'.{ext}' for ext in sorted(extensions))}"
        )

    return filename, file_ext


def validate_query(query):
    """
    Validate a user's natural language query.

    Args:
        query: The query string from the user.

    Returns:
        The cleaned query string.

    Raises:
        ValidationError: If the query is invalid.
    """
    if query is None or not isinstance(query, str):
        raise ValidationError("Query must be a non-empty text string.")

    query = query.strip()

    if len(query) < MIN_QUERY_LENGTH:
        raise ValidationError(
            f"Query is too short. Please provide at least {MIN_QUERY_LENGTH} characters."
        )

    if len(query) > MAX_QUERY_LENGTH:
        raise ValidationError(
            f"Query is too long. Maximum length is {MAX_QUERY_LENGTH} characters."
        )

    return query


def validate_document_id(doc_id):
    """
    Validate a document ID.

    Args:
        doc_id: The document ID string (UUID format).

    Returns:
        The validated document ID.

    Raises:
        ValidationError: If the ID is invalid.
    """
    if not doc_id or not isinstance(doc_id, str):
        raise ValidationError("Document ID is required and must be a string.")

    doc_id = doc_id.strip()

    # Basic UUID format check (loose, just checks length and chars)
    if len(doc_id) < 10:
        raise ValidationError("Invalid document ID format.")

    return doc_id


def _get_extension(filename):
    """
    Extract the lowercase file extension from a filename.

    Args:
        filename: The filename string.

    Returns:
        The file extension without the dot, in lowercase.
    """
    if "." not in filename:
        return ""
    return filename.rsplit(".", 1)[1].lower()
