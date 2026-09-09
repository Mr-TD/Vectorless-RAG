"""
Vectorless RAG — Custom Exception Classes
============================================
Defines a hierarchy of application-specific exceptions.
Each exception includes an HTTP status code for proper API responses.
"""


class VectorlessRAGError(Exception):
    """
    Base exception for all Vectorless RAG application errors.
    All custom exceptions inherit from this so the global error handler
    can catch them uniformly.
    """
    status_code = 500

    def __init__(self, message="An application error occurred.", status_code=None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code


class LLMServiceError(VectorlessRAGError):
    """
    Raised when the NVIDIA NIM API call fails.
    Covers: network errors, auth failures, rate limiting, invalid responses.
    """
    status_code = 502

    def __init__(self, message="LLM service encountered an error."):
        super().__init__(message, self.status_code)


class LLMAuthenticationError(LLMServiceError):
    """Raised when the NVIDIA API key is invalid or missing."""
    status_code = 401

    def __init__(self, message="NVIDIA API key is invalid or not configured."):
        super().__init__(message)
        self.status_code = 401


class LLMRateLimitError(LLMServiceError):
    """Raised when the NVIDIA API rate limit is hit."""
    status_code = 429

    def __init__(self, message="NVIDIA API rate limit exceeded. Please retry later."):
        super().__init__(message)
        self.status_code = 429


class DocumentParseError(VectorlessRAGError):
    """
    Raised when a document cannot be parsed.
    Covers: corrupted PDFs, encoding issues, empty files.
    """
    status_code = 422

    def __init__(self, message="Failed to parse the uploaded document."):
        super().__init__(message, self.status_code)


class IndexBuildError(VectorlessRAGError):
    """
    Raised when the hierarchical tree index cannot be built.
    Covers: LLM returns invalid structure, document too large, timeout.
    """
    status_code = 500

    def __init__(self, message="Failed to build the document index."):
        super().__init__(message, self.status_code)


class TraversalError(VectorlessRAGError):
    """
    Raised when the tree traversal fails during a query.
    Covers: max depth exceeded, no viable path found.
    """
    status_code = 500

    def __init__(self, message="Tree traversal failed during query processing."):
        super().__init__(message, self.status_code)


class ValidationError(VectorlessRAGError):
    """
    Raised for input validation failures.
    Covers: invalid file type, empty query, missing fields.
    """
    status_code = 400

    def __init__(self, message="Validation failed."):
        super().__init__(message, self.status_code)


class DocumentNotFoundError(VectorlessRAGError):
    """Raised when a requested document does not exist."""
    status_code = 404

    def __init__(self, message="Document not found."):
        super().__init__(message, self.status_code)
