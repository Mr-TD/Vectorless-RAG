"""
Vectorless RAG — Data Models
==============================
Defines the core data structures used throughout the application.
Uses Python dataclasses for clean, type-hinted, serializable models.
"""

import uuid
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


# ============================================================
# Document Model
# ============================================================

@dataclass
class Document:
    """
    Represents an uploaded document in the system.

    Attributes:
        id:          Unique identifier for the document.
        filename:    Original filename as uploaded by the user.
        filepath:    Absolute path to the stored file on disk.
        upload_time: ISO-format timestamp of when the document was uploaded.
        status:      Current processing status (uploaded/indexing/indexed/error).
        page_count:  Number of pages (for PDFs) or sections (for text).
        file_type:   File extension (pdf, txt, md).
        file_size:   File size in bytes.
        pages:       List of page contents, each entry is the text of one page.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    filepath: str = ""
    upload_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    status: str = "uploaded"  # uploaded | indexing | indexed | error
    page_count: int = 0
    file_type: str = ""
    file_size: int = 0
    pages: list = field(default_factory=list)
    error_message: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary (excludes bulky page content for listing)."""
        return {
            "id": self.id,
            "filename": self.filename,
            "upload_time": self.upload_time,
            "status": self.status,
            "page_count": self.page_count,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "error_message": self.error_message,
        }

    def to_json(self) -> str:
        """Full JSON serialization including page content."""
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_dict(cls, data: dict) -> "Document":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================
# Tree Node Model (Hierarchical Index)
# ============================================================

@dataclass
class TreeNode:
    """
    A node in the hierarchical document tree index.

    The tree is built by an LLM analyzing the document structure.
    Each node represents a logical section (chapter, section, subsection).

    Attributes:
        id:         Unique node identifier.
        title:      Section/chapter title.
        summary:    LLM-generated summary of what this section contains.
        page_start: Starting page number (1-indexed).
        page_end:   Ending page number (1-indexed, inclusive).
        depth:      Depth in the tree (0 = root).
        children:   Child nodes representing sub-sections.
        content:    The actual text content of this section (leaf nodes only).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    summary: str = ""
    page_start: int = 0
    page_end: int = 0
    depth: int = 0
    children: list = field(default_factory=list)  # List of TreeNode dicts
    content: str = ""

    def to_dict(self) -> dict:
        """Recursively serialize the tree to a dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "summary": self.summary,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "depth": self.depth,
            "children": [
                child.to_dict() if isinstance(child, TreeNode) else child
                for child in self.children
            ],
            "content": self.content,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TreeNode":
        """Recursively deserialize a tree from a dictionary."""
        children_data = data.get("children", [])
        children = [cls.from_dict(c) for c in children_data]
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            page_start=data.get("page_start", 0),
            page_end=data.get("page_end", 0),
            depth=data.get("depth", 0),
            children=children,
            content=data.get("content", ""),
        )

    def is_leaf(self) -> bool:
        """Check if this node has no children (leaf node)."""
        return len(self.children) == 0

    def get_node_count(self) -> int:
        """Count total nodes in this subtree (including self)."""
        count = 1
        for child in self.children:
            if isinstance(child, TreeNode):
                count += child.get_node_count()
        return count


# ============================================================
# Traversal Step (For traceability)
# ============================================================

@dataclass
class TraversalStep:
    """
    Records a single step in the LLM's tree navigation.

    Used to build a transparent audit trail of how the system
    arrived at its answer.
    """
    node_id: str = ""
    node_title: str = ""
    action: str = ""   # "enter" | "backtrack" | "select"
    reasoning: str = ""
    depth: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================
# Query Result
# ============================================================

@dataclass
class QueryResult:
    """
    The final result returned to the user after a query.

    Attributes:
        answer:          The LLM-generated answer.
        sources:         List of source references (page numbers, section titles).
        traversal_path:  Step-by-step record of the tree navigation.
        context_used:    The exact text block fed to the LLM for final answer.
        confidence:      Self-assessed confidence from the LLM (if available).
        query:           The original user query.
        document_id:     The document that was queried.
    """
    answer: str = ""
    sources: list = field(default_factory=list)
    traversal_path: list = field(default_factory=list)  # List of TraversalStep dicts
    context_used: str = ""
    confidence: str = "medium"
    query: str = ""
    document_id: str = ""
    timing_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "sources": self.sources,
            "traversal_path": [
                step.to_dict() if isinstance(step, TraversalStep) else step
                for step in self.traversal_path
            ],
            "context_used": self.context_used[:500] + "..." if len(self.context_used) > 500 else self.context_used,
            "confidence": self.confidence,
            "query": self.query,
            "document_id": self.document_id,
            "timing_ms": round(self.timing_ms, 2),
        }


# ============================================================
# Traditional RAG: Chunk Model
# ============================================================

@dataclass
class Chunk:
    """
    Represents a fixed-size text chunk in the Traditional RAG pipeline.

    Attributes:
        id:           Unique chunk identifier.
        text:         The text content of the chunk.
        document_id:  Identifier of the parent document.
        page_number:  1-indexed page number this chunk came from.
        start_pos:    Character start position in the original page/doc.
        end_pos:      Character end position in the original page/doc.
        chunk_index:  Sequence index of this chunk in the document.
        metadata:     Arbitrary additional metadata (e.g. headers, word count).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    text: str = ""
    document_id: str = ""
    page_number: int = 1
    start_pos: int = 0
    end_pos: int = 0
    chunk_index: int = 0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


# ============================================================
# Traditional RAG: Query Result
# ============================================================

@dataclass
class TraditionalQueryResult:
    """
    Result returned by the Traditional RAG pipeline.

    Attributes:
        answer:           The LLM-generated answer.
        retrieved_chunks: List of retrieved chunks with similarity scores.
        context_used:     The concatenated text of retrieved chunks sent to the LLM.
        confidence:       Self-assessed confidence level (HIGH/MEDIUM/LOW).
        query:            The original user question.
        document_id:      The document that was queried.
        timing_ms:        Total processing time in milliseconds.
    """
    answer: str = ""
    retrieved_chunks: list = field(default_factory=list)
    context_used: str = ""
    confidence: str = "medium"
    query: str = ""
    document_id: str = ""
    timing_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "retrieved_chunks": self.retrieved_chunks,
            "context_used": self.context_used[:500] + "..." if len(self.context_used) > 500 else self.context_used,
            "confidence": self.confidence,
            "query": self.query,
            "document_id": self.document_id,
            "timing_ms": round(self.timing_ms, 2),
        }


# ============================================================
# Comparison Result
# ============================================================

@dataclass
class ComparisonResult:
    """
    Side-by-side comparison of Traditional RAG vs Vectorless RAG.

    Attributes:
        query:                The user's query.
        document_id:          The document queried.
        traditional:          TraditionalQueryResult as dict or object.
        vectorless:           QueryResult as dict or object.
        traditional_time_ms:  Total runtime for traditional pipeline in ms.
        vectorless_time_ms:   Total runtime for vectorless pipeline in ms.
    """
    query: str = ""
    document_id: str = ""
    traditional: dict = field(default_factory=dict)
    vectorless: dict = field(default_factory=dict)
    traditional_time_ms: float = 0.0
    vectorless_time_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "document_id": self.document_id,
            "traditional": self.traditional if isinstance(self.traditional, dict) else self.traditional.to_dict(),
            "vectorless": self.vectorless if isinstance(self.vectorless, dict) else self.vectorless.to_dict(),
            "traditional_time_ms": round(self.traditional_time_ms, 2),
            "vectorless_time_ms": round(self.vectorless_time_ms, 2),
        }
