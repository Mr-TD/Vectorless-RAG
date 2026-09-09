"""
Traditional RAG — Document Chunker
====================================
Splits documents into fixed-size overlapping text chunks.
Preserves page number, character offsets, and metadata.
"""

from typing import List
from flask import current_app
from app.models.document import Document, Chunk
from app.utils.logger import get_logger

logger = get_logger(__name__)


class Chunker:
    """
    Splits text documents into overlapping chunks for vector embedding.

    Traditional RAG relies on slicing text into small uniform windows
    (e.g., 500 characters with 100 character overlap) so each piece can
    be converted into a single dense vector.
    """

    def __init__(self, chunk_size: int = None, overlap: int = None, app=None):
        self.app = app
        app_ctx = app or (current_app._get_current_object() if current_app else None)

        if chunk_size is not None:
            self.chunk_size = chunk_size
        elif app_ctx and "CHUNK_SIZE" in app_ctx.config:
            self.chunk_size = int(app_ctx.config["CHUNK_SIZE"])
        else:
            self.chunk_size = 500

        if overlap is not None:
            self.overlap = overlap
        elif app_ctx and "CHUNK_OVERLAP" in app_ctx.config:
            self.overlap = int(app_ctx.config["CHUNK_OVERLAP"])
        else:
            self.overlap = 100

        if self.overlap >= self.chunk_size:
            raise ValueError(
                f"Overlap ({self.overlap}) must be strictly less than chunk_size ({self.chunk_size})."
            )

    def chunk_document(self, doc: Document) -> List[Chunk]:
        """
        Split an entire document's pages into chunks.

        Args:
            doc: Document dataclass instance containing pages.

        Returns:
            List of Chunk objects.
        """
        all_chunks: List[Chunk] = []
        chunk_idx = 0

        for page_num_0, page_text in enumerate(doc.pages):
            page_number = page_num_0 + 1
            if not page_text or not page_text.strip():
                continue

            page_chunks = self.chunk_text(
                text=page_text,
                page_number=page_number,
                document_id=doc.id,
                start_index=chunk_idx,
            )
            all_chunks.extend(page_chunks)
            chunk_idx += len(page_chunks)

        logger.info(
            f"Chunked document '{doc.filename}' ({doc.page_count} pages) "
            f"into {len(all_chunks)} chunks (size={self.chunk_size}, overlap={self.overlap})"
        )
        return all_chunks

    def chunk_text(
        self, text: str, page_number: int = 1, document_id: str = "", start_index: int = 0
    ) -> List[Chunk]:
        """
        Split a single text string into overlapping chunks.
        Tries to break at whitespace near the target boundary to avoid cutting words.

        Args:
            text: Text to split.
            page_number: 1-indexed page number.
            document_id: Parent document ID.
            start_index: Starting sequential chunk index.

        Returns:
            List of Chunk instances.
        """
        if not text or not text.strip():
            return []

        text = text.strip()
        text_len = len(text)
        chunks: List[Chunk] = []
        step = self.chunk_size - self.overlap
        start = 0
        current_idx = start_index

        while start < text_len:
            end = min(start + self.chunk_size, text_len)

            # If not at the end of the text, try to find a natural word boundary
            if end < text_len:
                # Look backwards up to 60 characters for a whitespace or newline
                lookback_limit = max(start + self.overlap, end - 60)
                split_point = -1
                for p in range(end, lookback_limit, -1):
                    if text[p] in ('\n', ' ', '\t', '.', ';'):
                        split_point = p + 1
                        break
                if split_point > start:
                    end = split_point

            chunk_content = text[start:end].strip()
            if chunk_content:
                chunk = Chunk(
                    text=chunk_content,
                    document_id=document_id,
                    page_number=page_number,
                    start_pos=start,
                    end_pos=end,
                    chunk_index=current_idx,
                    metadata={
                        "char_count": len(chunk_content),
                        "word_count": len(chunk_content.split()),
                    },
                )
                chunks.append(chunk)
                current_idx += 1

            if end >= text_len:
                break

            start += step

        return chunks
