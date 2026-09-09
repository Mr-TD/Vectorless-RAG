"""
Vectorless RAG — Index Service
==================================
Builds the Hierarchical Tree Index from a document using LLM reasoning.

This is the HEART of the Vectorless RAG approach:
    Instead of chunking and embedding, we use the LLM to analyze
    the document structure and build an intelligent "table of contents"
    tree. Each node contains a title, summary, page range, and content.

The tree is then serialized to JSON and saved to disk for later
traversal by the QueryService.
"""

import os
import json
from flask import current_app
from app.models.document import Document, TreeNode
from app.services.llm_service import LLMService
from app.services.document_service import DocumentService
from app.utils.logger import get_logger, log_performance
from app.utils.exceptions import IndexBuildError, DocumentNotFoundError

logger = get_logger(__name__)


class IndexService:
    """
    Builds and manages hierarchical tree indexes for documents.

    Uses the LLM to analyze document structure and create a
    navigable tree that represents the document's logical organization.
    """

    def __init__(self, app=None):
        self.app = app
        self.llm = LLMService(app)
        self.doc_service = DocumentService(app)

    def _get_index_folder(self):
        app = self.app or current_app
        return app.config["INDEX_FOLDER"]

    def _get_index_path(self, doc_id):
        return os.path.join(self._get_index_folder(), f"{doc_id}.json")

    @log_performance
    def build_index(self, doc_id):
        """
        Build a hierarchical tree index for a document.

        This is a multi-step process:
        1. Load the document and its page content
        2. Send the content to the LLM with tree-building instructions
        3. Parse the LLM's response into a TreeNode structure
        4. Attach actual page content to leaf nodes
        5. Save the tree to disk

        Args:
            doc_id: The document's unique identifier.

        Returns:
            The root TreeNode of the built tree.

        Raises:
            IndexBuildError: If the tree cannot be constructed.
            DocumentNotFoundError: If the document doesn't exist.
        """
        # Step 1: Load the document
        doc = self.doc_service.get_document(doc_id)
        if doc.status == "indexed":
            logger.info(f"Document {doc_id} is already indexed. Rebuilding...")

        self.doc_service.update_status(doc_id, "indexing")
        logger.info(f"Building index for '{doc.filename}' ({doc.page_count} pages)")

        try:
            # Step 2: Build page summary for the LLM
            page_summaries = self._build_page_overview(doc)

            # Step 3: Ask the LLM to create the tree structure
            tree_data = self._generate_tree_structure(doc.filename, page_summaries, doc.page_count)

            # Step 4: Parse into TreeNode objects and attach content
            root = TreeNode.from_dict(tree_data)
            self._attach_content(root, doc.pages)

            # Step 5: Save to disk
            self._save_index(doc_id, root)

            # Update document status
            self.doc_service.update_status(doc_id, "indexed")

            node_count = root.get_node_count()
            logger.info(
                f"Index built successfully for '{doc.filename}': "
                f"{node_count} nodes in tree"
            )
            return root

        except IndexBuildError:
            self.doc_service.update_status(doc_id, "error", "Index build failed.")
            raise
        except Exception as e:
            self.doc_service.update_status(doc_id, "error", str(e))
            raise IndexBuildError(f"Failed to build index for '{doc.filename}': {e}")

    def get_index(self, doc_id):
        """
        Load a previously built tree index from disk.

        Args:
            doc_id: The document's unique identifier.

        Returns:
            The root TreeNode.

        Raises:
            DocumentNotFoundError: If no index exists for this document.
        """
        index_path = self._get_index_path(doc_id)
        if not os.path.exists(index_path):
            raise DocumentNotFoundError(
                f"No index found for document '{doc_id}'. "
                f"Please build the index first."
            )

        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return TreeNode.from_dict(data)

    def has_index(self, doc_id):
        """Check if a document has an existing index."""
        return os.path.exists(self._get_index_path(doc_id))

    # ================================================================
    # Private: Tree Building Methods
    # ================================================================

    def _build_page_overview(self, doc):
        """
        Create a condensed overview of all pages for the LLM to analyze.

        For each page, we send a preview (first ~500 chars) so the LLM
        can understand the document structure without exceeding context limits.
        """
        overview_parts = []
        for i, page_text in enumerate(doc.pages):
            page_num = i + 1
            # Take first 500 chars as a preview
            preview = page_text[:500].strip()
            if len(page_text) > 500:
                preview += "..."
            overview_parts.append(f"--- PAGE {page_num} ---\n{preview}")

        return "\n\n".join(overview_parts)

    def _generate_tree_structure(self, filename, page_summaries, page_count):
        """
        Use the LLM to analyze the document and produce a tree structure.

        The LLM reads the page previews and identifies the logical hierarchy:
        chapters, sections, subsections, etc.

        Returns:
            A dictionary representing the tree structure (JSON-parseable).
        """
        system_prompt = """You are a document structure analyzer. Your job is to read the page previews of a document and create a hierarchical tree index that represents the document's logical structure.

RULES:
1. Create a tree with a root node (the document title) and children representing major sections.
2. Each section can have sub-sections as children, creating a multi-level hierarchy.
3. Every node MUST have: title, summary, page_start, page_end, depth.
4. The summary should be 1-2 sentences describing what that section covers.
5. page_start and page_end are 1-indexed page numbers (inclusive).
6. depth is 0 for root, 1 for top-level sections, 2 for sub-sections, etc.
7. Leaf nodes (no children) should cover specific, narrow topics.
8. Ensure page ranges don't have gaps — every page should be covered.
9. Maximum tree depth: 4 levels (root + 3 levels of sections).
10. Respond with ONLY valid JSON. No explanations outside JSON."""

        user_prompt = f"""Analyze this document and create a hierarchical tree index.

DOCUMENT: {filename}
TOTAL PAGES: {page_count}

PAGE PREVIEWS:
{page_summaries}

Return a JSON object with this structure:
{{
    "title": "Document Title",
    "summary": "Brief description of the entire document",
    "page_start": 1,
    "page_end": {page_count},
    "depth": 0,
    "children": [
        {{
            "title": "Section Title",
            "summary": "What this section covers",
            "page_start": 1,
            "page_end": 3,
            "depth": 1,
            "children": [
                {{
                    "title": "Subsection Title",
                    "summary": "What this subsection covers",
                    "page_start": 1,
                    "page_end": 2,
                    "depth": 2,
                    "children": []
                }}
            ]
        }}
    ]
}}"""

        try:
            tree_data = self.llm.generate_json(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=8192,
            )

            # Validate the tree structure
            self._validate_tree_data(tree_data, page_count)
            return tree_data

        except Exception as e:
            logger.error(f"Tree generation failed: {e}")
            # Fallback: Create a simple flat tree
            return self._build_fallback_tree(filename, page_count)

    def _validate_tree_data(self, tree_data, page_count):
        """
        Validate the LLM-generated tree structure.

        Checks:
        - Required fields are present
        - Page ranges are valid
        - Structure is consistent
        """
        if not isinstance(tree_data, dict):
            raise IndexBuildError("LLM returned non-dict tree structure.")

        required_fields = ["title", "summary", "page_start", "page_end"]
        for field in required_fields:
            if field not in tree_data:
                raise IndexBuildError(f"Tree root missing required field: {field}")

        # Basic range validation
        if tree_data.get("page_start", 0) < 1:
            tree_data["page_start"] = 1
        if tree_data.get("page_end", 0) > page_count:
            tree_data["page_end"] = page_count

        logger.debug(f"Tree validation passed: root='{tree_data.get('title', 'Unknown')}'")

    def _build_fallback_tree(self, filename, page_count):
        """
        Build a simple fallback tree when LLM tree generation fails.

        Creates a flat structure with one child per page.
        """
        logger.warning("Using fallback flat tree structure.")

        children = []
        for i in range(1, page_count + 1):
            children.append({
                "title": f"Page {i}",
                "summary": f"Content from page {i} of the document.",
                "page_start": i,
                "page_end": i,
                "depth": 1,
                "children": [],
            })

        return {
            "title": filename,
            "summary": f"Document '{filename}' with {page_count} pages.",
            "page_start": 1,
            "page_end": page_count,
            "depth": 0,
            "children": children,
        }

    def _attach_content(self, node, pages):
        """
        Recursively attach the actual page content to tree nodes.

        Leaf nodes get the full text of their page range.
        Internal nodes get a concatenation of their children's content
        (useful if traversal stops at a non-leaf node).
        """
        if node.is_leaf():
            # Attach the actual text content from the document pages
            start = max(0, node.page_start - 1)  # Convert to 0-indexed
            end = min(len(pages), node.page_end)
            content_parts = pages[start:end]
            node.content = "\n\n".join(content_parts)
        else:
            # Recursively process children
            for child in node.children:
                if isinstance(child, TreeNode):
                    self._attach_content(child, pages)

    def _save_index(self, doc_id, root):
        """Save the tree index to disk as JSON."""
        index_path = self._get_index_path(doc_id)
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(root.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info(f"Index saved to: {index_path}")
