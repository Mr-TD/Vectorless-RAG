"""
Vectorless RAG — Query Service
==================================
Implements reasoning-driven tree traversal and answer synthesis.

This is the INTELLIGENCE layer of the Vectorless RAG system.

When a user submits a query:
1. The LLM examines the top-level nodes of the document tree
2. It reasons about which branch is most likely to contain the answer
3. It iteratively narrows focus, descending into sub-sections
4. If a dead end is reached, it BACKTRACKS and tries an alternative path
5. Once the right section is found, the exact text is extracted
6. A final LLM call synthesizes the answer with source citations

Key safety features:
- Maximum traversal depth to prevent infinite loops
- Backtracking capability for dead-end recovery
- Graceful degradation (returns best partial answer if stuck)
"""

import json
from flask import current_app
from app.models.document import TreeNode, TraversalStep, QueryResult
from app.services.llm_service import LLMService
from app.services.vectorless_rag.index_service import IndexService
from app.services.document_service import DocumentService
from app.utils.logger import get_logger, log_performance
from app.utils.exceptions import TraversalError, DocumentNotFoundError

logger = get_logger(__name__)


class QueryService:
    """
    Handles user queries against indexed documents.

    Uses LLM reasoning to traverse the document tree and
    synthesize answers with source attribution.
    """

    def __init__(self, app=None):
        self.app = app
        self.llm = LLMService(app)
        self.index_service = IndexService(app)
        self.doc_service = DocumentService(app)

    def _get_max_depth(self):
        app = self.app or current_app
        return app.config["MAX_TRAVERSAL_DEPTH"]

    @log_performance
    def answer_query(self, query, doc_id):
        """
        Process a user query against an indexed document.

        Orchestrates the full pipeline:
        1. Load the document tree
        2. Traverse the tree using LLM reasoning
        3. Extract targeted context
        4. Synthesize the final answer

        Args:
            query:  The user's natural language question.
            doc_id: The document to query against.

        Returns:
            A QueryResult object with the answer, sources, and traversal path.

        Raises:
            TraversalError: If traversal fails completely.
            DocumentNotFoundError: If the document/index doesn't exist.
        """
        logger.info(f"Processing query: '{query[:100]}...' against document {doc_id}")

        import time
        start_time = time.time()

        # Load the tree and document metadata
        tree = self.index_service.get_index(doc_id)
        doc = self.doc_service.get_document(doc_id)

        # Traverse the tree
        traversal_path, target_node = self._traverse_tree(query, tree)

        # Extract context from the target node
        context = self._extract_context(target_node)

        # Synthesize the final answer
        answer, confidence = self._synthesize_answer(query, context, doc.filename)

        # Build source citations
        sources = self._build_sources(traversal_path, target_node, doc.filename)

        timing_ms = (time.time() - start_time) * 1000

        # Build the result
        result = QueryResult(
            answer=answer,
            sources=sources,
            traversal_path=[step.to_dict() for step in traversal_path],
            context_used=context,
            confidence=confidence,
            query=query,
            document_id=doc_id,
            timing_ms=timing_ms,
        )

        logger.info(
            f"Query answered. Confidence: {confidence}. "
            f"Traversal depth: {len(traversal_path)} steps."
        )

        return result

    # ================================================================
    # Tree Traversal Engine
    # ================================================================

    def _traverse_tree(self, query, root):
        """
        Navigate the document tree using LLM reasoning.

        The LLM acts as a human researcher:
        - Reads the available section titles and summaries
        - Decides which section is most relevant to the query
        - Descends into that section
        - Repeats until a leaf node or max depth is reached
        - Can BACKTRACK if it reaches a dead end

        Args:
            query: The user's question.
            root:  The root TreeNode of the document.

        Returns:
            Tuple of (traversal_path, target_node)
            - traversal_path: List of TraversalStep objects
            - target_node: The TreeNode selected as most relevant
        """
        max_depth = self._get_max_depth()
        traversal_path = []
        current_node = root
        visited_nodes = set()  # Prevent infinite loops
        backtrack_stack = []   # Stack of (node, children_tried) for backtracking

        # Record entering the root
        traversal_path.append(TraversalStep(
            node_id=current_node.id,
            node_title=current_node.title,
            action="enter",
            reasoning="Starting at document root.",
            depth=0,
        ))

        step_count = 0

        while step_count < max_depth:
            step_count += 1
            visited_nodes.add(current_node.id)

            # If this is a leaf node, we've found our target
            if current_node.is_leaf():
                logger.debug(f"Reached leaf node: '{current_node.title}'")
                traversal_path.append(TraversalStep(
                    node_id=current_node.id,
                    node_title=current_node.title,
                    action="select",
                    reasoning="Reached leaf node — this is the target section.",
                    depth=current_node.depth,
                ))
                return traversal_path, current_node

            # Get unvisited children
            available_children = [
                child for child in current_node.children
                if isinstance(child, TreeNode) and child.id not in visited_nodes
            ]

            if not available_children:
                # Dead end: all children visited, try backtracking
                if backtrack_stack:
                    logger.debug(f"Dead end at '{current_node.title}'. Backtracking...")
                    traversal_path.append(TraversalStep(
                        node_id=current_node.id,
                        node_title=current_node.title,
                        action="backtrack",
                        reasoning="All children explored. Backtracking to try another path.",
                        depth=current_node.depth,
                    ))
                    current_node = backtrack_stack.pop()
                    continue
                else:
                    # No backtrack options — use current node
                    logger.warning("Traversal exhausted all paths. Using current node.")
                    break

            # Ask the LLM which child to explore
            chosen_child, reasoning = self._choose_next_node(
                query, current_node, available_children
            )

            if chosen_child is None:
                # LLM couldn't decide — use current node's content
                logger.warning(f"LLM couldn't choose a child at '{current_node.title}'")
                break

            # Record the choice
            traversal_path.append(TraversalStep(
                node_id=chosen_child.id,
                node_title=chosen_child.title,
                action="enter",
                reasoning=reasoning,
                depth=chosen_child.depth,
            ))

            # Push current node to backtrack stack (in case we need to come back)
            backtrack_stack.append(current_node)

            # Descend into the chosen child
            current_node = chosen_child

        if step_count >= max_depth:
            logger.warning(f"Max traversal depth ({max_depth}) reached.")

        return traversal_path, current_node

    def _choose_next_node(self, query, parent_node, children):
        """
        Ask the LLM to decide which child node to explore next.

        Presents the section titles and summaries of the available children,
        and asks the LLM to reason about which one is most likely to contain
        the answer.

        Args:
            query:       The user's question.
            parent_node: The current parent TreeNode.
            children:    List of available child TreeNodes.

        Returns:
            Tuple of (chosen_child, reasoning_text).
            If decision fails, returns (None, "").
        """
        # Build the options list for the LLM
        options = []
        for i, child in enumerate(children):
            options.append(
                f"OPTION {i+1}:\n"
                f"  Title: {child.title}\n"
                f"  Summary: {child.summary}\n"
                f"  Pages: {child.page_start}-{child.page_end}\n"
                f"  Has sub-sections: {'Yes' if not child.is_leaf() else 'No'}"
            )

        options_text = "\n\n".join(options)

        system_prompt = """You are a document navigation agent. Your job is to decide which section of a document is most likely to contain the answer to the user's question.

You must respond with ONLY valid JSON in this exact format:
{
    "choice": <option_number>,
    "reasoning": "<brief explanation of why this section is most relevant>"
}

The choice must be a number between 1 and the number of options.
Be decisive. Pick the single best option."""

        user_prompt = f"""I'm currently in the section: "{parent_node.title}"
Section summary: {parent_node.summary}

The user's question is: "{query}"

Which of these sub-sections should I explore to find the answer?

{options_text}

Choose the most relevant section and explain why."""

        try:
            result = self.llm.generate_json(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                max_tokens=500,
            )

            choice_idx = int(result.get("choice", 1)) - 1
            reasoning = result.get("reasoning", "LLM selected this section.")

            if 0 <= choice_idx < len(children):
                chosen = children[choice_idx]
                logger.debug(
                    f"LLM chose: '{chosen.title}' — {reasoning[:100]}"
                )
                return chosen, reasoning
            else:
                logger.warning(f"LLM returned invalid choice index: {choice_idx}")
                return children[0], "Defaulted to first available section."

        except Exception as e:
            logger.error(f"LLM navigation decision failed: {e}")
            # Fallback: pick the first child
            return children[0], f"Automatic fallback selection (LLM error: {e})"

    # ================================================================
    # Context Extraction
    # ================================================================

    def _extract_context(self, target_node):
        """
        Extract the relevant text content from the target tree node.

        Args:
            target_node: The TreeNode identified by traversal.

        Returns:
            The text content to use for answer generation.
        """
        if target_node.content:
            return target_node.content

        # If the node has no direct content, collect from all leaf descendants
        content_parts = []
        self._collect_leaf_content(target_node, content_parts)

        if content_parts:
            return "\n\n".join(content_parts)

        return f"[No content available for section: {target_node.title}]"

    def _collect_leaf_content(self, node, parts):
        """Recursively collect content from all leaf nodes under this node."""
        if node.is_leaf():
            if node.content:
                parts.append(node.content)
        else:
            for child in node.children:
                if isinstance(child, TreeNode):
                    self._collect_leaf_content(child, parts)

    # ================================================================
    # Answer Synthesis
    # ================================================================

    @log_performance
    def _synthesize_answer(self, query, context, filename):
        """
        Generate the final answer using the targeted context.

        Sends the query and extracted context to the LLM for synthesis.

        Args:
            query:    The user's original question.
            context:  The extracted text from the relevant section.
            filename: The source document name (for citation).

        Returns:
            Tuple of (answer_text, confidence_level).
        """
        # Truncate context if too long (stay within context window)
        max_context_chars = 12000
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n...[truncated]"

        system_prompt = """You are a precise question-answering assistant. You answer questions based ONLY on the provided document context. 

RULES:
1. Answer the question using ONLY information from the provided context.
2. If the context does not contain enough information, say so clearly.
3. Be specific and cite relevant details from the text.
4. At the end of your answer, provide a confidence level: HIGH, MEDIUM, or LOW.
5. Format your response as:
   ANSWER: <your detailed answer>
   CONFIDENCE: <HIGH/MEDIUM/LOW>"""

        user_prompt = f"""QUESTION: {query}

DOCUMENT CONTEXT (from "{filename}"):
---
{context}
---

Answer the question based on the context above."""

        try:
            response = self.llm.generate(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
            )

            # Parse answer and confidence from the response
            answer, confidence = self._parse_answer_response(response)
            return answer, confidence

        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return (
                f"I found a relevant section but encountered an error generating the answer: {e}",
                "low"
            )

    def _parse_answer_response(self, response):
        """
        Parse the LLM's answer response to extract the answer text
        and confidence level.
        """
        response = response.strip()

        # Try to extract structured ANSWER/CONFIDENCE format
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

        # Remove "ANSWER:" prefix if present
        if answer.upper().startswith("ANSWER:"):
            answer = answer[7:].strip()

        return answer, confidence

    # ================================================================
    # Source Attribution
    # ================================================================

    def _build_sources(self, traversal_path, target_node, filename):
        """
        Build source citation information from the traversal.

        Returns a list of source references showing exactly where
        the answer came from.
        """
        sources = []

        # Main source: the target node
        sources.append({
            "document": filename,
            "section": target_node.title,
            "pages": f"{target_node.page_start}-{target_node.page_end}",
            "type": "primary",
        })

        # Add the navigation path as context
        path_sections = [
            step.node_title for step in traversal_path
            if isinstance(step, TraversalStep) and step.action == "enter"
        ]
        if path_sections:
            sources.append({
                "navigation_path": " → ".join(path_sections),
                "type": "traversal_trace",
            })

        return sources
