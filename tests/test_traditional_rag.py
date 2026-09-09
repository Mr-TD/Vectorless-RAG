"""
Unit tests for Traditional RAG and Comparison components.
"""

import os
import shutil
import tempfile
import unittest
import numpy as np

from app.models.document import Document, Chunk, TraditionalQueryResult, ComparisonResult, QueryResult
from app.services.traditional_rag.chunker import Chunker
from app.services.traditional_rag.vector_store import VectorStore
from app.services.compare_service import CompareService
from app import create_app


class TestTraditionalRAG(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.app = create_app("testing")
        self.app.config["TRADITIONAL_INDEX_FOLDER"] = self.temp_dir

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_chunker_basic(self):
        chunker = Chunker(chunk_size=100, overlap=20)
        text = "This is a test document to verify that the chunking mechanism in traditional RAG properly splits long text into overlapping chunks."
        chunks = chunker.chunk_text(text, page_number=1, document_id="doc-1")

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[0].document_id, "doc-1")
        self.assertEqual(chunks[0].chunk_index, 0)
        self.assertEqual(chunks[1].chunk_index, 1)

    def test_chunker_document(self):
        chunker = Chunker(chunk_size=80, overlap=15)
        doc = Document(
            id="test-doc-123",
            filename="sample.pdf",
            pages=["First page content is here.", "Second page content has more details."]
        )
        chunks = chunker.chunk_document(doc)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].page_number, 1)
        self.assertEqual(chunks[1].page_number, 2)

    def test_vector_store_cosine_similarity(self):
        store = VectorStore()
        c1 = Chunk(text="Chunk 1 about artificial intelligence", chunk_index=0)
        c2 = Chunk(text="Chunk 2 about deep ocean marine biology", chunk_index=1)
        c3 = Chunk(text="Chunk 3 about neural networks and machine learning", chunk_index=2)

        # Mock embeddings (3 chunks, 4 dimensions)
        # c1 and c3 are closer in vector space
        e1 = [1.0, 0.9, 0.0, 0.0]
        e2 = [0.0, 0.0, 1.0, 0.8]
        e3 = [0.9, 1.0, 0.1, 0.0]

        store.add([c1, c2, c3], [e1, e2, e3])
        self.assertEqual(store.count(), 3)

        # Query vector close to AI
        query_vec = [1.0, 0.8, 0.0, 0.0]
        results = store.search(query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        # Top 1 should be c1
        self.assertEqual(results[0][0].chunk_index, 0)
        self.assertGreater(results[0][1], 0.9)
        # Top 2 should be c3
        self.assertEqual(results[1][0].chunk_index, 2)

    def test_vector_store_save_and_load(self):
        store = VectorStore()
        c1 = Chunk(text="Sample text 1", chunk_index=0)
        c2 = Chunk(text="Sample text 2", chunk_index=1)
        e1 = [0.5, 0.5]
        e2 = [0.1, 0.9]

        store.add([c1, c2], [e1, e2])

        base_path = os.path.join(self.temp_dir, "test_index")
        store.save(base_path)

        self.assertTrue(os.path.exists(f"{base_path}.npz"))
        self.assertTrue(os.path.exists(f"{base_path}.json"))

        loaded_store = VectorStore()
        loaded = loaded_store.load(base_path)
        self.assertTrue(loaded)
        self.assertEqual(loaded_store.count(), 2)
        self.assertEqual(loaded_store.chunks[0].text, "Sample text 1")

    def test_flask_routes_exist(self):
        client = self.app.test_client()

        # Health endpoint
        res = client.get("/api/health")
        self.assertIn(res.status_code, [200, 503])

        # Documents endpoint
        res = client.get("/api/documents")
        self.assertEqual(res.status_code, 200)

        # Compare page route (for a doc ID)
        res = client.get("/compare/non-existent-doc")
        self.assertEqual(res.status_code, 200)


if __name__ == "__main__":
    unittest.main()
