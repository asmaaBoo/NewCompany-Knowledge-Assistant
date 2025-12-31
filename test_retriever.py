"""
Unit Tests for Retriever
"""

import unittest
import os
import tempfile
import shutil
from src.config import Config

from src.retriever import Retriever
from langchain_core.documents import Document


class TestRetriever(unittest.TestCase):
    """
    Unit tests for the Retriever component.
    Focuses on vector store creation, persistence, and retrieval behavior.
    """

    def setUp(self):
        """
        Runs before each test.
        Creates a temporary vector store and sample documents.
        """
        self.temp_dir = tempfile.mkdtemp()
        Config.setup()
        self.retriever = Retriever(
            top_k=3,
            vector_store_path=self.temp_dir
        )

        self.test_docs = self._create_test_documents()

    def tearDown(self):
        """
        Runs after each test.
        Cleans up temporary files.
        """
        shutil.rmtree(self.temp_dir)

    def _create_test_documents(self, num_docs=5):
        """
        Generate simple in-memory documents for testing.
        (No PDFs or external files involved)
        """
        return [
            Document(
                page_content=f"Test document {i} about retrieval.",
                metadata={"id": i}
            )
            for i in range(num_docs)
        ]

    # ---------- Core Tests ----------

    def test_initialization(self):
        """Retriever initializes with correct defaults"""
        self.assertEqual(self.retriever.top_k, 3)
        self.assertIsNotNone(self.retriever.embeddings)
        self.assertIsNone(self.retriever.db)

    def test_create_vector_store(self):
        """Vector store is created successfully"""
        self.retriever.create_vector_store(self.test_docs)
        self.assertIsNotNone(self.retriever.db)

    def test_save_and_load_vector_store(self):
        """Vector store can be saved and loaded"""
        self.retriever.create_vector_store(self.test_docs)
        self.retriever.save()

        new_retriever = Retriever(vector_store_path=self.temp_dir)
        self.assertTrue(new_retriever.load_vector_store())
        self.assertIsNotNone(new_retriever.db)

    def test_retrieve_documents(self):
        """Retriever returns relevant documents"""
        self.retriever.create_vector_store(self.test_docs)

        results = self.retriever.retrieve("retrieval")
        self.assertGreater(len(results), 0)
        self.assertLessEqual(len(results), self.retriever.top_k)

    def test_top_k_limit(self):
        """Retriever respects top_k limit"""
        retriever = Retriever(top_k=2, vector_store_path=self.temp_dir)
        retriever.create_vector_store(self._create_test_documents(10))

        results = retriever.retrieve("document")
        self.assertLessEqual(len(results), 2)

    def test_load_missing_store(self):
        """Loading a missing vector store fails safely"""
        retriever = Retriever(vector_store_path="/invalid/path")
        self.assertFalse(retriever.load_vector_store())
        self.assertIsNone(retriever.db)

    def test_retrieval_relevance(self):
        """Retriever favors semantically relevant content"""
        docs = [
            Document(page_content="Python programming", metadata={}),
            Document(page_content="Web development", metadata={}),
            Document(page_content="Python data science", metadata={})
        ]

        self.retriever.create_vector_store(docs)
        results = self.retriever.retrieve("Python")

        self.assertTrue(
            any("Python" in doc.page_content for doc in results)
        )


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRetriever)
    result = unittest.TextTestRunner(verbosity=0).run(suite)

    total = result.testsRun
    failed = len(result.failures) + len(result.errors)
    passed = total - failed

    print("\n" + "=" * 40)
    print(" Retriever Unit Test Summary")
    print("=" * 40)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {failed}")
    print("=" * 40)

    exit(0 if result.wasSuccessful() else 1)
