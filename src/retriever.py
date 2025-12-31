"""
Retriever Module
"""

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import os


class Retriever:
    """FAISS vector store for document retrieval"""

    def __init__(self, embedding_model="text-embedding-3-small", top_k=6, vector_store_path=None):
        self.top_k = top_k
        self.vector_store_path = vector_store_path or "data/vector_store"
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        self.db = None

    def create_vector_store(self, chunks):
        """Build FAISS index from chunks"""
        self.db = FAISS.from_documents(chunks, self.embeddings)
        self.save()

    def save(self, path=None):
        """Save vector store to disk"""
        path = path or self.vector_store_path
        if self.db:
            os.makedirs(path, exist_ok=True)
            self.db.save_local(path)

    def load_vector_store(self):
        """Load vector store from disk"""
        path = self.vector_store_path

        # Check if path exists
        if not os.path.exists(path):
            return False

        # Check FAISS index file
        index_file = os.path.join(path, "index.faiss")
        if not os.path.exists(index_file):
            return False

        try:
            self.db = FAISS.load_local(
                path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception:
            return False

    def retrieve(self, question):
        """Get relevant chunks for a question"""
        retriever = self.db.as_retriever(search_kwargs={"k": self.top_k})
        return retriever.invoke(question)
