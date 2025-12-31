"""
Document Processing Module - 
"""

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import re


class DocumentProcessor:
    """Load PDFs and split into chunks"""
    
    def __init__(self, chunk_size=900, chunk_overlap=150):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
    
    def process_pdfs(self, pdf_paths):
        """Load PDFs → Clean → Chunk"""
        # Load all PDFs
        all_docs = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            all_docs.extend(docs)
        
        # Clean text
        for doc in all_docs:
            doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
        
        # Split into chunks
        chunks = self.splitter.split_documents(all_docs)
        
        return chunks