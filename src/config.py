"""
Configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")  
    
    # LLM Provider Settings 
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # Default: openai
    
    # Model Settings
    EMBEDDING_MODEL = "text-embedding-3-small"
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")  # Can be overridden in .env
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
    
    # Retrieval Settings
    CHUNK_SIZE = 900
    CHUNK_OVERLAP = 150
    TOP_K = 6
    
    # Paths
    PDF_FOLDER = "data/pdfs"
    VECTOR_STORE_PATH = "data/vector_store"
    
    # PDF Files
    PDF_FILES = [
        "KSSC_General_Policies.pdf",
        "KSSC_HR_Policies.pdf",
        "KSSC_Financial_Policies.pdf"
    ]
    
    # App Settings
    PAGE_TITLE = "مساعد مركز الملك سلمان الاجتماعي"
    LAYOUT = "wide"
    
    # Cache
    ENABLE_CACHE = True
    
    @classmethod
    def get_pdf_paths(cls):
        return [os.path.join(cls.PDF_FOLDER, pdf) for pdf in cls.PDF_FILES]
    
    @classmethod
    def setup(cls):
        """Setup environment - supports both OpenAI and Groq"""
        
        # OpenAI is always required (for embeddings)
        if not cls.OPENAI_API_KEY:
            raise ValueError("❌ OPENAI_API_KEY not found in .env file!")
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        
        # Check LLM provider
        if cls.LLM_PROVIDER == "openai":
            # Using OpenAI for generation
            pass
            
        elif cls.LLM_PROVIDER == "groq":
            # Using Groq for generation
            if not cls.GROQ_API_KEY:
                print("⚠️  Warning: GROQ_API_KEY not found. Falling back to OpenAI.")
                cls.LLM_PROVIDER = "openai"
            else:
                os.environ["GROQ_API_KEY"] = cls.GROQ_API_KEY
        
        else:
            print(f"⚠️  Warning: Unknown LLM_PROVIDER '{cls.LLM_PROVIDER}'. Using OpenAI.")
            cls.LLM_PROVIDER = "openai"