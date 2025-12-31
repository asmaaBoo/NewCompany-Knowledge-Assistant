"""
Utility Functions -
"""

import os
import streamlit as st


def create_directories():
    """Create data folders"""
    os.makedirs("data/pdfs", exist_ok=True)
    os.makedirs("data/vector_store", exist_ok=True)


def check_pdf_files(pdf_paths):
    """Check if PDFs exist"""
    missing = [p for p in pdf_paths if not os.path.exists(p)]
    
    if missing:
        st.error("❌ ملفات PDF غير موجودة في data/pdfs/")
        return False
    return True


def format_time(seconds):
    """Format time: 0.5s → 500ms"""
    return f"{seconds*1000:.0f}ms" if seconds < 1 else f"{seconds:.1f}s"


def show_contexts(contexts):
    """Show retrieved chunks"""
    with st.expander(f"📄 السياقات المسترجعة ({len(contexts)})"):
        for i, text in enumerate(contexts, 1):
            st.text_area(f"مقتطف {i}", text, height=100, disabled=True)