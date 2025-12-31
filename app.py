"""
مساعد مركز الملك سلمان الاجتماعي
نظام ذكي للإجابة على استفساراتك حول سياسات المركز
"""

import streamlit as st
from src.config import Config
from src.document_processor import DocumentProcessor
from src.retriever import Retriever
from src.generator import Generator
from src.rag_pipeline import RAGPipeline
from src.utils import create_directories, check_pdf_files

# Page Configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    layout=Config.LAYOUT
)

# Custom CSS
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .main {
        background: linear-gradient(135deg, #1976D2 0%, #1565C0 100%);
        padding: 2rem;
    }
    
    .block-container {
        background: white;
        border-radius: 20px;
        padding: 3rem 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.15);
    }
    
    /* User message - Yellow - RIGHT side */
    .stChatMessage[data-testid*="user"] {
        background-color: #FFF9C4 !important;
        padding: 15px !important;
        border-radius: 15px !important;
        margin: 10px 0 !important;
        margin-right: 20% !important;
        text-align: right !important;
        direction: rtl !important;
    }
    
    .stChatMessage[data-testid*="user"] * {
        text-align: right !important;
        direction: rtl !important;
    }
    
    /* Assistant message - Blue - LEFT side */
    .stChatMessage[data-testid*="assistant"] {
        background-color: #E3F2FD !important;
        padding: 15px !important;
        border-radius: 15px !important;
        margin: 10px 0 !important;
        margin-left: 20% !important;
        text-align: right !important;
        direction: rtl !important;
    }
    
    .stChatMessage[data-testid*="assistant"] * {
        text-align: right !important;
        direction: rtl !important;
    }
    
    /* Chat input - Yellow background! */
    .stChatInput {
        border-radius: 15px !important;
        border: 2px solid #FFC107 !important;
    }
    
    .stChatInput > div {
        background: linear-gradient(135deg, #FFFDE7 0%, #FFF9C4 100%) !important;
        border-radius: 13px !important;
    }
    
    .stChatInput textarea {
        direction: rtl !important;
        text-align: right !important;
        background: transparent !important;
        color: #1A1A1A !important;
    }
    
    .stChatInput textarea::placeholder {
        color: #666 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D47A1 0%, #1565C0 100%);
        padding: 2rem 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    [data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.2);
    }
</style>
""", unsafe_allow_html=True)


def initialize_pipeline():
    """Initialize RAG pipeline"""
    Config.setup()
    create_directories()
    
    retriever = Retriever(
        embedding_model=Config.EMBEDDING_MODEL,
        top_k=Config.TOP_K,
        vector_store_path=Config.VECTOR_STORE_PATH
    )
    
    generator = Generator(
        model=Config.LLM_MODEL,
        temperature=Config.LLM_TEMPERATURE
    )
    
    # Load or build vector store
    if not retriever.load_vector_store():
        pdf_paths = Config.get_pdf_paths()
        if not check_pdf_files(pdf_paths):
            st.error("❌ ملفات PDF غير موجودة في data/pdfs/")
            st.stop()
        
        processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )
        chunks = processor.process_pdfs(pdf_paths)
        retriever.create_vector_store(chunks)
    
    return RAGPipeline(retriever, generator, enable_cache=Config.ENABLE_CACHE)


# Initialize
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'pipeline' not in st.session_state:
    with st.spinner("جاري التهيئة..."):
        st.session_state.pipeline = initialize_pipeline()


# Sidebar
with st.sidebar:
    st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h2 style='margin: 0; font-size: 1.4rem;'>مركز الملك سلمان الاجتماعي</h2>
        <p style='font-size: 0.85rem; opacity: 0.8; margin-top: 0.5rem;'>مساعد ذكي للاستفسارات</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("""
    <h4 style='text-align: right; direction: rtl;'>أسئلة سريعة</h4>
    """, unsafe_allow_html=True)
    
    questions = [
        "كم مدة الإجازة السنوية؟",
        "ما سياسة الاستقطاب والتوظيف؟",
        "ما مدة فترة الاختبار؟",
        "ما الموارد المالية للمركز؟"
    ]
    
    for q in questions:
        if st.button(q, key=f"q_{q}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            result = st.session_state.pipeline.query(q)
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
            st.rerun()
    
    st.divider()
    
    if st.button("محادثة جديدة", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Main
st.markdown("""
<div style='text-align: center; padding: 1.5rem 0;'>
    <h1>مساعد مركز الملك سلمان الاجتماعي</h1>
    <p style='color: #666; font-size: 1.1rem;'>اسألني عن سياسات المركز وأنظمته</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input with yellow background
if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.spinner("جاري البحث..."):
        result = st.session_state.pipeline.query(prompt)
    
    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
    
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
    
    st.rerun()