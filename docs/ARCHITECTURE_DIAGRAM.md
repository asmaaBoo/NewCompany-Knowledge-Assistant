# System Architecture Diagram

 A visual representation of the RAG system architecture with Multi-LLM support.

---

## System Architecture (Bottom-Up)

```
                    ┌──────────────────────┐
                    │   3 PDF Files        │
                    │   (Policies)         │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Chunking (900c)    │
                    │   + Text Cleaning    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Embeddings         │
                    │   (OpenAI API)       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   FAISS Index        │
                    │   (Vector Store)     │
                    └──────────┬───────────┘
                               │
                               │ Retrieval
                               ▼
                    ┌──────────────────────┐
                    │   RAG Pipeline       │
                    │   (Orchestrator)     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Multi-LLM          │
                    │   (OpenAI or Groq)   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   User Interface     │
                    │   (Streamlit App)    │
                    └──────────────────────┘
```

---

## Query Flow (Step by Step)

```
Step 1: User asks question
        │
        ▼
Step 2: Check Cache (Redis)
        │
        ├─→ Found? → Return answer (Fast!)
        │
        └─→ Not found? Continue ↓
                │
                ▼
Step 3: Retrieve Top-6 chunks from FAISS
        │
        ▼
Step 4: Generate answer with LLM
        │   (OpenAI GPT-4o-mini or Groq Llama-3.3-70b)
        │
        ▼
Step 5: RAG Pipeline checks answer automatically
        │   (Looks for: "لا تتوفر إجابة", "غير واضح", etc.)
        │
        ├─→ Clear? → Cache (Redis) & Return answer
        │
        └─→ Unclear? Continue ↓
                │
                ▼
Step 6: Search web (DuckDuckGo)
        │
        ▼
Step 7: Combine PDF context + Web results
        │
        ▼
Step 8: Regenerate answer with LLM
        │
        ▼
Step 9: Cache (Redis) & Return to user
```

**Note:** The system automatically detects unclear answers by checking for specific Arabic phrases in the LLM response.

---

## Multi-LLM Architecture

```
┌────────────────────────────────────────────────┐
│              Generator Module                  │
│                                                │
│  ┌──────────────┐         ┌──────────────┐     │
│  │   Provider   │         │   Provider   │     │
│  │   Selector   │ ------> │   Selector   │     │
│  └──────┬───────┘         └──────┬───────┘     │
│         │                        │             │
│         ▼                        ▼             │
│  ┌──────────────┐         ┌──────────────┐     │
│  │   OpenAI     │         │    Groq      │     │
│  │ GPT-4o-mini  │         │  Llama-3.3   │     │
│  │              │         │     70b      │     │
│  │ $0.15/1M     │         │    Free tier │     │
│  │ 2-3s query   │         │  0.5-1s      │     │
│  └──────────────┘         └──────────────┘     │
│                                                │
└────────────────────────────────────────────────┘
```



---

## Detailed Component Flow

```
┌─────────────────────────────────────────────────────────┐
│                    Data Preparation                     │
└─────────────────────────────────────────────────────────┘

┌─────────┐     ┌──────────┐     ┌──────────┐     ┌─────────┐
│   PDF   │ --> │ Chunking │ --> │Embedding │ --> │  FAISS  │
│  Files  │     │   900c   │     │  OpenAI  │     │  Index  │
└─────────┘     └──────────┘     └──────────┘     └─────────┘

┌─────────────────────────────────────────────────────────┐
│                    Query Processing                     │
└─────────────────────────────────────────────────────────┘

                    ┌─────────┐
                    │Question │
                    └────┬────┘
                         │
                         ▼
                    ┌─────────┐
                    │  Cache? │
                    └────┬────┘
                         │
                    No   │
                         ▼
                    ┌──────────┐
                    │ Retrieve │
                    │  Top-6   │
                    └────┬─────┘
                         │
                         ▼
                    ┌──────────┐
                    │Multi-LLM │
                    │ Generate │
                    │(GPT/Llama)
                    └────┬─────┘
                         │
                         ▼
                    ┌──────────┐
                    │  Clear?  │
                    └────┬─────┘
                         │
                    No   │
                         ▼
                    ┌──────────┐
                    │   Web    │
                    │  Search  │
                    │  (DDGS)  │
                    └────┬─────┘
                         │
                         ▼
                    ┌──────────┐
                    │ Re-Gen   │
                    │Multi-LLM │
                    └────┬─────┘
                         │
                         ▼
                    ┌──────────┐
                    │  Answer  │
                    └──────────┘
```

---

## Components Overview

| Component | Technology | Purpose | Details |
|-----------|-----------|---------|---------|
| **PDF Loader** | PyPDF | Load PDF documents | Extracts text from 3 policy files |
| **Text Chunking** | RecursiveCharacterTextSplitter | Split into chunks | 900 chars per chunk, 150 overlap |
| **Embeddings** | text-embedding-3-small (OpenAI) | Convert text to vectors | Creates numerical representations |
| **Vector Database** | FAISS | Store & search vectors | Fast similarity search (CPU-based) |
| **LLM Generator** | Multi-LLM (OpenAI/Groq) | Generate answers | OpenAI: GPT-4o-mini, Groq: Llama-3.3-70b |
| **Web Search** | DuckDuckGo Search | Fallback search | Triggered when PDFs lack answer |
| **Cache** | Redis | Speed up queries | Fast key-value cache for repeated questions |
| **User Interface** | Streamlit | Web application | Chat interface with RTL support |

---



## Testing Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Unit Tests                           │
└─────────────────────────────────────────────────────────┘

tests/
└── test_retriever.py
    │
    ├── Test 1: Initialization
    ├── Test 2: Create Vector Store
    ├── Test 3: Save and Load
    ├── Test 4: Retrieve Documents
    ├── Test 5: Top-K Limit
    ├── Test 6: Load Nonexistent Store
    ├── Test 7: Empty Documents
    └── Test 8: Retrieval Relevance

┌─────────────────────────────────────────────────────────┐
│                RAGAS Evaluation                         │
└─────────────────────────────────────────────────────────┘

evaluate.py
    │
    ├── Faithfulness: 0.9917
    ├── Answer Relevancy: 0.8058
    ├── Context Recall: 0.9375
    └── Context Precision: 0.8303
    
    Overall Score: 0.8913 (Excellent!)
```

---

