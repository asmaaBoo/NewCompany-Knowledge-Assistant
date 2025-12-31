# King Salman Social Center Assistant

AI-powered intelligent system for answering employee inquiries about center policies

---

## Overview

**Company Knowledge Assistant** is an advanced RAG (Retrieval-Augmented Generation) application that uses:
- **Multi-LLM Support**: OpenAI GPT-4o-mini or Groq Llama-3.3-70b
- **FAISS** for efficient vector storage and retrieval
- **Arabic PDF processing** for company policies
- **Web Search fallback** using DuckDuckGo
- **Response caching** for faster performance
- **RAGAS evaluation** for quality metrics

---

## Architecture

See detailed architecture diagrams and flow charts in `docs/ARCHITECTURE_DIAGRAM.md`

**Quick Overview:**
- PDFs → Chunking (900c) → Embeddings (OpenAI) → FAISS Vector Store
- User Question → Retrieve Top-6 → Generate Answer (GPT/Llama)
- If unclear → Web Search (DuckDuckGo) → Enhanced Answer

---

## Key Features

### Core Functionality
- **Arabic PDF Processing**: Loads and processes company policy documents
- **Semantic Search**: Uses text-embedding-3-small for accurate retrieval
- **Smart Chunking**: RecursiveCharacterTextSplitter (900 tokens, 150 overlap)
- **Multi-LLM Generation**: Choose between OpenAI or Groq
- **Response Caching**:  cache (Redis) for repeated questions
- **Web Search Fallback**: DuckDuckGo integration for missing info

### User Interface
- **Modern Streamlit UI**: Blue gradient theme
- **Chat Interface**: WhatsApp-style message bubbles
- **Quick Questions**: Pre-defined common queries
- **New Conversation**: Easy chat reset

### Evaluation & Testing
- **RAGAS Metrics**: Faithfulness, Answer Relevancy, Context Recall, Context Precision
- **Unit Tests**: Comprehensive retriever tests (8 test cases)
- **Performance Tracking**: Latency monitoring and cache statistics

---

## Quick Start

### Prerequisites
- Python 3.10+
- OpenAI API Key (required for embeddings)
- Groq API Key (optional, for Llama generation)


### Installation

```bash
# 1. Clone the repository
git clone <repository-url>
cd company-knowledge-assistant

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment variables
cp .env.example .env
# Edit .env and add your API keys
```

---

## Configuration

### Environment Variables (.env)

```env
# Required: OpenAI (for embeddings)
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxx

# Optional: Groq (for generation)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxx

# LLM Provider Selection
LLM_PROVIDER=openai          # Options: "openai" or "groq"
LLM_MODEL=gpt-4o-mini        # OpenAI: gpt-4o-mini, 
                             # Groq: llama-3.3-70b-versatile, 

# Temperature (0-1, lower = more deterministic)
LLM_TEMPERATURE=0
```

### Config Parameters (src/config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EMBEDDING_MODEL` | text-embedding-3-small | OpenAI embedding model |
| `LLM_PROVIDER` | openai | LLM provider: "openai" or "groq" |
| `LLM_MODEL` | gpt-4o-mini | Model name |
| `LLM_TEMPERATURE` | 0 | Model creativity (0-1) |
| `CHUNK_SIZE` | 900 | Document chunk size (tokens) |
| `CHUNK_OVERLAP` | 150 | Overlap between chunks |
| `TOP_K` | 6 | Number of retrieved chunks |
| `ENABLE_CACHE` | True | Enable response caching |

---

## Project Structure

```
company-knowledge-assistant/
│
├── app.py                      # Main Streamlit application
├── evaluate.py                 # RAGAS evaluation script
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables 
│
├── data/
│   ├── pdfs/                   # Input PDF documents
│   │   ├── KSSC_General_Policies.pdf
│   │   ├── KSSC_HR_Policies.pdf
│   │   └── KSSC_Financial_Policies.pdf
│   │
│   └── vector_store/           # FAISS index (auto-generated)
│       ├── index.faiss
│       └── index.pkl
│
├── src/                        # Source code modules
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── document_processor.py   # PDF loading and chunking
│   ├── retriever.py            # FAISS retrieval logic
│   ├── generator.py            # Multi-LLM generation (OpenAI/Groq)
│   ├── rag_pipeline.py         # Main RAG orchestration
│   └── utils.py                # Helper functions
│
├── tests/                      # Unit tests
│   ├── __init__.py
│   └── test_retriever.py       # Retriever unit tests (8 tests)
│
├── docs/                       # Documentation
│   ├── ARCHITECTURE_DIAGRAM.md # System architecture
│   ├── README.md      
│
└── outputs/                    # Evaluation results
    └── evaluation_results.csv
```

---

## Usage

### 1. Prepare PDF Documents

Place your Arabic PDF policy documents in the `data/pdfs/` folder:

```bash
data/pdfs/
├── KSSC_General_Policies.pdf
├── KSSC_HR_Policies.pdf
└── KSSC_Financial_Policies.pdf
```

### 2. Run the Application

```bash
streamlit run app.py
```

The app will:
1. Check for existing vector store
2. If not found, process PDFs and build FAISS index
3. Launch the chat interface at `http://localhost:8501`

**Console Output:**
```
Using OpenAI: gpt-4o-mini
Web search enabled (DDGS)
```
or
```
Using Groq: llama-3.3-70b-versatile
Web search enabled (DDGS)
```

### 3. Ask Questions

**Example Questions:**
- "كم مدة الإجازة السنوية؟"
- "ما سياسة الاستقطاب والتوظيف؟"
- "ما مدة فترة الاختبار؟"
- "ما الموارد المالية للمركز؟"

### 4. Evaluate Performance

Run RAGAS evaluation on 12 test questions:

```bash
python evaluate.py
```

**Output:**
```
Starting Evaluation - RAGAS Evaluation
==================================================
Initializing system with full configuration...
   Embedding Model: text-embedding-3-small
   LLM Model: gpt-4o-mini
   LLM Provider: openai
   Top-K: 6
   Cache Enabled: True

Evaluation Results - RAGAS Metrics
==================================================
Faithfulness:       0.9917  (Adherence to context)
Answer Relevancy:   0.8058  (Relevance to question)
Context Recall:     0.9375  (Context coverage)
Context Precision:  0.8303  (Retrieved context accuracy)
==================================================

Overall Average Score: 0.8913

Detailed results saved to: evaluation_results.csv
```

### 5. Run Unit Tests

```bash
# Run retriever tests
python tests/test_retriever.py

# Expected output:
# Running 8 tests...
# All tests passed!
```

---

## Multi-LLM Support

### Supported Providers

**1. OpenAI (Default)**
- Models: gpt-4o-mini
- Cost: $0.15 per 1M tokens (gpt-4o-mini)
- Speed: 2-3 seconds per query
- Quality: Excellent

**2. Groq (Free Alternative)**
- Models: llama-3.3-70b-versatile
- Cost: FREE
- Speed: 0.5-1 second per query (3x faster!)
- Quality: Very Good

### Switching Between Providers

**Method 1: Environment Variable (Recommended)**

Edit `.env`:
```env
# Use OpenAI
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini

# Use Groq
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
```

**Method 2: Config File**

Edit `src/config.py`:
```python
LLM_PROVIDER = "groq"
LLM_MODEL = "llama-3.3-70b-versatile"
```

### Quick Comparison

| Feature | OpenAI (GPT-4o-mini) | Groq (Llama-3.3-70b) |
|---------|---------------------|----------------------|
| **Speed** | 2-3s per query | 0.5-1s per query |
| **Cost** | $0.15/1M tokens | FREE |
| **Quality** | Excellent | Very Good |
| **Arabic** | Native support | Good support |
| **Context** | 128K tokens | 32K tokens |

**Recommendation**: Use Groq for development (free & fast), OpenAI for production (more reliable).

---


## Evaluation Methodology

### RAGAS Metrics

| Metric | Description | Range | Our Score |
|--------|-------------|-------|-----------|
| **Faithfulness** | How well the answer is grounded in retrieved context | 0-1 | 0.9917 |
| **Answer Relevancy** | How relevant the answer is to the question | 0-1 | 0.8058 |
| **Context Recall** | How much of the ground truth is covered by contexts | 0-1 | 0.9375 |
| **Context Precision** | Precision of retrieved contexts | 0-1 | 0.8303 |

### Test Questions (12 samples)

Evaluation covers:
- HR policies (leave, probation, recruitment)
- Financial policies (budgeting, investment, resources)
- General policies (conduct, safety, conflict of interest)

---

## Testing

### Unit Tests

**Test Coverage:**
- Test 1: Initialization (verify setup)
- Test 2: Create Vector Store (build database)
- Test 3: Save and Load (persistence)
- Test 4: Retrieve Documents (search functionality)
- Test 5: Top-K Limit (result limiting)
- Test 6: Load Nonexistent Store (error handling)
- Test 7: Retrieval Relevance (accuracy)

**Run Tests:**
```bash
python tests/test_retriever.py
```

**Expected Output:**
```
Retriever Unit Tests - Generic Version
==================================================

[Test 1] Initialization
  top_k is set correctly
  embeddings initialized
  db is None (expected)

...

Test Results
==================================================
Passed:7/7
Failed: 0
Errors: 0
==================================================

All tests passed!
```



## Troubleshooting

### Issue: "OPENAI_API_KEY not found"
**Solution:** Create `.env` file with your API key:
```env
OPENAI_API_KEY=sk-your-key-here
```

### Issue: "GROQ_API_KEY not found" (when using Groq)
**Solution:** Add Groq API key to `.env`:
```env
GROQ_API_KEY=gsk-your-key-here
```
Or system will automatically fall back to OpenAI.

### Issue: "langchain-groq not installed"
**Solution:** Install Groq package:
```bash
pip install langchain-groq
```

### Issue: "PDF files not found"
**Solution:** Ensure PDFs are in `data/pdfs/` folder:
```bash
ls data/pdfs/
# Should show: KSSC_General_Policies.pdf, KSSC_HR_Policies.pdf, etc.
```

### Issue: "FAISS load error"
**Solution:** Delete and rebuild vector store:
```bash
rm -rf data/vector_store
streamlit run app.py
```

### Issue: "Web search not working"
**Solution:** Check DuckDuckGo installation:
```bash
pip install --upgrade duckduckgo-search
```

### Issue: "Tests failing"
**Solution:** Ensure all dependencies installed:
```bash
pip install -r requirements.txt
```

---

