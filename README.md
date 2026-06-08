# RAG Research Assistant

Chat with a collection of PDFs using Retrieval-Augmented Generation (RAG). Upload documents through the web UI, index them into Pinecone, and ask questions in natural language — the system retrieves the most relevant passages and generates a grounded answer.

<!-- Replace the path below once you add a screenshot -->
<!-- ![UI Screenshot](assets/screenshot.png) -->

---

## Architecture

```
┌─────────────────────── INGEST ────────────────────────┐
│                                                        │
│  PDF files  →  LangChain loader  →  text chunks       │
│                   (512 tokens, 64 overlap)             │
│                           │                           │
│                   Cohere Embeddings                    │
│                    (embed-english-v3.0)                │
│                           │                           │
│                   Pinecone Vector DB  ←── stored       │
└────────────────────────────────────────────────────────┘

┌─────────────────────── QUERY ─────────────────────────┐
│                                                        │
│  User question  →  Cohere Embeddings                  │
│                           │                           │
│              Pinecone similarity search                │
│                  (top 10 candidates)                   │
│                           │                           │
│              Cohere Reranker (rerank-v3.5)             │
│                  (keeps top 4)                         │
│                           │                           │
│         LangChain RAG prompt + GPT-4o mini             │
│                           │                           │
│                      Answer                           │
└────────────────────────────────────────────────────────┘
```

---

## Stack

| Layer | Technology |
| :--- | :--- |
| **PDF Loading** | `PyPDFDirectoryLoader` via LangChain |
| **Chunking** | `RecursiveCharacterTextSplitter` — 512 tokens, 64 overlap |
| **Embeddings** | Cohere `embed-english-v3.0` (1024 dimensions) |
| **Vector Database** | Pinecone (serverless, cosine similarity) |
| **Reranker** | Cohere `rerank-v3.5` |
| **LLM** | OpenAI `gpt-4o-mini` |
| **API** | FastAPI + Pydantic |
| **Web UI** | Vanilla HTML/CSS/JS served by FastAPI |
| **Observability** | LangSmith tracing |

---

## Project Structure

```
rag-research-assistant/
├── assets/                     # Images for this README (add screenshots here)
├── data/
│   └── pdfs/                   # Source PDFs (populated via UI upload or manually)
├── src/
│   ├── config.py               # All settings and environment variable loading
│   ├── ingest.py               # PDF loading, chunking, and indexing pipeline
│   ├── pinecone_store.py       # Pinecone index management and vectorstore helpers
│   ├── retriever.py            # Similarity search + Cohere reranking
│   ├── chain.py                # LangChain RAG chain (retrieval → prompt → LLM)
│   ├── prompts.py              # Prompt templates
│   └── __init__.py
├── api/
│   ├── main.py                 # FastAPI app — serves UI and exposes REST endpoints
│   └── __init__.py
├── ui/
│   └── index.html              # Single-page web UI
├── .env                        # API keys — not commited
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install dependencies

```bash
git clone <your-repo-url>
cd rag-research-assistant
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=pcsk_...
COHERE_API_KEY=...
LANGCHAIN_API_KEY=...          # Optional — for LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-research-assistant
```

### 3. Start the server

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000` in your browser.

---

## How to Use

### Via the Web UI

1. **Upload PDFs** — drag and drop PDF files (or click to browse) in the left sidebar. Click **Upload** to save them to `data/pdfs/`.
2. **Ingest** — click **⚡ Ingest** to embed the PDFs and store them in Pinecone. This may take a minute depending on document size.
3. **Ask questions** — type a question in the chat box and press Enter. The assistant retrieves the most relevant passages and generates a grounded answer.

### Via the API

```bash
# Ingest documents currently in data/pdfs/
curl -X POST http://localhost:8000/ingest

# Upload a PDF
curl -X POST http://localhost:8000/upload \
  -F "files=@path/to/your.pdf"

# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'

# Health check
curl http://localhost:8000/health
```

Interactive API docs are available at `http://localhost:8000/docs`.

---


## Adding Images to This README

Place screenshots or diagrams in the `assets/` folder and reference them like this:

```markdown
![Description](assets/your-image.png)
```

