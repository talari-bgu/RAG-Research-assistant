# RAG Research Assistant

Chat with a collection of PDFs using Retrieval-Augmented Generation (RAG). This repository provides the full file structure and every core file required to deploy the system.

## What You Will Build

A production-ready system that ingests PDF documents, processes and embeds them into a vector database, and answers user questions by retrieving relevant text chunks and feeding them to a Large Language Model (LLM).

---

## Architecture at a Glance

| Component | Technology Stack |
| :--- | :--- |
| **PDF Loader** | `pypdf` via LangChain |
| **Chunking Strategy** | `RecursiveCharacterTextSplitter` (512 tokens, 64 token overlap) |
| **Embeddings** | OpenAI's `text-embedding-3-small` |
| **Vector Database** | Chroma (for local development) $\rightarrow$ Pinecone (for production) |
| **Reranker** | Cohere `rerank-v3.5` |
| **Core LLM** | `gpt-4o-mini` (fast + cost-efficient) or `claude-3-5-sonnet` |
| **API Framework** | FastAPI + Pydantic |
| **Observability/Tracing** | LangSmith |

---

## Project File Structure

```text
rag-research-assistant/
├── data/
│   └── pdfs/               # Drop your source PDFs here
├── src/
│   ├── __init__.py
│   ├── ingest.py           # Load, chunk, embed, and store logic
│   ├── retriever.py        # Query vector DB + apply reranking
│   ├── chain.py            # Main RAG chain setup (retrieval + LLM)
│   ├── prompts.py          # All prompt templates
│   └── config.py           # Application settings / environment variables
├── api/
│   ├── __init__.py
│   └── main.py             # FastAPI backend app
├── eval/
│   ├── golden_set.json     # Test question/answer evaluation pairs
│   └── run_eval.py         # Evaluation pipeline
├── notebooks/
│   └── explore.ipynb       # Prototyping and testing notebook
├── .env                    # Local API keys (Never commit this to source control!)
├── .env.example            # Template for environment configurations
├── .gitignore              # Files ignored by Git
├── requirements.txt        # System dependencies
└── README.md               # Project documentation