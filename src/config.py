from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # API keys
    openai_api_key: str
    pinecone_api_key: str = ""
    cohere_api_key: str = ""
    langchain_api_key: str = ""       # LangSmith
    langchain_tracing_v2: str = "true"
    langchain_project: str = "rag-research-assistant"

    # Paths
    pdf_dir: Path = Path("data/pdfs")
    chroma_dir: Path = Path("data/chroma_db")

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_retrieve: int = 10         # fetch more, then rerank
    top_k_rerank: int = 4            # keep top 4 after rerank

    # Model
    llm_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"

    class Config:
        env_file = ".env"

settings = Settings()