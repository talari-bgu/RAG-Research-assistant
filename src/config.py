from pydantic_settings import BaseSettings
from pathlib import Path

_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    # API keys
    openai_api_key: str
    pinecone_api_key: str
    cohere_api_key: str
    langchain_api_key: str = ""
    langchain_tracing_v2: str = "true"
    langchain_project: str = "rag-research-assistant"

    # Paths
    pdf_dir: Path = _ROOT / "data" / "pdfs"

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_retrieve: int = 10
    top_k_rerank: int = 4

    # Models
    llm_model: str = "gpt-4o-mini"
    cohere_embed_model: str = "embed-english-v3.0"
    pinecone_index_name: str = "rag-agent"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    class Config:
        env_file = str(_ROOT / ".env")


settings = Settings()
