import cohere
from langchain_core.documents import Document

from src.config import settings
from src.pinecone_store import get_vectorstore


def retrieve_and_rerank(query: str) -> list[Document]:
    candidates = get_vectorstore().similarity_search(query, k=settings.top_k_retrieve)

    if not candidates:
        return []

    co = cohere.Client(settings.cohere_api_key)
    response = co.rerank(
        query=query,
        documents=[doc.page_content for doc in candidates],
        model="rerank-v3.5",
        top_n=settings.top_k_rerank,
    )
    return [candidates[r.index] for r in response.results]
