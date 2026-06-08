import time

from langchain_cohere import CohereEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from src.config import settings

_COHERE_EMBED_DIMENSIONS = {
    "embed-english-v3.0": 1024,
    "embed-multilingual-v3.0": 1024,
    "embed-english-light-v3.0": 384,
    "embed-multilingual-light-v3.0": 384,
}


def _embed_dimension() -> int:
    return _COHERE_EMBED_DIMENSIONS.get(settings.cohere_embed_model, 1024)


def _pinecone_client() -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def ensure_pinecone_index() -> None:
    pc = _pinecone_client()
    name = settings.pinecone_index_name
    if name in [idx.name for idx in pc.list_indexes()]:
        return

    pc.create_index(
        name=name,
        dimension=_embed_dimension(),
        metric="cosine",
        spec=ServerlessSpec(
            cloud=settings.pinecone_cloud,
            region=settings.pinecone_region,
        ),
    )
    while not pc.describe_index(name).status.get("ready"):
        time.sleep(1)


def get_embeddings() -> CohereEmbeddings:
    return CohereEmbeddings(
        model=settings.cohere_embed_model,
        cohere_api_key=settings.cohere_api_key,
    )


def get_vectorstore() -> PineconeVectorStore:
    ensure_pinecone_index()
    index = _pinecone_client().Index(settings.pinecone_index_name)
    return PineconeVectorStore(index=index, embedding=get_embeddings())


def index_documents(chunks) -> PineconeVectorStore:
    vs = get_vectorstore()
    vs.add_documents(chunks)
    return vs
