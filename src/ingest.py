from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.pinecone_store import index_documents


def load_and_split() -> list:
    loader = PyPDFDirectoryLoader(str(settings.pdf_dir))
    docs = loader.load()
    if not docs:
        raise ValueError(f"No PDFs found in {settings.pdf_dir}")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(docs)


def ingest():
    print(f"Loading PDFs from {settings.pdf_dir} ...")
    chunks = load_and_split()
    print(f"Split into {len(chunks)} chunks. Embedding with Cohere and storing in Pinecone ...")
    index_documents(chunks)
    print("Done.")


if __name__ == "__main__":
    ingest()
