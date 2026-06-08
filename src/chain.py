from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from src.config import settings
from src.prompts import RAG_PROMPT
from src.retriever import retrieve_and_rerank


def _format_docs(docs) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, p.{doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for doc in docs
    )


def get_chain():
    llm = ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=0,
    )

    return (
        {
            "context": itemgetter("question") | RunnableLambda(retrieve_and_rerank) | RunnableLambda(_format_docs),
            "question": itemgetter("question"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )


def ask(question: str) -> str:
    return get_chain().invoke({"question": question})
