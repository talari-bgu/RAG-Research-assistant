from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research assistant. Answer the question using only the provided context.\n"
        "If the context is insufficient, say so clearly — do not speculate.\n\n"
        "Context:\n{context}",
    ),
    ("human", "{question}"),
])
