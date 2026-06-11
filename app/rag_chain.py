"""RAG pipeline: retriever + LLM answer generation with source attribution."""
from typing import Dict

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.vectorstore import load_vectorstore

RAG_SYSTEM_PROMPT = (
    "You are a helpful company assistant. Answer the user's question using ONLY the "
    "context below. If the answer is not in the context, say you don't know and "
    "suggest contacting support. Be concise and cite the relevant policy when possible.\n\n"
    "Context:\n{context}"
)


def get_chat_llm() -> ChatOpenAI:
    settings = get_settings()
    kwargs = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "api_key": settings.chat_api_key,
    }
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url  # e.g. DeepSeek's OpenAI-compatible API
    return ChatOpenAI(**kwargs)


class RAGChain:
    def __init__(self):
        settings = get_settings()
        self.vectorstore = load_vectorstore()
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": settings.retrieval_k})

        prompt = ChatPromptTemplate.from_messages(
            [("system", RAG_SYSTEM_PROMPT), ("human", "{input}")]
        )
        docs_chain = create_stuff_documents_chain(get_chat_llm(), prompt)
        self.chain = create_retrieval_chain(retriever, docs_chain)

    def query(self, question: str) -> Dict:
        result = self.chain.invoke({"input": question})
        sources = sorted(
            {doc.metadata.get("source", "unknown") for doc in result.get("context", [])}
        )
        return {"answer": result["answer"], "sources": sources}
