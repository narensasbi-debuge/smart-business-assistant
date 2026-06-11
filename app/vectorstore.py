"""Vector store factory: local FAISS by default, Pinecone when configured.

FAISS keeps the project runnable for free with no external account.
Switch to Pinecone by setting VECTOR_BACKEND=pinecone in .env
(and installing langchain-pinecone + pinecone).
"""
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings

from app.config import get_settings


def get_embeddings() -> Embeddings:
    settings = get_settings()
    return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)


def index_exists() -> bool:
    """True when a vector index is available to query."""
    settings = get_settings()
    if settings.vector_backend == "pinecone":
        return bool(settings.pinecone_api_key)
    return (Path(settings.faiss_index_dir) / "index.faiss").exists()


def load_vectorstore() -> VectorStore:
    """Open the existing index for retrieval. Raises with a helpful message if missing."""
    settings = get_settings()
    embeddings = get_embeddings()

    if settings.vector_backend == "pinecone":
        from langchain_pinecone import PineconeVectorStore  # lazy: optional dependency

        return PineconeVectorStore(
            index_name=settings.pinecone_index_name,
            embedding=embeddings,
            pinecone_api_key=settings.pinecone_api_key,
        )

    from langchain_community.vectorstores import FAISS

    index_dir = Path(settings.faiss_index_dir)
    if not (index_dir / "index.faiss").exists():
        raise FileNotFoundError(
            f"No FAISS index found at '{index_dir}'. "
            "Run `python scripts/ingest_docs.py` first to ingest documents."
        )
    return FAISS.load_local(str(index_dir), embeddings, allow_dangerous_deserialization=True)


def add_to_index(chunks: List[Document]) -> None:
    """Incrementally add chunks to the existing index (create it if missing)."""
    settings = get_settings()

    if settings.vector_backend == "pinecone":
        build_vectorstore(chunks)  # Pinecone upserts, so this is already incremental
        return

    from langchain_community.vectorstores import FAISS

    index_dir = Path(settings.faiss_index_dir)
    if (index_dir / "index.faiss").exists():
        store = FAISS.load_local(
            str(index_dir), get_embeddings(), allow_dangerous_deserialization=True
        )
        store.add_documents(chunks)
        store.save_local(str(index_dir))
    else:
        build_vectorstore(chunks)


def build_vectorstore(chunks: List[Document]) -> VectorStore:
    """Embed chunks and persist the index (used by the ingestion script)."""
    settings = get_settings()
    embeddings = get_embeddings()

    if settings.vector_backend == "pinecone":
        from langchain_pinecone import PineconeVectorStore

        return PineconeVectorStore.from_documents(
            chunks,
            embedding=embeddings,
            index_name=settings.pinecone_index_name,
            pinecone_api_key=settings.pinecone_api_key,
        )

    from langchain_community.vectorstores import FAISS

    store = FAISS.from_documents(chunks, embeddings)
    store.save_local(settings.faiss_index_dir)
    return store
