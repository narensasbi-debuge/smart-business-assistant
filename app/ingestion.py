"""Document ingestion: load -> chunk -> embed -> index.

Shared by the CLI script (scripts/ingest_docs.py) and the /upload endpoint.
"""
import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.vectorstore import add_to_index, build_vectorstore

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def load_file(path: Path) -> List[Document]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path)).load()
    if suffix in {".md", ".txt"}:
        return TextLoader(str(path), encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {suffix}")


def load_directory(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.extend(load_file(path))
    return docs


def split_documents(docs: List[Document]) -> List[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return splitter.split_documents(docs)


def rebuild_index() -> int:
    """Re-embed everything in the data directory into a fresh index."""
    settings = get_settings()
    docs = load_directory(Path(settings.data_dir))
    if not docs:
        raise ValueError(f"No supported documents found in '{settings.data_dir}'.")
    chunks = split_documents(docs)
    build_vectorstore(chunks)
    logger.info("Rebuilt index with %d chunks from %d documents", len(chunks), len(docs))
    return len(chunks)


def ingest_file(path: Path, rebuild: bool = False) -> int:
    """Index a single file. With rebuild=True the whole index is rebuilt instead
    (used when a file is re-uploaded, to avoid duplicate chunks)."""
    if rebuild:
        return rebuild_index()
    chunks = split_documents(load_file(path))
    if not chunks:
        raise ValueError(f"No text could be extracted from '{path.name}'.")
    add_to_index(chunks)
    logger.info("Indexed %d chunks from %s", len(chunks), path.name)
    return len(chunks)
