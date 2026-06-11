"""One-time ingestion: load documents from data/, chunk, embed, persist index.

Usage:
    python scripts/ingest_docs.py

Supports .pdf, .md and .txt files. With the default FAISS backend the index is
saved to ./vector_index; with VECTOR_BACKEND=pinecone the chunks are upserted
into the configured Pinecone index.

(The web UI's upload button uses the same logic via app/ingestion.py.)
"""
import sys
from pathlib import Path

# Allow running as `python scripts/ingest_docs.py` from the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.ingestion import rebuild_index


def main() -> int:
    settings = get_settings()
    if not settings.openai_api_key:
        print("ERROR: OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.")
        return 1

    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory '{data_dir}' not found.")
        return 1

    print(f"Ingesting documents from {data_dir.resolve()} ...")
    try:
        chunk_count = rebuild_index()
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

    if settings.vector_backend == "faiss":
        print(f"Done. {chunk_count} chunks saved to ./{settings.faiss_index_dir}")
    else:
        print(f"Done. {chunk_count} chunks upserted to Pinecone index '{settings.pinecone_index_name}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
