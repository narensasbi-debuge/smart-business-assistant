"""Build the vector index at startup if it doesn't exist yet.

Used by the container entrypoint so cloud deployments (Render, ECS, ...)
self-initialize from the documents bundled in data/ on first boot.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import get_settings
from app.vectorstore import index_exists


def main() -> int:
    settings = get_settings()
    if settings.vector_backend != "faiss":
        print("Pinecone backend configured; no local index needed.")
        return 0
    if index_exists():
        print("Vector index present; skipping ingestion.")
        return 0
    if not settings.openai_api_key:
        print("WARNING: no vector index and no OPENAI_API_KEY; starting without RAG index.")
        return 0

    from app.ingestion import rebuild_index

    print("No vector index found - building from data/ ...")
    chunks = rebuild_index()
    print(f"Built index with {chunks} chunks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
