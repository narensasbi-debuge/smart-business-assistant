FROM python:3.11-slim

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000

# Build the FAISS index on first boot if missing, then start the server.
# $PORT is set by Render/Railway; defaults to 8000 locally.
CMD ["sh", "-c", "python scripts/ensure_index.py && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
