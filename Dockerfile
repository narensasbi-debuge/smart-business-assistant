FROM python:3.11-slim

WORKDIR /app

# Install dependencies first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY data ./data

EXPOSE 8000

# The FAISS index is built at container start if missing (requires OPENAI_API_KEY)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
