# Smart Business Assistant 🤖

![Python](https://img.shields.io/badge/Python-3.11-blue) ![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688) ![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C) ![Tests](https://img.shields.io/badge/tests-7%20passing-brightgreen) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Author:** Naren · [GitHub](https://github.com/narensasbi-debuge) · 📦 Live demo: *coming soon*

A **production-grade AI assistant** that answers questions from company documents
(RAG) and takes real actions — creating CRM contacts and sending emails — through
natural language. Includes a real-time web chat UI, a voice interface (Twilio),
and serverless AWS deployment with CI/CD.

> Portfolio project demonstrating: LLM application development, RAG pipelines,
> AI agents with tools, FastAPI/WebSockets, CRM integration, voice AI, Docker,
> AWS Lambda, and GitHub Actions CI/CD.

## Architecture

```
                                ┌─────────────────────────────┐
  Browser (chat UI) ──WS/REST──►│                             │
  curl / Postman ────REST──────►│   FastAPI  (app/main.py)    │
  Twilio phone call ──webhook──►│                             │
                                └──────────┬──────────────────┘
                                           ▼
                                ┌─────────────────────────────┐
                                │  LangChain Tool-Calling     │
                                │  Agent  (app/agent.py)      │
                                └───┬───────────┬─────────┬───┘
                          search docs│   create contact│  send email│
                                     ▼               ▼            ▼
                          ┌────────────────┐  ┌───────────┐  ┌─────────┐
                          │ RAG pipeline   │  │  HubSpot  │  │  SMTP   │
                          │ FAISS/Pinecone │  │  CRM API  │  │ (email) │
                          │ + OpenAI embed │  └───────────┘  └─────────┘
                          └────────────────┘
```

## Features

| Capability | Implementation |
|---|---|
| RAG pipeline | LangChain `create_retrieval_chain` + OpenAI embeddings |
| Vector store | FAISS (local, free, default) or Pinecone (managed) |
| AI agent + tools | LangChain tool-calling agent: doc search, CRM, email |
| Conversation memory | Per-session chat history (REST + WebSocket) |
| Chat interface | Built-in browser UI at `/` (WebSocket w/ REST fallback) |
| Document upload | 📎 button / `POST /upload`: chunk + embed + query instantly |
| CRM integration | HubSpot Contacts API (private app token) |
| Voice AI | Twilio `<Gather input="speech">` webhook at `/voice` |
| Cloud deployment | AWS Lambda + API Gateway via Mangum |
| CI/CD | GitHub Actions: pytest on PRs, deploy on push to main |
| Containerization | Dockerfile + docker-compose |
| LLM flexibility | OpenAI by default; DeepSeek via OpenAI-compatible API |

## Quickstart (local, free — only an OpenAI key needed)

```powershell
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
copy .env.example .env         # then edit .env and set OPENAI_API_KEY

# 4. Ingest the sample documents (or drop your own PDFs/MD/TXT into data/)
python scripts/ingest_docs.py

# 5. Run the server
uvicorn app.main:app --reload
```

Open <http://localhost:8000> for the chat UI, or <http://localhost:8000/docs>
for the interactive API documentation.

### Try these messages

- *"What is the refund policy?"* → RAG search over `data/`
- *"What plans do you offer and which has API access?"* → RAG search
- *"Add john@example.com (John Smith) to the CRM"* → HubSpot tool
- *"Email john@example.com our pricing details"* → email tool
- *"Add jane@acme.com to the CRM and email her the support hours"* → multi-tool

Without HubSpot/SMTP credentials the action tools run in **simulation mode**
and say so — the agent flow still works end to end.

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Browser chat UI |
| GET | `/health` | Health + configuration status |
| POST | `/chat` | Agent chat `{"message": "...", "session_id": "..."}` |
| WS | `/ws/{session_id}` | Real-time chat (plain text frames) |
| POST | `/rag` | Direct RAG query (no agent) — test retrieval quality |
| POST | `/upload` | Upload a document (.pdf/.md/.txt) — indexed and queryable immediately |
| POST | `/voice` | Twilio voice webhook (TwiML) |

```powershell
curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{\"message\": \"What is the refund policy?\"}'
```

## Run with Docker

```bash
docker compose up --build
# First time: ingest docs inside the container
docker compose exec ai-assistant python scripts/ingest_docs.py
```

## Optional integrations

### HubSpot CRM (free developer account)
1. Create a free account at developers.hubspot.com → create a **private app**.
2. Grant scopes `crm.objects.contacts.read` and `crm.objects.contacts.write`.
3. Put the access token in `.env` as `HUBSPOT_ACCESS_TOKEN`.

### Email (SMTP)
Set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `EMAIL_FROM`
(e.g. Gmail with an app password). Unset = simulation mode.

### Twilio voice
1. Buy/claim a Twilio number, set its **Voice webhook** to
   `https://<your-host>/voice` (POST). For local testing use `ngrok http 8000`.
2. Call the number: Twilio transcribes your speech, the agent answers, and
   Twilio speaks the reply back. Follow-up questions are supported.

### Pinecone instead of FAISS
```
pip install langchain-pinecone pinecone
```
Set in `.env`: `VECTOR_BACKEND=pinecone`, `PINECONE_API_KEY`,
`PINECONE_INDEX_NAME` (index dimension **1536** for `text-embedding-3-small`),
then re-run `python scripts/ingest_docs.py`.

### DeepSeek instead of OpenAI (chat model)
```
LLM_MODEL=deepseek-chat
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=sk-your-deepseek-key
```
Embeddings still use `OPENAI_API_KEY` (DeepSeek has no embeddings API).

## Deploy to AWS Lambda

1. Create a Lambda function `smart-business-assistant` (Python 3.11) and an
   **API Gateway HTTP API** with a `$default` route → Lambda integration.
   Set handler to `app.main.handler` and the environment variables from `.env`.
2. Add repo secrets `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` in GitHub.
3. Push to `main` — `.github/workflows/deploy.yml` runs tests, packages the
   app, and deploys automatically.

Notes: bundle the prebuilt `vector_index/` into the zip (or use Pinecone,
recommended for Lambda since the filesystem is read-only outside `/tmp`).
Increase Lambda memory to 1024 MB+ and timeout to 30 s.

Simpler alternative: deploy the Docker image to **Render.com** / **Railway**
(works as-is, WebSockets included).

## Run tests

```powershell
pytest tests/ -v
```

## Project structure

```
smart-business-assistant/
├── app/
│   ├── main.py          # FastAPI app: REST, WebSocket, voice webhook, Lambda handler
│   ├── agent.py         # Tool-calling agent + per-session memory
│   ├── rag_chain.py     # RAG pipeline (retriever + LLM, source attribution)
│   ├── tools.py         # search_company_docs / create_hubspot_contact / send_email
│   ├── vectorstore.py   # FAISS / Pinecone factory
│   ├── voice.py         # Twilio TwiML handlers
│   ├── models.py        # Pydantic schemas
│   ├── config.py        # Settings from .env (pydantic-settings)
│   └── static/index.html# Browser chat client
├── data/                # Source documents (sample docs included)
├── scripts/ingest_docs.py
├── tests/test_smoke.py
├── Dockerfile / docker-compose.yml
├── .github/workflows/deploy.yml
└── requirements.txt / .env.example
```

## Cost notes

- `gpt-4o-mini` + `text-embedding-3-small`: a full demo session costs **well
  under $1**; ingesting the sample docs costs fractions of a cent.
- FAISS is free and local; Pinecone, HubSpot, and Twilio all have free tiers.
- `temperature=0` + small models + capped agent iterations keep costs predictable.
