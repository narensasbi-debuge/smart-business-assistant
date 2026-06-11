"""FastAPI application: REST chat, WebSocket chat, direct RAG queries,
Twilio voice webhook, and an embedded browser chat UI.

Run locally:  uvicorn app.main:app --reload
AWS Lambda:   handler = Mangum(app)  (exposed below)
"""
import logging
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import HTMLResponse
from mangum import Mangum

from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RagRequest,
    RagResponse,
    UploadResponse,
)
from app.vectorstore import index_exists

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Smart Business Assistant",
    description="RAG chatbot + AI agent with CRM, email and voice integrations",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"

# Lazy singleton for the RAG chain used by the /rag endpoint
_rag_chain = None


def _require_llm_key() -> None:
    settings = get_settings()
    if not settings.chat_api_key:
        raise HTTPException(
            status_code=503,
            detail="LLM not configured: set OPENAI_API_KEY in your .env file.",
        )


def _get_rag():
    global _rag_chain
    if _rag_chain is None:
        from app.rag_chain import RAGChain

        try:
            _rag_chain = RAGChain()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    return _rag_chain


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def chat_ui():
    """Simple browser chat client (WebSocket with REST fallback)."""
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health", response_model=HealthResponse)
def health():
    settings = get_settings()
    return HealthResponse(
        status="ok",
        llm_configured=bool(settings.chat_api_key),
        vector_backend=settings.vector_backend,
        vector_index_ready=index_exists(),
        hubspot_configured=bool(settings.hubspot_access_token),
        email_mode="smtp" if settings.smtp_host else "simulated",
        twilio_configured=bool(settings.twilio_account_sid),
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Send a message to the AI agent (doc search + CRM + email tools)."""
    _require_llm_key()
    from app.agent import run_agent

    try:
        answer = run_agent(request.message, session_id=request.session_id)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Agent error")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}")
    return ChatResponse(response=answer, session_id=request.session_id)


@app.post("/rag", response_model=RagResponse)
def rag_query(request: RagRequest):
    """Query the RAG pipeline directly (no agent/tools) - useful for testing retrieval."""
    _require_llm_key()
    try:
        result = _get_rag().query(request.message)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("RAG error")
        raise HTTPException(status_code=500, detail=f"RAG error: {exc}")
    return RagResponse(answer=result["answer"], sources=result["sources"])


MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload a document (.pdf/.md/.txt): it is saved to the data folder,
    chunked, embedded, and becomes immediately queryable in the chat."""
    from app.ingestion import SUPPORTED_EXTENSIONS, ingest_file

    safe_name = Path(file.filename or "").name  # strip any client-side path
    suffix = Path(safe_name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    _require_llm_key()

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large (max 10 MB).")

    settings = get_settings()
    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / safe_name
    re_upload = target.exists()  # re-uploads trigger a full rebuild to avoid duplicate chunks
    target.write_bytes(content)

    try:
        chunks = ingest_file(target, rebuild=re_upload)
    except ValueError as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Ingestion error for %s", safe_name)
        raise HTTPException(status_code=500, detail=f"Failed to index document: {exc}")

    # Make the new content visible to the agent and the /rag endpoint immediately
    global _rag_chain
    _rag_chain = None
    from app.tools import reset_rag_cache

    reset_rag_cache()

    return UploadResponse(
        filename=safe_name,
        chunks_indexed=chunks,
        reindexed_all=re_upload,
        message=f"'{safe_name}' indexed ({chunks} chunks). You can now ask questions about it.",
    )


@app.websocket("/ws/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    """Real-time chat over WebSocket. Send plain text, receive plain text."""
    await websocket.accept()
    settings = get_settings()
    if not settings.chat_api_key:
        await websocket.send_text("Server not configured: set OPENAI_API_KEY in .env.")
        await websocket.close()
        return

    from app.agent import run_agent

    try:
        while True:
            message = await websocket.receive_text()
            try:
                answer = run_agent(message, session_id=session_id)
            except Exception as exc:
                logger.exception("Agent error (websocket)")
                answer = f"Sorry, something went wrong: {exc}"
            await websocket.send_text(answer)
    except WebSocketDisconnect:
        logger.info("WebSocket session %s disconnected", session_id)


@app.post("/voice", response_class=HTMLResponse)
async def voice_webhook(request: Request):
    """Twilio voice webhook: greets the caller, then answers transcribed speech."""
    from app.voice import answer_twiml, greeting_twiml

    form = await request.form()
    speech = form.get("SpeechResult")
    caller = form.get("From", "unknown")
    if speech:
        return HTMLResponse(content=answer_twiml(speech, caller), media_type="application/xml")
    return HTMLResponse(content=greeting_twiml(), media_type="application/xml")


# AWS Lambda entry point (API Gateway -> Lambda -> FastAPI)
handler = Mangum(app)
