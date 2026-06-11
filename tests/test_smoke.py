"""Smoke tests: modules import cleanly and the API responds without external keys."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["vector_backend"] in ("faiss", "pinecone")


def test_chat_ui_served():
    res = client.get("/")
    assert res.status_code == 200
    assert "Smart Business Assistant" in res.text


def test_chat_requires_llm_key(monkeypatch):
    from app import config

    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "")
    config.get_settings.cache_clear()
    try:
        res = client.post("/chat", json={"message": "hello"})
        # Without a key the API must fail gracefully with 503, not crash
        assert res.status_code == 503
        assert "OPENAI_API_KEY" in res.json()["detail"]
    finally:
        config.get_settings.cache_clear()


def test_chat_validation():
    res = client.post("/chat", json={})
    assert res.status_code == 422


def test_voice_greeting():
    res = client.post("/voice", data={})
    assert res.status_code == 200
    assert "<Response>" in res.text
    assert "Gather" in res.text


def test_upload_rejects_unsupported_type():
    res = client.post(
        "/upload", files={"file": ("malware.exe", b"binary", "application/octet-stream")}
    )
    assert res.status_code == 400
    assert "Unsupported file type" in res.json()["detail"]


def test_tools_degrade_gracefully():
    from app.tools import create_hubspot_contact, send_email

    out = create_hubspot_contact.invoke(
        {"email": "john@example.com", "firstname": "John"}
    )
    assert "Simulated" in out or "created" in out

    out = send_email.invoke({"to": "john@example.com", "subject": "Hi", "body": "Test"})
    assert "simulated" in out.lower() or "sent" in out.lower()
