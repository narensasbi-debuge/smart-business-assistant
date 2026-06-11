"""Agent tools: company-doc search (RAG), HubSpot CRM contact creation, email sending.

Every external integration degrades gracefully: if its credentials are missing,
the tool returns a clear message instead of crashing, so the whole app stays
runnable with just an OpenAI key.
"""
import logging
import smtplib
from email.mime.text import MIMEText

from langchain_core.tools import tool

from app.config import get_settings

logger = logging.getLogger(__name__)

# The RAG chain is created lazily and cached: it needs the vector index, which
# only exists after scripts/ingest_docs.py has been run.
_rag_chain = None


def _get_rag():
    global _rag_chain
    if _rag_chain is None:
        from app.rag_chain import RAGChain

        _rag_chain = RAGChain()
    return _rag_chain


def reset_rag_cache() -> None:
    """Force the next doc search to reload the vector index (after new uploads)."""
    global _rag_chain
    _rag_chain = None


@tool
def search_company_docs(query: str) -> str:
    """Search the company's internal documents (policies, FAQs, product info).
    Use this for ANY question about the company, its policies, or its products."""
    try:
        result = _get_rag().query(query)
    except FileNotFoundError as exc:
        return f"Document search unavailable: {exc}"
    sources = ", ".join(result["sources"]) or "n/a"
    return f"{result['answer']}\n\n[Sources: {sources}]"


@tool
def create_hubspot_contact(email: str, firstname: str = "", lastname: str = "", phone: str = "") -> str:
    """Create a new contact in the HubSpot CRM. Requires an email address;
    firstname, lastname and phone are optional."""
    settings = get_settings()
    if not settings.hubspot_access_token:
        return (
            "HubSpot is not configured (set HUBSPOT_ACCESS_TOKEN in .env). "
            f"Simulated: contact {email} ({firstname} {lastname}, {phone}) would be created."
        )
    try:
        from hubspot import HubSpot
        from hubspot.crm.contacts import SimplePublicObjectInputForCreate
        from hubspot.crm.contacts.exceptions import ApiException

        client = HubSpot(access_token=settings.hubspot_access_token)
        payload = SimplePublicObjectInputForCreate(
            properties={
                "email": email,
                "firstname": firstname,
                "lastname": lastname,
                "phone": phone,
            }
        )
        contact = client.crm.contacts.basic_api.create(
            simple_public_object_input_for_create=payload
        )
        return f"HubSpot contact created successfully with ID {contact.id} for {email}."
    except ApiException as exc:
        logger.exception("HubSpot API error")
        return f"HubSpot API error: {exc.status} {exc.reason}"
    except Exception as exc:  # network errors etc.
        logger.exception("HubSpot error")
        return f"Failed to create HubSpot contact: {exc}"


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient. Provide the recipient address, a subject and the body text."""
    settings = get_settings()
    if not settings.smtp_host:
        logger.info("SIMULATED EMAIL -> to=%s subject=%s body=%s", to, subject, body)
        return (
            f"Email simulated (SMTP not configured - set SMTP_HOST etc. in .env). "
            f"Would send to {to} with subject '{subject}'."
        )
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = settings.email_from or settings.smtp_user
        msg["To"] = to
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
            server.starttls()
            if settings.smtp_user:
                server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)
        return f"Email sent to {to} with subject '{subject}'."
    except Exception as exc:
        logger.exception("SMTP error")
        return f"Failed to send email: {exc}"


ALL_TOOLS = [search_company_docs, create_hubspot_contact, send_email]
