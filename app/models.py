"""Pydantic request/response schemas for the API."""
from typing import List, Optional

from pydantic import BaseModel, EmailStr, Field


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message for the agent")
    session_id: str = Field(default="default", description="Conversation session id")


class ChatResponse(BaseModel):
    response: str
    session_id: str


class RagRequest(BaseModel):
    message: str = Field(..., min_length=1)


class RagResponse(BaseModel):
    answer: str
    sources: List[str] = []


class ContactCreate(BaseModel):
    email: EmailStr
    firstname: Optional[str] = None
    lastname: Optional[str] = None
    phone: Optional[str] = None


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    reindexed_all: bool = False
    message: str


class HealthResponse(BaseModel):
    status: str
    llm_configured: bool
    vector_backend: str
    vector_index_ready: bool
    hubspot_configured: bool
    email_mode: str
    twilio_configured: bool
