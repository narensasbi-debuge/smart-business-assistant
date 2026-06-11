"""Central configuration loaded from environment variables / .env file."""
from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM / embeddings
    openai_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_base_url: Optional[str] = None  # e.g. https://api.deepseek.com for DeepSeek
    llm_api_key: Optional[str] = None   # falls back to openai_api_key when unset
    llm_temperature: float = 0.0
    embedding_model: str = "text-embedding-3-small"

    # Vector store
    vector_backend: Literal["faiss", "pinecone"] = "faiss"
    faiss_index_dir: str = "vector_index"
    pinecone_api_key: str = ""
    pinecone_index_name: str = "business-assistant"

    # Retrieval / chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retrieval_k: int = 4

    # Document source folder for ingestion
    data_dir: str = "data"

    # HubSpot CRM (private app access token)
    hubspot_access_token: str = ""

    # Email via SMTP (simulation mode when unset)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    email_from: str = ""

    # Twilio voice
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    @property
    def chat_api_key(self) -> str:
        return self.llm_api_key or self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
