from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    gemini_api_key: SecretStr | None = None
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai"
    gemini_model: str = "gemini-3.8-flash"
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    groq_fallback_model: str | None = None
    llm_timeout_seconds: float = Field(default=30.0, gt=0)
    llm_cache_path: Path = Path("data/llm_cache.sqlite")
    database_url: SecretStr | None = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dimension: int = Field(default=384, gt=0)
    retrieval_candidates: int = Field(default=50, gt=0)
    rerank_results: bool = False
    protect_built_in_documents: bool = True
    chunk_max_words: int = Field(default=120, gt=0)
    max_upload_bytes: int = Field(default=5_000_000, gt=0)
    max_question_chars: int = Field(default=1000, gt=0)
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
