"""Application settings and configuration."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    pinecone_api_key: Optional[str] = None

    # Pinecone settings
    pinecone_environment: str = "us-west1-gcp"
    pinecone_index_name: str = "rag-production"

    # Redis settings
    redis_url: str = "redis://localhost:6379"
    redis_cache_ttl: int = 3600

    # ChromaDB settings
    chroma_persist_dir: str = "./chroma_data"

    # Application settings
    log_level: str = "INFO"
    max_chunk_size: int = 1000
    chunk_overlap: int = 200

    # Embedding settings
    embedding_model: str = "text-embedding-3-small"

    # Vector store backend
    vector_store_backend: str = "chroma"  # "chroma" or "pinecone"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
