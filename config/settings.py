from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from typing import Optional


class Settings(BaseSettings):
    # API Keys
    openai_api_key: str
    pinecone_api_key: str
    pinecone_environment: str
    cohere_api_key: Optional[str] = None

    # Model Settings
    model_name: str = "gpt-4"
    embedding_model: str = "text-embedding-3-small"
    temperature: float = 0.7
    max_tokens: int = 2000

    # Vector Store Settings
    index_name: str = "finhealth-advisor"
    chunk_size: int = 1000
    chunk_overlap: int = 200

    model_config = ConfigDict(
        env_file=".env",
        extra="ignore"   # 🔥 THIS FIXES THE ERROR
    )


settings = Settings()

