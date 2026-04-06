from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str
    qdrant_url: str
    qdrant_collection: str
    openai_api_key: str
    basic_model: str
    complex_model: str
    app_name: str
    log_level: str

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
