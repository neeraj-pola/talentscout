# app/config.py
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: str
    openai_model_heavy: str = "gpt-4o-mini"     # default to the cheap, tested model
    openai_model_light: str = "gpt-4o-mini"
    openai_model_cheap: str | None = None       # alias used by profile_summary
    openai_embedding_model: str = "text-embedding-3-small"

    database_url: str = "sqlite:///./talentscout.db"
    chroma_path: str = "./chroma_db"
    log_level: str = "INFO"
    mock_sources_base_url: str = "http://localhost:9417"

    # Tunables
    max_profiles_per_source: int = 40
    top_k_retrieval: int = 20
    top_k_after_rerank: int = 10
    top_n_shortlist: int = 10
    must_have_weight: float = 1.0
    nice_have_weight: float = 0.4
    must_have_penalty_threshold: float = 0.3
    must_have_penalty_multiplier: float = 0.5


settings = Settings()