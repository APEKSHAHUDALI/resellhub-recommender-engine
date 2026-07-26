"""
Centralized application configuration.

All environment-dependent values live here. Nothing else in the codebase
should call os.environ directly - this keeps config auditable and testable.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=("settings_",)
    )

    # App
    app_name: str = "ResellHub Recommendation Engine"
    environment: str = "development"  # development | staging | production
    api_v1_prefix: str = "/api/v1"
    debug: bool = True

    # Database
    database_url: str = "sqlite:///./resellhub.db"

    # Redis (recommendation result cache + celery broker)
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 900  # 15 minutes

    # Auth
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_USE_ENV_VAR"
    access_token_expire_minutes: int = 60 * 24
    algorithm: str = "HS256"

    # Recommender
    model_artifact_dir: str = "./model_artifacts"
    collaborative_factors: int = 32
    collaborative_regularization: float = 0.05
    collaborative_iterations: int = 20
    hybrid_weight_collaborative: float = 0.55
    hybrid_weight_content: float = 0.30
    hybrid_weight_popularity: float = 0.15
    min_interactions_for_cf: int = 3  # below this, a user is "cold start"

    # CORS
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]


@lru_cache
def get_settings() -> Settings:
    return Settings()
