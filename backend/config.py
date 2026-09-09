"""
Backend configuration using Pydantic settings.
All settings can be overridden via environment variables or .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # App
    app_name: str = "Agentic AI Credit Underwriting System"
    app_version: str = "1.0.0"
    debug: bool = False

    # CORS
    cors_origins: list[str] = ["http://localhost:3001", "http://127.0.0.1:3001"]

    # Database — SQLite by default; set DATABASE_URL for PostgreSQL
    database_url: str = f"sqlite:///{PROJECT_ROOT}/data/audit.db"

    # ML / Data Paths
    models_dir: Path = PROJECT_ROOT / "models"
    model_path: Path = PROJECT_ROOT / "models" / "pd_xgboost_calibrated.joblib"
    feature_path: Path = PROJECT_ROOT / "data" / "features" / "model_features.parquet"
    evaluation_path: Path = PROJECT_ROOT / "data" / "features" / "pd_evaluation.csv"
    benchmark_path: Path = PROJECT_ROOT / "data" / "features" / "models_benchmark.json"
    shap_cache_path: Path = PROJECT_ROOT / "data" / "explanations" / "on_demand_shap.parquet"
    vector_db_path: Path = PROJECT_ROOT / "data" / "policy_vector_db"

    # Real-world research policy thresholds
    pd_approve_max: float = 0.10
    pd_decline_min: float = 0.20
    pd_risk_tier_low_max: float = 0.05
    pd_risk_tier_moderate_max: float = 0.10
    pd_risk_tier_high_max: float = 0.20
    min_income: float = 50000.0
    max_credit_to_income: float = 10.0
    max_cc_utilization: float = 1.50

    # LLM Provider
    llm_provider: str = "ollama"
    ollama_model: str = "llama3.1:8b"
    ollama_base_url: str = "http://localhost:11434"

    # Versions
    model_version: str = "pd_xgboost_calibrated_v1"
    policy_version: str = "research_policy_v1"

    # RAG
    rag_collection_name: str = "credit_underwriting_policies"
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 4

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
