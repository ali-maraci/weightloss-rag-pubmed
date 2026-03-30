from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Environment and project settings."""

    # -------------------------
    # API Keys & Cloud Services
    # -------------------------
    GOOGLE_API_KEY: str = ""
    GROQ_API_KEY: str = ""
    NCBI_API_KEY: str
    NCBI_EMAIL: str
    OPENAI_API_KEY: str = ""
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""

    # -------------------------
    # Evaluation Judge LLM
    # -------------------------
    EVAL_JUDGE_PROVIDER: str = "openai"   # "openai" or "groq"
    EVAL_JUDGE_MODEL: str = "gpt-4o-mini"

    # -------------------------
    # LangChain Tracing
    # -------------------------
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_ENDPOINT: str = "https://eu.api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "weightloss-rag-pubmed"

    # -------------------------
    # Project Paths
    # -------------------------
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def create_dirs(self):
        """Ensure all necessary directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.create_dirs()
