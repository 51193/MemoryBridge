from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    dashscope_api_key: str = ""
    embedding_model: str = "text-embedding-v4"
    embedding_dims: int = 1024

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    memory_bridge_host: str = "0.0.0.0"
    memory_bridge_port: int = 8000
    session_max_history: int = 50
    prompts_dir: str = "prompts"
    token_db_path: str = "data/tokens.db"

    def validate_secrets(self) -> None:
        """Raise ValueError if any required secret is not configured."""
        missing: list[str] = []
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if not self.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
