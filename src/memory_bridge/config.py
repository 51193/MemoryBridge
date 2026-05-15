from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    deepseek_thinking_enabled: bool = False
    deepseek_reasoning_effort: str | None = None

    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dims: int = 1024

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    mem0_collection_name: str = "memory_bridge"
    mem0_history_db_path: str = "./data/mem0_history.db"

    memory_bridge_host: str = "0.0.0.0"
    memory_bridge_port: int = 8000
    session_window_size: int = 10
    session_db_path: str = "data/sessions.db"
    prompts_dir: str = "prompts"
    token_db_path: str = "data/tokens.db"

    @model_validator(mode="after")
    def validate_secrets(self) -> "Settings":
        """Validate required secrets are non-empty after construction."""
        missing: list[str] = []
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if not self.dashscope_api_key:
            missing.append("DASHSCOPE_API_KEY")
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        return self
