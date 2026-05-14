import pytest

from memory_bridge.config import Settings


def _make_settings(**kwargs: object) -> Settings:
    return Settings(_env_file=None, **kwargs)  # type: ignore[arg-type, call-arg]


class TestSettings:
    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_MODEL", raising=False)
        settings: Settings = _make_settings()
        assert settings.deepseek_api_key == ""
        assert settings.dashscope_api_key == ""
        assert settings.deepseek_base_url == "https://api.deepseek.com"
        assert settings.deepseek_model == "deepseek-chat"
        assert settings.deepseek_thinking_enabled is False
        assert settings.deepseek_reasoning_effort is None
        assert settings.embedding_model == "text-embedding-v4"
        assert settings.embedding_dims == 1024
        assert settings.qdrant_host == "localhost"
        assert settings.qdrant_port == 6333
        assert settings.memory_bridge_host == "0.0.0.0"
        assert settings.memory_bridge_port == 8000
        assert settings.session_max_history == 50

    def test_custom_values(self) -> None:
        settings: Settings = _make_settings(
            deepseek_api_key="sk-custom",
            deepseek_base_url="https://custom.deepseek.com",
            deepseek_model="deepseek-v4-pro",
            dashscope_api_key="sk-custom",
            embedding_model="text-embedding-v3",
            embedding_dims=768,
            qdrant_host="127.0.0.1",
            qdrant_port=9999,
            memory_bridge_host="127.0.0.1",
            memory_bridge_port=9998,
            session_max_history=100,
        )
        assert settings.deepseek_api_key == "sk-custom"
        assert settings.deepseek_base_url == "https://custom.deepseek.com"
        assert settings.deepseek_model == "deepseek-v4-pro"
        assert settings.embedding_model == "text-embedding-v3"
        assert settings.embedding_dims == 768
        assert settings.qdrant_host == "127.0.0.1"
        assert settings.qdrant_port == 9999
        assert settings.memory_bridge_port == 9998
        assert settings.session_max_history == 100

    def test_validate_secrets_raises_when_empty(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        settings: Settings = _make_settings()
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            settings.validate_secrets()

    def test_validate_secrets_pass_with_keys(self) -> None:
        settings: Settings = _make_settings(
            deepseek_api_key="sk-test",
            dashscope_api_key="sk-test",
        )
        settings.validate_secrets()

    def test_validate_secrets_raises_partial(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        settings: Settings = _make_settings(
            deepseek_api_key="sk-test",
        )
        with pytest.raises(ValueError, match="DASHSCOPE_API_KEY"):
            settings.validate_secrets()
