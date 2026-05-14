"""Tests for prompts module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from memory_bridge.core.prompts import load_prompt


class TestLoadPrompt:
    async def test_returns_content_when_file_exists(self, tmp_path: Path) -> None:
        prompt_file: Path = tmp_path / "agent-1.md"
        prompt_file.write_text("重点关注技术偏好\n忽略闲聊", encoding="utf-8")

        result: str | None = await load_prompt("agent-1", str(tmp_path))
        assert result == "重点关注技术偏好\n忽略闲聊"

    async def test_returns_none_when_file_does_not_exist(self, tmp_path: Path) -> None:
        result: str | None = await load_prompt("nonexistent-agent", str(tmp_path))
        assert result is None

    async def test_returns_none_when_file_is_empty_or_whitespace(self, tmp_path: Path) -> None:
        prompt_file: Path = tmp_path / "empty-agent.md"
        prompt_file.write_text("\n  \n", encoding="utf-8")

        result: str | None = await load_prompt("empty-agent", str(tmp_path))
        assert result is None

    async def test_returns_none_when_directory_missing(self, tmp_path: Path) -> None:
        missing_dir: str = os.path.join(str(tmp_path), "new-prompts")
        assert not os.path.exists(missing_dir)
        result: str | None = await load_prompt("any-agent", missing_dir)
        assert result is None
        assert not os.path.exists(missing_dir)

    async def test_same_file_reread_on_second_call(self, tmp_path: Path) -> None:
        prompt_file: Path = tmp_path / "hot-reload.md"
        prompt_file.write_text("v1", encoding="utf-8")

        result1: str | None = await load_prompt("hot-reload", str(tmp_path))
        assert result1 == "v1"

        prompt_file.write_text("v2", encoding="utf-8")
        result2: str | None = await load_prompt("hot-reload", str(tmp_path))
        assert result2 == "v2"

    async def test_encoding_utf8(self, tmp_path: Path) -> None:
        prompt_file: Path = tmp_path / "cn.md"
        prompt_file.write_text("请关注用户的技术偏好和项目上下文", encoding="utf-8")

        result: str | None = await load_prompt("cn", str(tmp_path))
        assert result == "请关注用户的技术偏好和项目上下文"

    async def test_read_text_runs_in_thread_pool(self, tmp_path: Path) -> None:
        prompt_file: Path = tmp_path / "agent-1.md"
        prompt_file.write_text("test content", encoding="utf-8")

        with patch("memory_bridge.core.prompts.asyncio.to_thread") as mock_to_thread:
            mock_to_thread.return_value = "test content"
            result: str | None = await load_prompt("agent-1", str(tmp_path))
            assert result == "test content"
            mock_to_thread.assert_called_once()

    async def test_raises_on_permission_error(self, tmp_path: Path) -> None:
        prompt_file: Path = tmp_path / "locked.md"
        prompt_file.write_text("secret", encoding="utf-8")

        with patch(
            "memory_bridge.core.prompts.asyncio.to_thread",
            side_effect=PermissionError("denied"),
        ):
            with pytest.raises(PermissionError, match="denied"):
                await load_prompt("locked", str(tmp_path))
