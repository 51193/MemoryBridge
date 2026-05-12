"""Tests for prompts module."""

import os
from pathlib import Path

from memory_bridge.core.prompts import load_prompt


class TestLoadPrompt:
    def test_returns_content_when_file_exists(self, tmp_path: Path) -> None:
        prompt_file: Path = tmp_path / "agent-1.md"
        prompt_file.write_text("重点关注技术偏好\n忽略闲聊", encoding="utf-8")

        result: str | None = load_prompt("agent-1", str(tmp_path))
        assert result == "重点关注技术偏好\n忽略闲聊"

    def test_returns_none_when_file_does_not_exist(self, tmp_path: Path) -> None:
        result: str | None = load_prompt("nonexistent-agent", str(tmp_path))
        assert result is None

    def test_returns_none_when_file_is_empty_or_whitespace(self, tmp_path: Path) -> None:
        prompt_file: Path = tmp_path / "empty-agent.md"
        prompt_file.write_text("\n  \n", encoding="utf-8")

        result: str | None = load_prompt("empty-agent", str(tmp_path))
        assert result is None

    def test_auto_creates_directory_if_missing(self, tmp_path: Path) -> None:
        missing_dir: str = os.path.join(str(tmp_path), "new-prompts")
        assert not os.path.exists(missing_dir)

        load_prompt("any-agent", missing_dir)
        assert os.path.exists(missing_dir)

    def test_same_file_reread_on_second_call(self, tmp_path: Path) -> None:
        prompt_file: Path = tmp_path / "hot-reload.md"
        prompt_file.write_text("v1", encoding="utf-8")

        result1: str | None = load_prompt("hot-reload", str(tmp_path))
        assert result1 == "v1"

        prompt_file.write_text("v2", encoding="utf-8")
        result2: str | None = load_prompt("hot-reload", str(tmp_path))
        assert result2 == "v2"

    def test_encoding_utf8(self, tmp_path: Path) -> None:
        prompt_file: Path = tmp_path / "cn.md"
        prompt_file.write_text("请关注用户的技术偏好和项目上下文", encoding="utf-8")

        result: str | None = load_prompt("cn", str(tmp_path))
        assert result == "请关注用户的技术偏好和项目上下文"
