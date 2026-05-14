"""Tests for host_manager module."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from memory_bridge.config import Settings
from memory_bridge.host_manager import HostManagerError, _start_qdrant


def _make_settings(**kwargs: str) -> Settings:
    defaults: dict[str, str] = {
        "deepseek_api_key": "sk-test",
        "dashscope_api_key": "sk-test",
    }
    defaults.update(kwargs)
    return Settings(_env_file=None, **defaults)  # type: ignore[arg-type, call-arg]


class TestStartQdrant:
    def test_missing_binary_raises(self, tmp_path: Path) -> None:
        settings: Settings = _make_settings()
        missing_bin: Path = tmp_path / "nonexistent" / "qdrant"
        with pytest.raises(HostManagerError, match="Qdrant binary not found"):
            _start_qdrant(settings, missing_bin, tmp_path / "data")

    def test_missing_data_dir_raises(self, tmp_path: Path) -> None:
        settings: Settings = _make_settings()
        fake_bin: Path = tmp_path / "fake_qdrant"
        fake_bin.write_text("#!/bin/sh\nsleep 60\n")
        fake_bin.chmod(0o755)

        missing_dir: Path = tmp_path / "nonexistent_data"
        with pytest.raises(HostManagerError, match="Data directory not found"):
            _start_qdrant(settings, fake_bin, missing_dir)

    def test_starts_with_existing_data_dir(self, tmp_path: Path) -> None:
        settings: Settings = _make_settings()
        fake_bin: Path = tmp_path / "fake_qdrant"
        fake_bin.write_text("#!/bin/sh\nsleep 60\n")
        fake_bin.chmod(0o755)

        data_dir: Path = tmp_path / "data"
        data_dir.mkdir()

        mock_proc: MagicMock = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.stderr = None

        with (
            patch("memory_bridge.host_manager.subprocess.Popen", return_value=mock_proc),
            patch("memory_bridge.host_manager.httpx.get") as mock_get,
        ):
            mock_resp: MagicMock = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            proc: subprocess.Popen[bytes] = _start_qdrant(
                settings, fake_bin, data_dir
            )
            assert proc is mock_proc

    def test_startup_timeout_raises(self, tmp_path: Path) -> None:
        settings: Settings = _make_settings()
        fake_bin: Path = tmp_path / "fake_qdrant"
        fake_bin.write_text("#!/bin/sh\nsleep 60\n")
        fake_bin.chmod(0o755)

        data_dir: Path = tmp_path / "data"
        data_dir.mkdir()

        mock_proc: MagicMock = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.stderr = None

        with (
            patch(
                "memory_bridge.host_manager.subprocess.Popen",
                return_value=mock_proc,
            ),
            patch(
                "memory_bridge.host_manager.httpx.get",
                side_effect=httpx.ConnectError("refused"),
            ),
            patch("memory_bridge.host_manager.QDRANT_STARTUP_TIMEOUT", 1),
            patch("memory_bridge.host_manager.time.sleep", return_value=None),
        ):
            with pytest.raises(HostManagerError, match="Qdrant failed to start"):
                _start_qdrant(settings, fake_bin, data_dir)

            mock_proc.kill.assert_called_once()

    def test_uses_correct_env_and_args(self, tmp_path: Path) -> None:
        settings: Settings = _make_settings(qdrant_port="9999")
        fake_bin: Path = tmp_path / "fake_qdrant"
        fake_bin.write_text("#!/bin/sh\nsleep 60\n")
        fake_bin.chmod(0o755)

        data_dir: Path = tmp_path / "data"
        data_dir.mkdir()

        mock_proc: MagicMock = MagicMock(spec=subprocess.Popen)
        mock_proc.poll.return_value = None
        mock_proc.pid = 12345
        mock_proc.stderr = None

        mock_popen: MagicMock
        mock_get: MagicMock
        with (
            patch(
                "memory_bridge.host_manager.subprocess.Popen",
                return_value=mock_proc,
            ) as mock_popen,
            patch(
                "memory_bridge.host_manager.httpx.get",
            ) as mock_get,
        ):
            mock_resp: MagicMock = MagicMock()
            mock_resp.status_code = 200
            mock_get.return_value = mock_resp

            _start_qdrant(settings, fake_bin, data_dir)

            assert mock_popen.call_args is not None
            popen_args: tuple[object, ...] = mock_popen.call_args[0]
            popen_kwargs: dict[str, object] = mock_popen.call_args[1]
            assert popen_args[0] == [str(fake_bin)]
            env: object = popen_kwargs["env"]
            assert isinstance(env, dict)
            assert env["QDRANT__STORAGE__STORAGE_PATH"] == str(data_dir)
            assert env["QDRANT__SERVICE__HTTP_PORT"] == "9999"
