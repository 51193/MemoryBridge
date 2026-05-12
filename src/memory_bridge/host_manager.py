"""Host Manager — orchestrate Qdrant and MemoryBridge processes."""

import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import httpx

from .config import Settings
from .core.logging import setup_logging

logger: logging.Logger = logging.getLogger(__name__)

QDRANT_STARTUP_TIMEOUT: int = 30
BRIDGE_STARTUP_TIMEOUT: int = 15
SHUTDOWN_TIMEOUT: int = 10

_qdrant_proc: subprocess.Popen[bytes] | None = None
_bridge_proc: subprocess.Popen[bytes] | None = None


class HostManagerError(Exception):
    """Raised when host manager encounters a fatal error during startup."""


def main() -> None:
    setup_logging()

    if len(sys.argv) > 1 and sys.argv[1] == "--setup":
        try:
            _run_setup()
        except HostManagerError as e:
            print(f"Error: {e}")
            sys.exit(1)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--init-token":
        _run_init_token()
        return

    settings: Settings = Settings()
    try:
        settings.validate_secrets()
    except ValueError as e:
        print(f"Configuration error: {e}")
        sys.exit(1)
    try:
        _run(settings)
    except HostManagerError as e:
        print(f"Error: {e}")
        sys.exit(1)


def _run_setup() -> None:
    """Initialize directories, download Qdrant, and create .env template."""
    data_dir: Path = Path("data/qdrant")
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created data directory: {data_dir}")

    qdrant_bin: Path = Path("bin/qdrant")
    if not qdrant_bin.exists():
        _download_qdrant(qdrant_bin)
    else:
        print(f"Qdrant binary found at {qdrant_bin}, skipping download.")

    env_file: Path = Path(".env")
    if not env_file.exists():
        env_file.write_text(
            "DEEPSEEK_API_KEY=\n"
            "DASHSCOPE_API_KEY=\n"
            "LOG_LEVEL=INFO\n"
            "DEEPSEEK_BASE_URL=https://api.deepseek.com\n"
            "DEEPSEEK_MODEL=deepseek-chat\n"
            "#DEEPSEEK_THINKING_ENABLED=false\n"
            "#DEEPSEEK_REASONING_EFFORT=high\n"
            "EMBEDDING_MODEL=text-embedding-v4\n"
            "EMBEDDING_DIMS=1024\n"
            "DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n"
            "MEM0_COLLECTION_NAME=memory_bridge\n"
            "MEM0_HISTORY_DB_PATH=./data/mem0_history.db\n"
            "QDRANT_HOST=localhost\n"
            "QDRANT_PORT=6333\n"
            "MEMORY_BRIDGE_HOST=0.0.0.0\n"
            "MEMORY_BRIDGE_PORT=8000\n"
            "SESSION_MAX_HISTORY=50\n"
            "PROMPTS_DIR=prompts\n"
            "TOKEN_DB_PATH=data/tokens.db\n"
        )
        print("Created .env template. Edit it to add your API keys.")
    else:
        print(".env already exists, skipping.")

    print()
    print("Setup complete. Edit .env with your API keys, then run:")
    print("  python memorybridge.pyz")


def _download_qdrant(qdrant_bin: Path) -> None:
    os_name: str = platform.system()
    arch: str = platform.machine()

    target_map: dict[tuple[str, str], str] = {
        ("Linux", "x86_64"): "x86_64-unknown-linux-gnu",
        ("Linux", "aarch64"): "aarch64-unknown-linux-gnu",
        ("Darwin", "x86_64"): "x86_64-apple-darwin",
        ("Darwin", "arm64"): "aarch64-apple-darwin",
    }
    target: str | None = target_map.get((os_name, arch))
    if target is None:
        raise HostManagerError(f"Unsupported platform: {os_name} {arch}")

    url: str = (
        "https://github.com/qdrant/qdrant/releases/latest/download/"
        f"qdrant-{target}.tar.gz"
    )
    print(f"Downloading Qdrant for {target} ...")
    print(f"  {url}")

    qdrant_bin.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tgz_path: Path = Path(tmp) / "qdrant.tar.gz"
        _download_file(url, tgz_path)

        print("Extracting ...")
        with tarfile.open(tgz_path, "r:gz") as tf:
            tf.extractall(path=tmp)

        extracted: Path = Path(tmp) / "qdrant"
        if not extracted.exists():
            raise HostManagerError("Failed to extract Qdrant binary")

        shutil.copy2(extracted, qdrant_bin)
        qdrant_bin.chmod(0o755)

    print(f"Qdrant installed to {qdrant_bin}")


def _download_file(url: str, dest: Path) -> None:
    with urlopen(url) as resp:
        dest.write_bytes(resp.read())


def _run_init_token() -> None:
    """Generate initial API token."""
    from .core.tokens import TokenStore

    store: TokenStore = TokenStore(os.environ.get("TOKEN_DB_PATH", "data/tokens.db"))
    token: str = store.create(label="admin")
    print(f"Token: {token}")
    print()
    print("Use this token in your requests:")
    print(f"  Authorization: Bearer {token}")


def _run(settings: Settings) -> None:
    global _qdrant_proc, _bridge_proc

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    qdrant_bin: Path = Path("bin/qdrant")
    data_dir: Path = Path("data/qdrant")

    _qdrant_proc = _start_qdrant(settings, qdrant_bin, data_dir)

    try:
        _bridge_proc = _start_bridge(settings)

        logger.info("all processes started, running")
        while True:
            if _qdrant_proc is not None and _qdrant_proc.poll() is not None:
                logger.error("Qdrant exited unexpectedly")
                break
            if _bridge_proc is not None and _bridge_proc.poll() is not None:
                logger.error("MemoryBridge exited unexpectedly")
                break
            time.sleep(1.0)
    finally:
        _shutdown()


def _start_qdrant(
    settings: Settings,
    qdrant_bin: Path,
    data_dir: Path,
) -> subprocess.Popen[bytes]:
    if not qdrant_bin.exists():
        raise HostManagerError(
            f"Qdrant binary not found at {qdrant_bin}."
            " Run: python memorybridge.pyz --setup"
        )

    data_dir.mkdir(parents=True, exist_ok=True)

    env: dict[str, str] = os.environ.copy()
    env["QDRANT__STORAGE__STORAGE_PATH"] = str(data_dir)
    env["QDRANT__SERVICE__HTTP_PORT"] = str(settings.qdrant_port)

    proc: subprocess.Popen[bytes] = subprocess.Popen(
        [str(qdrant_bin)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    health_url: str = f"http://{settings.qdrant_host}:{settings.qdrant_port}/healthz"
    logger.info("starting Qdrant on %s ...", health_url)
    for _ in range(QDRANT_STARTUP_TIMEOUT):
        try:
            r: httpx.Response = httpx.get(health_url, timeout=1.0)
            if r.status_code == 200:
                break
        except httpx.RequestError:
            pass
        time.sleep(1.0)
    else:
        proc.kill()
        raise HostManagerError("Qdrant failed to start within timeout")

    logger.info("Qdrant started pid=%s", proc.pid)
    return proc


def _start_bridge(settings: Settings) -> subprocess.Popen[bytes]:
    health_url: str = (
        f"http://{settings.memory_bridge_host}:{settings.memory_bridge_port}/health"
    )

    env: dict[str, str] = os.environ.copy()
    if not sys.argv[0].endswith(".pyz"):
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent.parent)

    proc: subprocess.Popen[bytes] = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "memory_bridge.main:app",
            "--host",
            settings.memory_bridge_host,
            "--port",
            str(settings.memory_bridge_port),
        ],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    logger.info("starting MemoryBridge on %s ...", health_url)
    for _ in range(BRIDGE_STARTUP_TIMEOUT):
        try:
            r: httpx.Response = httpx.get(health_url, timeout=1.0)
            if r.status_code == 200:
                break
        except httpx.RequestError:
            pass
        time.sleep(1.0)
    else:
        _shutdown_qdrant()
        raise HostManagerError("MemoryBridge failed to start within timeout")

    logger.info("MemoryBridge started pid=%s", proc.pid)
    return proc


def _handle_signal(signum: int, frame: object) -> None:
    logger.info("received signal, shutting down ...")
    _shutdown()
    sys.exit(0)


def _shutdown() -> None:
    global _bridge_proc, _qdrant_proc
    _shutdown_bridge()
    _shutdown_qdrant()


def _shutdown_bridge() -> None:
    global _bridge_proc
    if _bridge_proc is not None and _bridge_proc.poll() is None:
        logger.info("shutting down MemoryBridge ...")
        _bridge_proc.terminate()
        try:
            _bridge_proc.wait(timeout=SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            _bridge_proc.kill()
            _bridge_proc.wait()
        logger.info("MemoryBridge stopped")
        _bridge_proc = None


def _shutdown_qdrant() -> None:
    global _qdrant_proc
    if _qdrant_proc is not None and _qdrant_proc.poll() is None:
        logger.info("shutting down Qdrant ...")
        _qdrant_proc.terminate()
        try:
            _qdrant_proc.wait(timeout=SHUTDOWN_TIMEOUT)
        except subprocess.TimeoutExpired:
            _qdrant_proc.kill()
            _qdrant_proc.wait()
        logger.info("Qdrant stopped")
        _qdrant_proc = None


if __name__ == "__main__":
    main()
