#!/usr/bin/env python3
"""Clean up test data from integration smoke tests.

Deletes:
  - Stops running Qdrant and MemoryBridge services
  - Qdrant vectors for test agents (agent-smoke, agent-other)
  - Mem0 SQLite history database
  - Qdrant storage directory

Usage:
  bash tests/integration/cleanup.sh
  uv run python tests/integration/cleanup.py
"""

import os
import shutil
import signal
import socket
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _kill_services() -> None:
    """Kill any running MemoryBridge or Qdrant processes."""
    for port in (8000, 6333):
        try:
            result: subprocess.CompletedProcess[str] = subprocess.run(
                ["lsof", "-ti", f":{port}"], capture_output=True, text=True
            )
            for pid_str in result.stdout.strip().splitlines():
                pid: int = int(pid_str)
                os.kill(pid, signal.SIGTERM)
                print(f"  ✓ killed pid {pid} on port {port}")
        except (subprocess.SubprocessError, ValueError, ProcessLookupError):
            pass
    import time
    time.sleep(1)


def _qdrnt_running() -> bool:
    """Check if Qdrant is reachable on localhost:6333."""
    try:
        with socket.create_connection(("localhost", 6333), timeout=1):
            return True
    except OSError:
        return False


def _cleanup_via_api() -> None:
    from mem0 import Memory

    agent_ids: list[str] = ["agent-smoke", "agent-other"]

    config: dict[str, object] = {
        "llm": {
            "provider": "deepseek",
            "config": {
                "model": "deepseek-chat",
                "api_key": os.environ["DEEPSEEK_API_KEY"],
                "deepseek_base_url": "https://api.deepseek.com",
                "temperature": 0.2,
                "max_tokens": 2000,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-v4",
                "api_key": os.environ["DASHSCOPE_API_KEY"],
                "openai_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                "embedding_dims": 1024,
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "memory_bridge",
                "embedding_model_dims": 1024,
                "on_disk": True,
            },
        },
    }

    memory: Memory = Memory.from_config(config)

    for agent_id in agent_ids:
        print(f"Deleting memories for agent: {agent_id}")
        try:
            memory.delete_all(user_id=agent_id)
            print("  ✓ done")
        except Exception as e:
            print(f"  ✗ error: {e}")

    memory.close()


def main() -> None:
    if not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("DASHSCOPE_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY and DASHSCOPE_API_KEY must be set in .env file.")
        sys.exit(1)

    print("Stopping services ...")
    _kill_services()

    has_data: bool = (
        Path("data/mem0_history.db").exists()
        or Path("data/qdrant").exists()
    )
    if not has_data:
        print("No data found. Nothing to clean up.")
        return

    print("This will delete ALL data in:")
    if Path("data/qdrant").exists():
        print("  data/qdrant/ (Qdrant vector storage)")
    if Path("data/mem0_history.db").exists():
        print("  data/mem0_history.db (Mem0 history)")
    print()
    answer: str = input("Type 'yes' to confirm: ")
    if answer.strip() != "yes":
        print("Aborted.")
        sys.exit(0)

    if _qdrnt_running():
        print("Qdrant is running — using API to delete vectors ...")
        _cleanup_via_api()
    else:
        print("Qdrant is not running — skipping API cleanup.")

    # Remove Mem0 SQLite history DB
    history_db: Path = Path("data/mem0_history.db")
    if history_db.exists():
        history_db.unlink()
        print("  ✓ removed data/mem0_history.db")

    # Remove Qdrant storage directory
    qdrant_dir: Path = Path("data/qdrant")
    if qdrant_dir.exists():
        shutil.rmtree(qdrant_dir)
        print("  ✓ removed data/qdrant/")

    # Ensure data dir exists for next run
    Path("data").mkdir(exist_ok=True)

    print()
    print("Cleanup complete.")


if __name__ == "__main__":
    main()
