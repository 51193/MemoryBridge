"""Logging setup and structured log helpers for MemoryBridge."""

import logging
import os
import sys
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with uniform formatting.

    Format: [time] [level] [module] message

    Controlled by LOG_LEVEL env var (DEBUG / INFO / WARNING / ERROR).
    Defaults to INFO — debug output is opt-in.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level: int = getattr(logging, level.upper(), logging.INFO)

    handler: logging.StreamHandler[Any] = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s.%(msecs)03d [%(levelname)-5s] [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )

    root: logging.Logger = logging.getLogger("memory_bridge")
    root.setLevel(numeric_level)
    root.handlers.clear()
    root.addHandler(handler)
    root.propagate = False


def structured_debug(
    logger: logging.Logger,
    msg: str,
    **kv: object,
) -> None:
    """Log a DEBUG-level structured message with key=value pairs.

    Example:
        structured_debug(logger, "search complete", query="hello", results=3)
        → [DEBUG] search complete query="hello" results=3
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    parts: list[str] = [msg]
    for k, v in kv.items():
        parts.append(f"{k}={v!r}")
    logger.debug(" ".join(parts))


def structured_info(
    logger: logging.Logger,
    msg: str,
    **kv: object,
) -> None:
    """Log an INFO-level structured message with key=value pairs."""
    if not logger.isEnabledFor(logging.INFO):
        return
    parts: list[str] = [msg]
    for k, v in kv.items():
        parts.append(f"{k}={v!r}")
    logger.info(" ".join(parts))
