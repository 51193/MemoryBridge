"""Logging setup for MemoryBridge."""

import logging
import os
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with uniform formatting.

    Format: [time] [level] [module] message

    Controlled by LOG_LEVEL env var (DEBUG / INFO / WARNING / ERROR).
    Defaults to INFO — debug output is opt-in.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level: int = getattr(logging, level.upper(), logging.INFO)

    handler: logging.StreamHandler = logging.StreamHandler(sys.stderr)  # type: ignore[type-arg]
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
