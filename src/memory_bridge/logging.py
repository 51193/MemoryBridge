"""Shared structured logging helpers for MemoryBridge.

These are at package root so all layers (core, providers, api) can import them
without creating upward dependencies between packages.
"""

import logging


def structured_debug(logger: logging.Logger, msg: str, **kv: object) -> None:
    """Log a DEBUG-level structured message with key=value pairs."""
    if not logger.isEnabledFor(logging.DEBUG):
        return
    parts: list[str] = [msg]
    for k, v in kv.items():
        parts.append(f"{k}={v!r}")
    logger.debug(" ".join(parts))


def structured_info(logger: logging.Logger, msg: str, **kv: object) -> None:
    """Log an INFO-level structured message with key=value pairs."""
    if not logger.isEnabledFor(logging.INFO):
        return
    parts: list[str] = [msg]
    for k, v in kv.items():
        parts.append(f"{k}={v!r}")
    logger.info(" ".join(parts))
