"""Unified logging configuration for QUANTAXIS.

Provides structured logging with rotation support.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(
    name: str = "quantaxis",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
    format_string: str | None = None,
) -> logging.Logger:
    """Configure and return a logger instance.

    Args:
        name: Logger name.
        level: Logging level (default: INFO).
        log_file: Optional file path for persistent logs.
        format_string: Optional custom format string.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = format_string or (
        "%(asctime)s | %(name)s | %(levelname)-8s | %(message)s"
    )
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler (optional)
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            str(path), maxBytes=10 * 1024 * 1024, backupCount=5,
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "quantaxis") -> logging.Logger:
    """Get a QUANTAXIS logger, creating it if needed."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name)
    return logger
