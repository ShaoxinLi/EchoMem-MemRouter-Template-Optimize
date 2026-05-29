"""Centralized logging configuration for EchoMem."""

import logging
import sys

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-30s | %(message)s"


def configure_logging(level: int = logging.INFO, fmt: str = DEFAULT_LOG_FORMAT) -> None:
    """Configure root logger with consistent formatting.

    Args:
        level: Logging level (default: INFO).
        fmt: Log format string.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("echomem")
    root.setLevel(level)
    root.handlers = []
    root.addHandler(handler)
