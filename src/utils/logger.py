"""src/utils/logger.py

Centralised, structured logging factory.
Every module should call ``get_logger(__name__)`` rather than
configuring its own handler.
"""

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_INITIALIZED: bool = False


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
) -> None:
    """Configure root logger once for the whole application.

    Parameters
    ----------
    level:
        Logging level string, e.g. "DEBUG", "INFO", "WARNING".
    log_file:
        Optional file path.  If provided, logs are written to both
        stdout and the file.

    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
    )

    # Suppress noisy third-party loggers
    for noisy in ("py4j", "pyspark", "urllib3", "prophet", "cmdstanpy"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _INITIALIZED = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger, initialising root logging if necessary.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger

    """
    if not _INITIALIZED:
        setup_logging()
    return logging.getLogger(name)
