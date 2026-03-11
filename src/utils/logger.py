"""src/utils/logger.py.

Centralised, structured logging factory.
Every module should call ``get_logger(__name__)`` rather than
configuring its own handler.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"
_INITIALIZED: bool = False


def setup_logging(
    level: str = "INFO",
    log_file: Path | None = None,
    max_bytes: int = 5 * 1024 * 1024,  # 5 MB per file
    backup_count: int = 3,  # keep pipeline.log + 3 rotated copies
) -> None:
    """Configure root logger once for the whole application.

    Parameters
    ----------
    level:
        Logging level string, e.g. "DEBUG", "INFO", "WARNING".
    log_file:
        Optional file path.  A single rotating file is used rather than
        a new timestamped file per run, so logs accumulate in one place.
    max_bytes:
        Maximum size of a single log file before rotation.
    backup_count:
        Number of rotated backup files to retain.

    """
    global _INITIALIZED
    if _INITIALIZED:
        return

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        handlers.append(file_handler)

    logging.basicConfig(
        level=numeric_level,
        format=_LOG_FORMAT,
        datefmt=_DATE_FORMAT,
        handlers=handlers,
    )

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
