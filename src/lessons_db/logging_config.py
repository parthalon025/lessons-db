"""Centralized logging configuration for lessons-db."""

import logging
import logging.handlers
from pathlib import Path


LOG_FILE = Path.home() / ".local" / "share" / "lessons-db" / "lessons-db.log"
LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def configure_logging(level: int = logging.WARNING, verbose: bool = False) -> None:
    """Configure root logger with console handler and rotating file handler.

    File: ~/.local/share/lessons-db/lessons-db.log
    Max size: 1MB × 3 backup files = ~4MB cap total
    Console: level passed in (WARNING by default, DEBUG if verbose)
    File: always DEBUG — captures everything for post-hoc diagnosis
    """
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # handlers filter individually

    # Console handler — respects --verbose flag
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root.addHandler(console)

    # Rotating file handler — always DEBUG
    if not any(isinstance(h, logging.handlers.RotatingFileHandler) for h in root.handlers):
        fh = logging.handlers.RotatingFileHandler(
            LOG_FILE,
            maxBytes=1_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        root.addHandler(fh)
