"""Central logging configuration (console + rotating file)."""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

from config.settings import LOG_LEVEL, LOGS_DIR

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        root = logging.getLogger()
        root.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
        fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        root.addHandler(console)
        file_handler = RotatingFileHandler(LOGS_DIR / "app.log", maxBytes=2_000_000,
                                           backupCount=3, encoding="utf-8")
        file_handler.setFormatter(fmt)
        root.addHandler(file_handler)
        _CONFIGURED = True
    return logging.getLogger(name)