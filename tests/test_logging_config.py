"""Tests for logging configuration."""

import logging
import logging.handlers
from pathlib import Path

import pytest

from lessons_db.logging_config import configure_logging, LOG_FILE


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset root logger handlers between tests."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.level = original_level


class TestConfigureLogging:
    def test_adds_rotating_file_handler(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.logging_config.LOG_FILE",
                            tmp_path / "test.log")
        root = logging.getLogger()
        # Clear handlers for clean test
        root.handlers.clear()
        configure_logging()
        assert any(isinstance(h, logging.handlers.RotatingFileHandler)
                   for h in root.handlers)

    def test_file_handler_debug_level(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.logging_config.LOG_FILE",
                            tmp_path / "test.log")
        root = logging.getLogger()
        root.handlers.clear()
        configure_logging(level=logging.WARNING)
        fh = next(h for h in root.handlers
                  if isinstance(h, logging.handlers.RotatingFileHandler))
        assert fh.level == logging.DEBUG  # file always captures everything

    def test_console_handler_respects_level(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.logging_config.LOG_FILE",
                            tmp_path / "test.log")
        root = logging.getLogger()
        root.handlers.clear()
        configure_logging(level=logging.DEBUG)
        stream_handlers = [h for h in root.handlers
                           if isinstance(h, logging.StreamHandler)
                           and not isinstance(h, logging.handlers.RotatingFileHandler)]
        assert len(stream_handlers) == 1
        assert stream_handlers[0].level == logging.DEBUG

    def test_creates_log_directory(self, tmp_path, monkeypatch):
        log_path = tmp_path / "nested" / "dir" / "app.log"
        monkeypatch.setattr("lessons_db.logging_config.LOG_FILE", log_path)
        root = logging.getLogger()
        root.handlers.clear()
        configure_logging()
        assert log_path.parent.exists()

    def test_idempotent_second_call_does_not_duplicate_handlers(self, tmp_path, monkeypatch):
        monkeypatch.setattr("lessons_db.logging_config.LOG_FILE",
                            tmp_path / "test.log")
        root = logging.getLogger()
        root.handlers.clear()
        configure_logging()
        configure_logging()
        fh_count = sum(1 for h in root.handlers
                       if isinstance(h, logging.handlers.RotatingFileHandler))
        assert fh_count == 1
