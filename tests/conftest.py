"""Shared test fixtures for lessons-db."""

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Temporary data directory for test isolation."""
    return tmp_path


@pytest.fixture
def db_path(tmp_data_dir):
    """Path to a temporary SQLite database."""
    return tmp_data_dir / "test_lessons.db"


@pytest.fixture
def lance_dir(tmp_data_dir):
    """Path to a temporary LanceDB directory."""
    d = tmp_data_dir / "lance"
    d.mkdir()
    return d
