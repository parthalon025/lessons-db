"""Central configuration for lessons-db."""

from pathlib import Path

# Data directory
DATA_DIR = Path.home() / ".local" / "share" / "lessons-db"
SQLITE_PATH = DATA_DIR / "lessons.db"
LANCE_DIR = DATA_DIR / "lance"
RULES_DIR = DATA_DIR / "rules"

# Source lesson files (for migration)
LESSONS_SOURCE_DIR = Path.home() / "Documents" / "docs" / "lessons"

# Ollama queue API (embeddings + analysis)
OLLAMA_QUEUE_URL = "http://127.0.0.1:7683"
EMBED_MODEL = "nomic-embed-text"
EMBED_DIMS = 768
ANALYSIS_MODEL = "qwen2.5:7b"

# Thresholds
DEDUP_THRESHOLD = 0.85
QUALITY_MIN_SCORE = 3
NEAR_MISS_TOP_N = 10

# Semgrep
SEMGREP_RULES_DIR = DATA_DIR / "rules"

# Valid enums (from FRAMEWORK.md)
VALID_TIERS = ("observation", "insight", "lesson", "lesson_learned")
VALID_CATEGORIES = (
    "data-model", "registration", "cold-start", "integration",
    "deployment", "monitoring", "ui", "testing", "performance", "security",
)
VALID_CLUSTERS = ("A", "B", "C", "D", "E", "F")
VALID_ENFORCEMENT = (
    "documentation", "semgrep_warning", "semgrep_error", "semgrep_autofix",
)
VALID_SOURCES = (
    "manual", "auto_diff", "auto_transcript", "auto_test", "community", "migrated",
)
