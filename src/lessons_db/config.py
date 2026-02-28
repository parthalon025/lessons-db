"""Central configuration for lessons-db."""

import os
from pathlib import Path

# Data directory
DATA_DIR = Path(
    os.environ.get(
        "LESSONS_DB_DATA_DIR",
        str(Path.home() / ".local" / "share" / "lessons-db"),
    )
)
SQLITE_PATH = DATA_DIR / "lessons.db"
LANCE_DIR = DATA_DIR / "lance"
RULES_DIR = DATA_DIR / "rules"

# Source lesson files (for migration)
LESSONS_SOURCE_DIR = Path(
    os.environ.get(
        "LESSONS_DB_SOURCE_DIR",
        str(Path.home() / "Documents" / "docs" / "lessons"),
    )
)

# Cross-project scanner
PROJECTS_DIR = Path(os.environ.get("LESSONS_DB_PROJECTS_DIR", str(Path.home() / "Documents" / "projects")))

# Ollama queue API (generation / analysis tasks) — used for queue-aware callers
OLLAMA_QUEUE_URL = os.environ.get("LESSONS_DB_OLLAMA_QUEUE_URL", "http://127.0.0.1:7683")

# Ollama direct API — bypasses queue, used for embeddings and batch analysis
OLLAMA_EMBED_URL = os.environ.get("LESSONS_DB_OLLAMA_EMBED_URL", "http://127.0.0.1:11434")
OLLAMA_ANALYSIS_URL = os.environ.get("LESSONS_DB_OLLAMA_ANALYSIS_URL", "http://127.0.0.1:11434")

EMBED_MODEL = "nomic-embed-text"
EMBED_DIMS = 768
ANALYSIS_MODEL = os.environ.get("LESSONS_DB_OLLAMA_ANALYSIS_MODEL", "qwen2.5:7b")

# Thresholds
DEDUP_THRESHOLD = 0.85
QUALITY_MIN_SCORE = 3
NEAR_MISS_TOP_N = 10

# Semgrep
SEMGREP_RULES_DIR = DATA_DIR / "rules"

# Positive promotion thresholds
PROMOTION_TESTED_THRESHOLD = 1  # reuse_count >= 1 → tested
PROMOTION_TEMPLATE_THRESHOLD = 2  # reuse_count >= 2 → proven, template generated
PROMOTION_STANDARD_THRESHOLD = 3  # reuse_count >= 3 → standard

# Valid enums (negative OIL ladder)
VALID_TIERS_NEGATIVE = ("observation", "insight", "lesson", "lesson_learned")

# Valid enums (positive ladder)
VALID_TIERS_POSITIVE = ("noticed", "tested", "proven", "standard")

# Combined
VALID_TIERS = VALID_TIERS_NEGATIVE + VALID_TIERS_POSITIVE

VALID_CATEGORIES_NEGATIVE = (
    "data-model",
    "registration",
    "cold-start",
    "integration",
    "deployment",
    "monitoring",
    "ui",
    "testing",
    "performance",
    "security",
)
VALID_CATEGORIES_POSITIVE = (
    "architecture-pattern",
    "planning-technique",
    "workflow-optimization",
    "value-multiplier",
    "debugging-strategy",
    "testing-pattern",
    "integration-approach",
    "tooling-innovation",
)
VALID_CATEGORIES = VALID_CATEGORIES_NEGATIVE + VALID_CATEGORIES_POSITIVE

VALID_CLUSTERS = ("A", "B", "C", "D", "E", "F")  # Historical seeds only
VALID_POLARITIES = ("negative", "positive")
VALID_ENTRY_TYPES = ("lesson", "insight", "pattern", "innovation")
VALID_ENFORCEMENT = (
    "documentation",
    "semgrep_warning",
    "semgrep_error",
    "semgrep_autofix",
)
VALID_SOURCES = (
    "manual",
    "auto_diff",
    "auto_transcript",
    "auto_transcript_positive",
    "auto_test",
    "community",
    "migrated",
    "auto_design_doc",
    "auto_plan",
    "semgrep_registry",
)

# OpenAI API config (for draft triage reviewer)
OPENAI_REVIEW_MODEL = os.environ.get("LESSONS_DB_OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Triage log directory — JSONL verdict logs written here by capture review
TRIAGE_LOG_DIR = DATA_DIR
