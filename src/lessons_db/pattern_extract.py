"""Cross-project pattern extraction — Stage 1.

Identifies candidate anti-patterns / best-practice patterns that appear in
two or more project repositories. Detection runs via two complementary paths:

1. Semgrep (Python): pattern-matched findings grouped by repo.
2. Embedding similarity (non-Python): 15-line sliding-window blocks clustered
   by cosine similarity across repos.
"""

import json
import logging
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import requests

from lessons_db.config import (
    ANALYSIS_MODEL,
    OLLAMA_ANALYSIS_URL,
    PROJECTS_DIR,
)
from lessons_db.vectors import get_embedding

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class CandidatePattern:
    snippet: str
    source_repos: list[str]
    source_lesson_id: int | None = None
    pattern_id: str = ""
    detection_method: str = "semgrep"


# ---------------------------------------------------------------------------
# Bootstrap patterns (used when DB has fewer than 10 corrective actions)
# ---------------------------------------------------------------------------

BOOTSTRAP_PATTERNS: list[dict] = [
    {
        "id": "error-handler-fallback",
        "source_lesson_id": None,
        "yaml": """rules:
  - id: error-handler-fallback
    patterns:
      - pattern: |
          try:
            ...
          except ...:
            $LOG(...)
            return $DEFAULT
    message: Exception handler logs and returns a fallback value
    languages: [python]
    severity: INFO
""",
    },
    {
        "id": "retry-with-backoff",
        "source_lesson_id": None,
        "yaml": """rules:
  - id: retry-with-backoff
    patterns:
      - pattern: |
          for ... in ...:
            ...
            time.sleep(...)
    message: Retry loop with sleep (possible backoff pattern)
    languages: [python]
    severity: INFO
""",
    },
    {
        "id": "env-config-loader",
        "source_lesson_id": None,
        "yaml": """rules:
  - id: env-config-loader
    pattern: os.environ.get($KEY, $DEFAULT)
    message: Environment variable config loading
    languages: [python]
    severity: INFO
""",
    },
    {
        "id": "async-context-manager",
        "source_lesson_id": None,
        "yaml": """rules:
  - id: async-context-manager
    pattern: |
      async with $CTX as $VAR:
        ...
    message: Async context manager usage
    languages: [python]
    severity: INFO
""",
    },
    {
        "id": "click-command-with-error",
        "source_lesson_id": None,
        "yaml": """rules:
  - id: click-command-with-error
    patterns:
      - pattern: |
          @click.command(...)
          def $FUNC(...):
            try:
              ...
            except ...:
              ...
    message: Click command with top-level error handling
    languages: [python]
    severity: INFO
""",
    },
    {
        "id": "sqlite-context-manager",
        "source_lesson_id": None,
        "yaml": """rules:
  - id: sqlite-context-manager
    pattern: |
      with closing($CONN) as $C:
        ...
    message: sqlite3 connection wrapped in contextlib.closing
    languages: [python]
    severity: INFO
""",
    },
    {
        "id": "logger-setup",
        "source_lesson_id": None,
        "yaml": """rules:
  - id: logger-setup
    pattern: logging.getLogger($NAME)
    message: Module-level logger initialisation
    languages: [python]
    severity: INFO
""",
    },
    {
        "id": "dataclass-with-validation",
        "source_lesson_id": None,
        "yaml": """rules:
  - id: dataclass-with-validation
    patterns:
      - pattern: |
          @dataclass
          class $CLS:
            ...
            def __post_init__(self):
              ...
    message: Dataclass with __post_init__ validation
    languages: [python]
    severity: INFO
""",
    },
]

# ---------------------------------------------------------------------------
# Non-Python file extensions to scan
# ---------------------------------------------------------------------------

_NONPYTHON_EXTS = {".sh", ".yaml", ".yml", ".js", ".ts", ".json", ".toml"}

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def list_active_repos(since_timestamp: str) -> list[Path]:
    """Return project repos with at least one commit since *since_timestamp*.

    Iterates ``PROJECTS_DIR``, skips non-git directories, and queries git log
    for each. Only repos where ``git log --since`` yields non-empty output are
    included.
    """
    repos: list[Path] = []
    if not PROJECTS_DIR.is_dir():
        _log.warning("PROJECTS_DIR does not exist: %s", PROJECTS_DIR)
        return repos

    for candidate in sorted(PROJECTS_DIR.iterdir()):
        if not candidate.is_dir():
            continue
        if not (candidate / ".git").exists():
            continue

        try:
            result = subprocess.run(
                [
                    "git", "-C", str(candidate),
                    "log", f"--since={since_timestamp}",
                    "--oneline", "--quiet",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                repos.append(candidate)
        except Exception as exc:
            _log.warning("git log failed for %s: %s", candidate, exc)

    return repos


def build_semgrep_patterns(conn) -> list[dict]:
    """Build Semgrep pattern dicts from DB lessons or fall back to bootstrap.

    Queries lessons where ``corrective_action`` is non-empty and polarity is
    'negative'. If fewer than 10 such rows exist, returns BOOTSTRAP_PATTERNS
    immediately. Otherwise asks Ollama to convert each corrective action to a
    Semgrep YAML rule and returns those (with ``source_lesson_id`` set). If
    Ollama generates nothing useful, falls back to BOOTSTRAP_PATTERNS.
    """
    rows = conn.execute(
        """
        SELECT id, corrective_action
        FROM lessons
        WHERE corrective_action IS NOT NULL
          AND corrective_action != ''
          AND polarity = 'negative'
        """,
    ).fetchall()

    if len(rows) < 10:
        return BOOTSTRAP_PATTERNS

    patterns: list[dict] = []
    for row in rows:
        lesson_id = row["id"]
        corrective_action = row["corrective_action"]

        prompt = (
            f"Convert the following corrective action into a Semgrep YAML rule. "
            f"Output only the YAML, nothing else.\n\n"
            f"Corrective action: {corrective_action}"
        )
        try:
            resp = requests.post(
                f"{OLLAMA_ANALYSIS_URL}/api/generate",
                json={"model": ANALYSIS_MODEL, "prompt": prompt, "stream": False},
                timeout=30,
            )
            yaml_text = resp.json().get("response", "").strip()
            if yaml_text and yaml_text.upper() != "SKIP":
                patterns.append({
                    "id": f"lesson-{lesson_id}",
                    "source_lesson_id": lesson_id,
                    "yaml": yaml_text,
                })
        except Exception as exc:
            _log.warning("Ollama pattern generation failed for lesson %d: %s", lesson_id, exc)

    if not patterns:
        return BOOTSTRAP_PATTERNS

    return patterns


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_semgrep_pattern(yaml_text: str, target_dir: Path) -> list[dict]:
    """Write *yaml_text* to a temp file, run semgrep --json, return results.

    Returns an empty list on any error (missing binary, timeout, bad JSON).
    """
    semgrep_bin = shutil.which("semgrep")
    if semgrep_bin is None:
        _log.debug("semgrep binary not found; skipping pattern run")
        return []

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".yaml", mode="w", delete=False
        ) as tmp:
            tmp.write(yaml_text)
            tmp_path = Path(tmp.name)

        result = subprocess.run(
            [semgrep_bin, "--config", str(tmp_path), "--json", "--quiet", str(target_dir)],
            capture_output=True,
            text=True,
            timeout=300,
        )

        if result.returncode not in (0, 1):
            _log.debug("semgrep exited %d for pattern", result.returncode)
            return []

        if not result.stdout:
            return []

        data = json.loads(result.stdout)
        return data.get("results", [])

    except (json.JSONDecodeError, subprocess.TimeoutExpired, Exception) as exc:
        _log.debug("semgrep run failed: %s", exc)
        return []
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


def _sliding_window(lines: list[str], size: int = 15) -> Generator:
    """Yield consecutive *size*-line windows from *lines*.

    Windows where every line is blank are skipped.
    """
    for i in range(len(lines) - size + 1):
        window = lines[i: i + size]
        if any(line.strip() for line in window):
            yield window


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------


def extract_python_candidates(
    repos: list[Path],
    patterns: list[dict],
    conn,
) -> list[CandidatePattern]:
    """Run each semgrep pattern against all repos; yield candidates spanning 2+ repos.

    For each pattern:
    - Run semgrep against every repo.
    - Group matches by repo (using the repo path as the key).
    - If 2+ distinct repos matched, emit a CandidatePattern.
    """
    candidates: list[CandidatePattern] = []

    for pattern in patterns:
        yaml_text = pattern.get("yaml", "")
        if not yaml_text:
            continue

        source_lesson_id = pattern.get("source_lesson_id")
        pattern_id = pattern.get("id", "")

        # Map repo → list of result snippets
        repo_hits: dict[str, list[str]] = {}

        for repo in repos:
            results = _run_semgrep_pattern(yaml_text, repo)
            snippets = [
                r.get("extra", {}).get("lines", "") for r in results
            ]
            non_empty = [s for s in snippets if s]
            if non_empty:
                repo_hits[str(repo)] = non_empty

        if len(repo_hits) < 2:
            continue

        # Representative snippet: first result from the first matching repo
        first_repo = next(iter(repo_hits))
        snippet = repo_hits[first_repo][0]

        candidates.append(CandidatePattern(
            snippet=snippet,
            source_repos=list(repo_hits.keys()),
            source_lesson_id=source_lesson_id,
            pattern_id=pattern_id,
            detection_method="semgrep",
        ))

    return candidates


def extract_nonpython_candidates(
    repos: list[Path],
    conn,
    similarity_threshold: float = 0.80,
) -> list[CandidatePattern]:
    """Embed 15-line windows from non-Python files; cluster by cosine similarity.

    For each pair of windows that exceed *similarity_threshold* and come from
    different repos, emit a CandidatePattern. Windows with failed embeddings
    are skipped silently.
    """
    # Collect (repo_name, snippet, vector) tuples
    blocks: list[tuple[str, str, list[float]]] = []

    for repo in repos:
        for ext in _NONPYTHON_EXTS:
            for filepath in repo.rglob(f"*{ext}"):
                try:
                    text = filepath.read_text(errors="replace")
                except Exception as exc:
                    _log.debug("Cannot read %s: %s", filepath, exc)
                    continue

                lines = text.splitlines()
                for window in _sliding_window(lines, size=15):
                    snippet = "\n".join(window)
                    vector = get_embedding(snippet)
                    if vector is None:
                        continue
                    blocks.append((str(repo), snippet, vector))

    if not blocks:
        return []

    # Cluster by cosine similarity: O(n²) — acceptable for small repo counts
    used: set[int] = set()
    candidates: list[CandidatePattern] = []

    for i in range(len(blocks)):
        if i in used:
            continue
        repo_i, snippet_i, vec_i = blocks[i]
        cluster_repos: set[str] = {repo_i}
        cluster_indices: list[int] = [i]

        for j in range(i + 1, len(blocks)):
            if j in used:
                continue
            repo_j, _, vec_j = blocks[j]
            if repo_j == repo_i:
                continue  # same repo, skip
            sim = _cosine_similarity(vec_i, vec_j)
            if sim >= similarity_threshold:
                cluster_repos.add(repo_j)
                cluster_indices.append(j)

        if len(cluster_repos) >= 2:
            for idx in cluster_indices:
                used.add(idx)
            candidates.append(CandidatePattern(
                snippet=snippet_i,
                source_repos=list(cluster_repos),
                source_lesson_id=None,
                pattern_id="",
                detection_method="embedding",
            ))

    return candidates


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two equal-length float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
