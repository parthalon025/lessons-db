"""GitHub repository miner.

Uses PyDriller to traverse commits, extract bug-fix diffs, and feed them
through the lessons-db capture pipeline.

Key behaviors:
- Mines BOTH polarity=negative (anti-patterns) AND polarity=positive (novel methods)
- Always captures errors at commit level → mining_runs.error_count
- Broad scope: categories not limited to user stack
- Compound commit filter: conventional commits + issue refs + test changes

Nightly entry point: mine_repos_for_gaps(conn, lance_dir, gap_categories)
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import requests

from lessons_db.config import ANALYSIS_MODEL, OLLAMA_QUEUE_URL
from lessons_db.db import insert_mined_repo, update_mined_repo

_log = logging.getLogger(__name__)

# Commit filter patterns
_BUG_FIX_CONVENTIONAL = re.compile(r"^(fix|bug|patch|hotfix|resolve|correction)(\(.+\))?:", re.IGNORECASE)
_BUG_FIX_ISSUE_REF = re.compile(r"(Fixes|Resolves|Closes):\s*#\d+", re.IGNORECASE)
_BUG_FIX_KEYWORD = re.compile(r"\b(fix|bug|error|patch|correct|resolve)\b", re.IGNORECASE)


@dataclass
class MiningConfig:
    min_stars: int = 50
    max_age_days: int = 180
    min_diff_lines: int = 5
    max_diff_lines: int = 200
    max_commits_per_repo: int = 500
    capture_positive: bool = True  # Always capture positive novel methods
    capture_errors: bool = True  # Always log errors as lesson candidates
    topics: list[str] = field(
        default_factory=lambda: [
            "asyncio",
            "home-assistant",
            "fastapi",
            "sqlalchemy",
            "python",
            "security",
            "performance",
        ]
    )


def is_bug_fix_commit(message: str) -> bool:
    """Return True if commit message indicates a bug fix.

    HIGH confidence: conventional commit + issue reference
    MEDIUM: conventional commit alone
    LOW: keyword only — still returns True but confidence is lower
    """
    if _BUG_FIX_CONVENTIONAL.search(message):
        return True
    if _BUG_FIX_ISSUE_REF.search(message):
        return True
    return bool(_BUG_FIX_KEYWORD.search(message))


def filter_diff_by_size(diff: str, min_lines: int = 5, max_lines: int = 200) -> bool:
    """Return True if diff is within [min_lines, max_lines] changed lines."""
    changed = sum(1 for line in diff.splitlines() if line.startswith("+") or line.startswith("-"))
    return min_lines <= changed <= max_lines


def _call_ollama_extract(diff: str, source_repo: str) -> list[dict]:
    """Call ollama-queue to extract polarized lesson candidates from a diff.

    Returns list of candidate dicts with polarity field.
    Always requests both positive and negative patterns.
    """
    prompt = f"""Analyze this git diff from {source_repo}. Extract coding lessons.

For each pattern found, output a JSON object with these fields:
- polarity: "negative" (anti-pattern to avoid) OR "positive" (novel/elegant method to adopt)
- title: short descriptive title
- one_liner: one sentence describing the lesson
- bad_code: the problematic code snippet (for negative) OR "N/A" (for positive)
- good_code: the correct/better code snippet
- category: one of: security, performance, db-queries, async, testing, integration,
  architecture-pattern, tooling-innovation

Extract ALL patterns — anti-patterns AND positive novel methods. Respond with a JSON array only.

Diff:
```
{diff[:3000]}
```"""

    try:
        resp = requests.post(
            f"{OLLAMA_QUEUE_URL}/api/generate",
            json={"model": ANALYSIS_MODEL, "prompt": prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        raw = resp.json().get("response", "")
        # Extract JSON array from response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        return json.loads(match.group())
    except Exception as exc:
        _log.warning("ollama extract failed for %s: %s", source_repo, exc)
        return []


def extract_polarized_candidates(
    conn: sqlite3.Connection,
    diff: str,
    source_repo: str,
) -> list[dict]:
    """Extract both positive and negative lesson candidates from a diff."""
    candidates = _call_ollama_extract(diff, source_repo)
    # Ensure polarity field present
    for c in candidates:
        if "polarity" not in c:
            c["polarity"] = "negative"
        c["source_repo"] = source_repo
    return candidates


def discover_repos(topics: list[str], min_stars: int = 50) -> list[str]:
    """Discover repos via gh CLI. Returns list of 'owner/repo' strings."""
    import subprocess

    repos = set()
    for topic in topics[:3]:  # Limit gh API calls
        try:
            gh_cmd = [  # noqa: S607
                "gh",
                "search",
                "repos",
                f"--topic={topic}",
                "--language=python",
                "--json",
                "nameWithOwner,stargazersCount,updatedAt",
                "--sort",
                "stars",
                "--order",
                "desc",
                "--limit",
                "20",
            ]
            result = subprocess.run(  # noqa: S603
                gh_cmd, capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                _log.warning("gh search failed for topic %s: %s", topic, result.stderr)
                continue
            for repo in json.loads(result.stdout or "[]"):
                if repo.get("stargazersCount", 0) >= min_stars:
                    repos.add(repo["nameWithOwner"])
        except Exception as exc:
            _log.warning("repo discovery failed for topic %s: %s", topic, exc)

    return list(repos)


def _process_modification(
    conn: sqlite3.Connection,
    diff: str,
    repo_name: str,
    cfg: MiningConfig,
    stats: dict,
) -> int:
    """Process one file modification from a commit. Returns lesson count extracted."""
    from lessons_db.capture import capture_from_diff
    from lessons_db.pattern_validator import validate_regex_self_consistency, validate_syntax

    lessons_extracted = 0
    candidates = extract_polarized_candidates(conn, diff, repo_name)
    for candidate in candidates:
        # Gate 0a: syntax check
        syn = validate_syntax(
            title=candidate.get("title", ""),
            one_liner=candidate.get("one_liner", ""),
            bad_code=candidate.get("bad_code", ""),
            good_code=candidate.get("good_code", ""),
            regex=candidate.get("regex"),
        )
        if not syn["passed"]:
            stats["gate0_rejected"] += 1
            continue

        # Gate 0b: regex self-consistency
        if candidate.get("regex"):
            reg = validate_regex_self_consistency(
                candidate["regex"],
                candidate.get("bad_code", ""),
                candidate.get("good_code", ""),
            )
            if not reg["passed"]:
                stats["gate0_rejected"] += 1
                continue

        # Feed into existing capture pipeline
        capture_from_diff(diff, conn)
        stats["auto_approved"] += 1
        lessons_extracted += 1

    return lessons_extracted


def mine_repos_for_gaps(
    conn: sqlite3.Connection,
    lance_dir: Path,
    gap_categories: list[str] | None = None,
    config: MiningConfig | None = None,
) -> dict:
    """Main entry point: mine repos targeting the identified gaps.

    Returns summary dict for mining_runs insertion.
    """
    from pydriller import Repository

    cfg = config or MiningConfig()
    repos = discover_repos(cfg.topics, cfg.min_stars)

    stats = {
        "repos_searched": 0,
        "commits_analyzed": 0,
        "gate0_rejected": 0,
        "gate1_rejected": 0,
        "auto_approved": 0,
        "conflicts_flagged": 0,
        "error_count": 0,
    }

    since = datetime.now() - timedelta(days=cfg.max_age_days)

    for repo_name in repos:
        stats["repos_searched"] += 1
        insert_mined_repo(conn, repo_name)

        try:
            commits_in_repo = 0
            lessons_in_repo = 0

            for commit in Repository(
                f"https://github.com/{repo_name}",
                since=since,
                only_no_merge=True,
                only_modifications_with_file_types=[".py"],
            ).traverse_commits():
                if commits_in_repo >= cfg.max_commits_per_repo:
                    break
                if not is_bug_fix_commit(commit.msg):
                    continue

                stats["commits_analyzed"] += 1
                commits_in_repo += 1

                for mod in commit.modifications:
                    diff = mod.diff or ""
                    if not filter_diff_by_size(diff, cfg.min_diff_lines, cfg.max_diff_lines):
                        stats["gate0_rejected"] += 1
                        continue

                    try:
                        lessons_in_repo += _process_modification(conn, diff, repo_name, cfg, stats)
                    except Exception as exc:
                        if cfg.capture_errors:
                            _log.warning("commit error in %s: %s", repo_name, exc)
                            stats["error_count"] += 1

            update_mined_repo(conn, repo_name, commit_count=commits_in_repo, lessons_extracted=lessons_in_repo)

        except Exception as exc:
            _log.error("repo mining failed for %s: %s", repo_name, exc)
            stats["error_count"] += 1

    return stats
