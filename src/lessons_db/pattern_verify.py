"""Cross-project pattern verification — Stage 2.

Takes a CandidatePattern from Stage 1 and runs it through three gates:

1. LanceDB dedup — skip if a near-identical lesson already exists.
2. Suppression check — skip if snippet resembles a previously rejected pattern.
3. Ollama two-pass scoring — specificity (is it reusable?) then generality
   (does it solve a real cross-project problem?).

Returns a VerifiedCandidate on success, None if any gate rejects.
"""

import logging
import re
import sqlite3
from dataclasses import dataclass, field

import requests

from lessons_db.config import ANALYSIS_MODEL, OLLAMA_ANALYSIS_URL
from lessons_db.pattern_extract import CandidatePattern
from lessons_db.vectors import cosine_similarity, get_embedding, init_lance, semantic_search

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEDUP_DISTANCE_THRESHOLD = 0.15   # LanceDB L2 distance; lower = more similar
SUPPRESSION_SIMILARITY = 0.85     # cosine similarity; above = suppressed
SPECIFICITY_MIN = 0.4             # reject before generality if below this


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class VerifiedCandidate:
    snippet: str
    source_repos: list[str]
    source_lesson_id: int | None
    confidence: float
    rationale: str


# ---------------------------------------------------------------------------
# LanceDB helper
# ---------------------------------------------------------------------------

def nearest_lessons(snippet: str, lance_dir: str, k: int = 3) -> list[dict]:
    """Return top-k nearest lessons from LanceDB.

    Returns empty list on any error (missing table, embedding failure, etc.).
    """
    try:
        db = init_lance(lance_dir)
        return semantic_search(db, snippet, top_k=k)
    except Exception as exc:
        _log.warning("nearest_lessons failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Suppression helpers
# ---------------------------------------------------------------------------

def _suppression_similarity(snippet: str, conn, lance_dir: str) -> float:
    """Return max cosine similarity between snippet and all rejected snippets.

    Queries suppression_vectors for all rejected_snippet values, embeds each,
    computes cosine similarity against the snippet embedding, returns the
    maximum (0.0 if no vectors or embedding fails).

    Exposed as a separate function so tests can patch it independently.
    """
    try:
        rows = conn.execute(
            "SELECT rejected_snippet FROM suppression_vectors LIMIT 500"
        ).fetchall()
    except sqlite3.OperationalError as exc:
        _log.warning("suppression_vectors query failed: %s", exc)
        return 0.0

    if not rows:
        return 0.0

    snippet_vec = get_embedding(snippet)
    if snippet_vec is None:
        return 0.0

    max_sim = 0.0
    for row in rows:
        rejected = row[0] if isinstance(row, (tuple, list)) else row["rejected_snippet"]
        rejected_vec = get_embedding(rejected)
        if rejected_vec is None:
            continue
        sim = cosine_similarity(snippet_vec, rejected_vec)
        if sim > max_sim:
            max_sim = sim

    return max_sim


def is_suppressed(snippet: str, conn, lance_dir: str) -> bool:
    """Return True if snippet is too similar to a previously rejected pattern."""
    return _suppression_similarity(snippet, conn, lance_dir) >= SUPPRESSION_SIMILARITY


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_float(text: str, default: float = 0.5) -> float:
    """Extract first probability float from text. Returns default on failure.

    Only matches valid probability strings: 0.xxx, 1.0, 1, or 0 as standalone
    words. Avoids false matches like "1 out of 10" or bare integers > 1.
    """
    m = re.search(r"\b(?:1(?:\.0+)?|0(?:\.\d+)?)\b", text)
    if m:
        return float(m.group())
    return default


def _parse_score_and_rationale(text: str) -> tuple[float, str]:
    """Split text on first newline: first line → score, rest → rationale."""
    parts = text.split("\n", 1)
    score = _parse_float(parts[0].strip())
    rationale = parts[1].strip() if len(parts) > 1 else ""
    return score, rationale


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _specificity_prompt(snippet: str, neighbors: list[dict]) -> str:
    context_lines = []
    for n in neighbors:
        text = n.get("text", "")
        if text:
            context_lines.append(f"- {text[:120]}")
    context_text = "\n".join(context_lines) if context_lines else "(none)"

    return (
        "Rate this code snippet's reusability across projects.\n"
        "0.0 = entirely project-specific, 1.0 = universally reusable.\n\n"
        "Similar patterns already captured:\n"
        f"{context_text}\n\n"
        "Code snippet:\n"
        "```\n"
        f"{snippet}\n"
        "```\n\n"
        "Respond with only a decimal score 0.0-1.0."
    )


def _generality_prompt_with_lesson(
    snippet: str, lesson_id: int, one_liner: str
) -> str:
    return (
        f"This was found while searching for correct implementations of "
        f"lesson #{lesson_id}: {one_liner}.\n"
        "Does this code correctly apply the fix?\n\n"
        "Code:\n"
        "```\n"
        f"{snippet}\n"
        "```\n\n"
        "Respond with a decimal score 0.0-1.0 on the first line, then one "
        "sentence explaining why on the second line."
    )


def _generality_prompt_no_lesson(snippet: str, source_repos: list[str]) -> str:
    repos_list = ", ".join(source_repos)
    return (
        f"This code appears in repos: {repos_list}.\n"
        "Does its presence in multiple repos suggest it solves a general "
        "problem worth capturing?\n\n"
        "Code:\n"
        "```\n"
        f"{snippet}\n"
        "```\n\n"
        "Respond with a decimal score 0.0-1.0 on the first line, then one "
        "sentence explaining why on the second line."
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def verify_candidate(
    candidate: CandidatePattern,
    conn,
    lance_dir: str,
) -> VerifiedCandidate | None:
    """Run a CandidatePattern through all verification gates.

    Returns a VerifiedCandidate on success, None if any gate rejects.

    Gate order:
    1. LanceDB dedup (distance < DEDUP_DISTANCE_THRESHOLD → reject)
    2. Suppression check (similarity >= SUPPRESSION_SIMILARITY → reject)
    3. Ollama specificity (score < SPECIFICITY_MIN → reject)
    4. Ollama generality (parse score + rationale)
    5. Compute confidence = specificity*0.4 + generality*0.6
    """
    snippet = candidate.snippet

    # Gate 1: LanceDB dedup
    neighbors = nearest_lessons(snippet, lance_dir)
    if neighbors and neighbors[0]["score"] < DEDUP_DISTANCE_THRESHOLD:
        _log.debug("verify_candidate: dedup match (score=%.3f), skipping",
                   neighbors[0]["score"])
        return None

    # Gate 2: Suppression
    if is_suppressed(snippet, conn, lance_dir):
        _log.debug("verify_candidate: suppressed snippet, skipping")
        return None

    # Gate 3: Specificity via Ollama
    spec_prompt = _specificity_prompt(snippet, neighbors)
    try:
        spec_resp = requests.post(
            f"{OLLAMA_ANALYSIS_URL}/api/generate",
            json={"model": ANALYSIS_MODEL, "prompt": spec_prompt, "stream": False},
            timeout=30,
        )
        spec_resp.raise_for_status()
        spec_text = spec_resp.json().get("response", "")
        specificity = _parse_float(spec_text.strip())
    except Exception as exc:
        _log.warning("verify_candidate: specificity Ollama call failed: %s", exc)
        return None

    if specificity < SPECIFICITY_MIN:
        _log.debug("verify_candidate: specificity %.2f below threshold, skipping",
                   specificity)
        return None

    # Gate 4: Generality via Ollama
    if candidate.source_lesson_id is not None:
        try:
            row = conn.execute(
                "SELECT one_liner FROM lessons WHERE id = ?",
                (candidate.source_lesson_id,),
            ).fetchone()
            one_liner = row[0] if row else ""
        except sqlite3.OperationalError as exc:
            _log.warning("verify_candidate: lessons lookup failed: %s", exc)
            one_liner = ""
        gen_prompt = _generality_prompt_with_lesson(
            snippet, candidate.source_lesson_id, one_liner
        )
    else:
        gen_prompt = _generality_prompt_no_lesson(snippet, candidate.source_repos)

    try:
        gen_resp = requests.post(
            f"{OLLAMA_ANALYSIS_URL}/api/generate",
            json={"model": ANALYSIS_MODEL, "prompt": gen_prompt, "stream": False},
            timeout=30,
        )
        gen_resp.raise_for_status()
        gen_text = gen_resp.json().get("response", "")
        generality, rationale = _parse_score_and_rationale(gen_text.strip())
    except Exception as exc:
        _log.warning("verify_candidate: generality Ollama call failed: %s", exc)
        return None

    # Gate 5: Confidence
    confidence = round(specificity * 0.4 + generality * 0.6, 6)

    return VerifiedCandidate(
        snippet=snippet,
        source_repos=candidate.source_repos,
        source_lesson_id=candidate.source_lesson_id,
        confidence=confidence,
        rationale=rationale,
    )
