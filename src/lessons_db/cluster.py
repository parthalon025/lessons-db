"""Adaptive clustering pipeline using HDBSCAN on LanceDB embeddings.

The discover_clusters() function requires optional dependencies:
    pip install 'lessons-db[clustering]'

All other functions (extract_representative_terms, apply_cluster_proposals,
find_seed_overlap, get_cluster_history) have no extra dependencies.
"""

import json
import logging
from collections import Counter
from datetime import date

_log = logging.getLogger(__name__)

from lessons_db.config import ANALYSIS_MODEL, LANCE_DIR, OLLAMA_QUEUE_URL

# Words to ignore when extracting representative terms
_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "in",
    "on",
    "of",
    "to",
    "is",
    "are",
    "was",
    "be",
    "with",
    "for",
    "it",
    "this",
    "that",
    "not",
    "no",
    "from",
    "by",
    "at",
    "as",
    "but",
    "if",
    "so",
    "do",
    "use",
}


def extract_representative_terms(conn, lesson_ids: list[int], top_n: int = 5) -> list[str]:
    """Extract the most frequent non-stopword terms from one-liners + keywords."""
    if not lesson_ids:
        return []
    placeholders = ",".join("?" * len(lesson_ids))
    rows = conn.execute(
        f"SELECT one_liner, keywords FROM lessons WHERE id IN ({placeholders})",
        lesson_ids,
    ).fetchall()
    words = []
    for row in rows:
        text = (row["one_liner"] or "") + " " + (row["keywords"] or "")
        words.extend(text.lower().split())
    counter = Counter(w.strip(".,;:()[]") for w in words if w not in _STOPWORDS and len(w) > 2)
    return [w for w, _ in counter.most_common(top_n)]


def find_seed_overlap(conn, lesson_ids: list[int], threshold: float = 0.6) -> str | None:
    """Return the dominant cluster_seed if >= threshold fraction of lessons share it."""
    if not lesson_ids:
        return None
    placeholders = ",".join("?" * len(lesson_ids))
    rows = conn.execute(
        f"SELECT cluster_seed FROM lessons WHERE id IN ({placeholders}) AND cluster_seed IS NOT NULL",
        lesson_ids,
    ).fetchall()
    if not rows:
        return None
    counter = Counter(r["cluster_seed"] for r in rows)
    top_seed, top_count = counter.most_common(1)[0]
    if top_count / len(lesson_ids) >= threshold:
        return top_seed
    return None


def apply_cluster_proposals(conn, proposals: list[dict], confirmed: dict[int, str]) -> int:
    """Write confirmed cluster names to lessons.cluster. Records the run.

    proposals: list of {"cluster_id": int, "lesson_ids": [...], "suggested_name": str}
    confirmed: {cluster_id: final_name} — only these get written
    Returns count of updated lesson rows."""
    updated = 0
    for proposal in proposals:
        cid = proposal["cluster_id"]
        if cid not in confirmed:
            continue
        name = confirmed[cid]
        for lid in proposal["lesson_ids"]:
            conn.execute(
                "UPDATE lessons SET cluster = ? WHERE id = ?",
                [name, lid],
            )
            updated += 1

    conn.execute(
        "INSERT INTO cluster_runs (run_date, proposal_count, confirmed_count, result_json) " "VALUES (?, ?, ?, ?)",
        [
            date.today().isoformat(),
            len(proposals),
            len(confirmed),
            json.dumps(
                [
                    {"id": p["cluster_id"], "name": confirmed.get(p["cluster_id"]), "size": len(p["lesson_ids"])}
                    for p in proposals
                ]
            ),
        ],
    )
    conn.commit()
    return updated


def get_cluster_history(conn) -> list[dict]:
    """Return all past clustering runs in descending date order."""
    rows = conn.execute(
        "SELECT id, run_date, proposal_count, confirmed_count, result_json "
        "FROM cluster_runs ORDER BY run_date DESC, id DESC"
    ).fetchall()
    return [dict(r) for r in rows]


def generate_cluster_name(terms: list[str]) -> str:
    """Ask Ollama to generate a human-readable cluster name from terms.
    Falls back to joining the top 2 terms if Ollama unavailable."""
    import requests

    try:
        r = requests.post(
            f"{OLLAMA_QUEUE_URL}/api/generate",
            json={
                "model": ANALYSIS_MODEL,
                "prompt": (
                    f"Generate a 3-5 word cluster name for a group of software engineering "
                    f"lessons with these key terms: {', '.join(terms)}. "
                    "Respond with only the cluster name, no explanation."
                ),
                "stream": False,
            },
            timeout=30,
        )
        return r.json().get("response", "").strip() or f"{terms[0].title()} Patterns"
    except Exception:
        return " ".join(t.title() for t in terms[:2]) + " Patterns"


def discover_clusters(conn, min_cluster_size: int = 5) -> list[dict]:
    """Run HDBSCAN on LanceDB embeddings and return cluster proposals.

    Requires: pip install 'lessons-db[clustering]'

    Returns list of proposal dicts:
      {"cluster_id": int, "lesson_ids": [...], "suggested_name": str,
       "representative_terms": [...], "overlaps_seed": str | None}
    """
    try:
        import hdbscan
        import lancedb
        import numpy as np
        import umap
    except ImportError as e:
        raise RuntimeError(
            f"Clustering dependencies not installed ({e}). Run:\n" "pip install 'lessons-db[clustering]'"
        ) from e

    db = lancedb.connect(str(LANCE_DIR))
    try:
        table = db.open_table("lessons")
    except Exception as e:
        _log.debug("discover_clusters: could not open 'lessons' table: %s", e)
        return []

    rows = table.to_pandas()
    if len(rows) < min_cluster_size * 2:
        return []

    vectors = np.vstack(rows["vector"].values)
    lesson_ids = rows["lesson_id"].tolist()

    reducer = umap.UMAP(n_components=min(5, len(rows) - 1), random_state=42)
    reduced = reducer.fit_transform(vectors)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
    labels = clusterer.fit_predict(reduced)

    clusters: dict[int, list[int]] = {}
    for lid, label in zip(lesson_ids, labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(lid)

    proposals = []
    for label, ids in clusters.items():
        terms = extract_representative_terms(conn, ids)
        name = generate_cluster_name(terms)
        seed = find_seed_overlap(conn, ids)
        proposals.append(
            {
                "cluster_id": label,
                "lesson_ids": ids,
                "suggested_name": name,
                "representative_terms": terms,
                "overlaps_seed": seed,
            }
        )
    return proposals
