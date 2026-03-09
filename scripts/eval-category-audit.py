#!/usr/bin/env python3
"""Audit category ground truth quality for eval pipeline.

Compares within-category vs across-category embedding similarity
to validate that category is a good grouping signal.
"""

import secrets
import sqlite3
from pathlib import Path

import numpy as np

DB_PATH = Path.home() / ".local/share/lessons-db/lessons.db"
LANCE_PATH = Path.home() / ".local/share/lessons-db/lance"

_rng = secrets.SystemRandom()


def cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def load_vectors() -> dict[int, np.ndarray]:
    """Load embedding vectors from LanceDB, keyed by lesson_id."""
    import lancedb

    lance_db = lancedb.connect(str(LANCE_PATH))
    tbl = lance_db.open_table("lessons")
    embeddings_df = tbl.to_pandas()

    vectors = {}
    for _, row in embeddings_df.iterrows():
        vectors[int(row["lesson_id"])] = np.array(row["vector"], dtype=np.float32)
    return vectors


def sample_within_sims(cat_ids: list[int], vectors: dict[int, np.ndarray], n_pairs: int = 3) -> list[float]:
    """Sample cosine similarities between pairs within a category."""
    pairs = min(n_pairs, len(cat_ids) * (len(cat_ids) - 1) // 2)
    sampled: set[tuple[int, int]] = set()
    sims: list[float] = []

    for _ in range(pairs * 10):
        if len(sampled) >= pairs:
            break
        i, j = _rng.sample(cat_ids, 2)
        if (i, j) in sampled or (j, i) in sampled:
            continue
        sampled.add((i, j))
        sims.append(cosine_sim(vectors[i], vectors[j]))

    return sims


def sample_across_sims(
    cat_ids: list[int],
    other_ids: list[int],
    vectors: dict[int, np.ndarray],
    n_pairs: int = 3,
) -> list[float]:
    """Sample cosine similarities between a category and all others."""
    sims: list[float] = []
    for _ in range(n_pairs):
        if not cat_ids or not other_ids:
            break
        i = _rng.choice(cat_ids)
        j = _rng.choice(other_ids)
        sims.append(cosine_sim(vectors[i], vectors[j]))
    return sims


def print_summary(results: list[dict]) -> None:
    """Print summary statistics and flag bad categories."""
    print(f"\n{'=' * 75}")
    good = sum(1 for r in results if not r["flag"])
    bad = sum(1 for r in results if r["flag"])
    avg_sep = np.mean([r["separation"] for r in results])
    print(f"Good categories: {good}/{len(results)}  |  Bad: {bad}  |  Avg separation: {avg_sep:.3f}")

    if bad > 0:
        print("\nBAD categories (within < across — may need merging or splitting):")
        for r in results:
            if r["flag"]:
                print(f"  - {r['category']} (within={r['within']:.3f}, across={r['across']:.3f})")


def main():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    vectors = load_vectors()

    cats = conn.execute("""
        SELECT category, GROUP_CONCAT(id) as ids, COUNT(*) as cnt
        FROM lessons
        WHERE category IS NOT NULL
        GROUP BY category
        HAVING cnt >= 5
        ORDER BY cnt DESC
    """).fetchall()

    print(f"{'Category':40s} {'Within':>8s} {'Across':>8s} {'Sep':>6s} {'N':>4s} {'Flag':>5s}")
    print("-" * 75)

    all_ids = list(vectors.keys())
    results = []

    for cat_row in cats:
        cat = cat_row["category"]
        cat_ids = [int(x) for x in cat_row["ids"].split(",") if int(x) in vectors]

        if len(cat_ids) < 2:
            continue

        within_sims = sample_within_sims(cat_ids, vectors)
        other_ids = [x for x in all_ids if x not in set(cat_ids)]
        across_sims = sample_across_sims(cat_ids, other_ids, vectors)

        within_mean = np.mean(within_sims) if within_sims else 0
        across_mean = np.mean(across_sims) if across_sims else 0
        separation = within_mean - across_mean
        flag = "BAD" if separation <= 0 else ""

        results.append(
            {
                "category": cat,
                "within": within_mean,
                "across": across_mean,
                "separation": separation,
                "n": len(cat_ids),
                "flag": flag,
            }
        )

        print(f"{cat:40s} {within_mean:8.3f} {across_mean:8.3f} {separation:6.3f} {len(cat_ids):4d} {flag:>5s}")

    print_summary(results)
    conn.close()


if __name__ == "__main__":
    _rng = secrets.SystemRandom()
    _rng.seed(42)
    main()
