"""LanceDB vector search with Ollama embeddings via ollama-queue."""

import logging
from typing import cast

import lancedb
import pyarrow as pa
import requests

from lessons_db.config import EMBED_DIMS, EMBED_MODEL, OLLAMA_EMBED_URL

logger = logging.getLogger(__name__)

TABLE_NAME = "lessons"

SCHEMA = pa.schema(
    [
        pa.field("lesson_id", pa.int64()),
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), EMBED_DIMS)),
        pa.field("cluster", pa.string()),
        pa.field("tier", pa.string()),
        pa.field("scope", pa.string()),
        pa.field("enforcement", pa.string()),
        pa.field("recurrence_count", pa.int64()),
    ]
)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equal-length vectors. Returns 0.0 on zero magnitude."""
    import math

    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


def get_embedding(text: str) -> list[float] | None:
    """Get embedding vector from Ollama via ollama-queue.

    POST to ollama-queue embed endpoint. Returns 768-dim float list
    or None on any failure.
    """
    try:
        resp = requests.post(
            f"{OLLAMA_EMBED_URL}/api/embed",
            json={"model": EMBED_MODEL, "input": text},
            timeout=300,
        )
        resp.raise_for_status()
        return cast(list[float], resp.json()["embeddings"][0])
    except Exception as exc:
        logger.warning(
            "Embedding request failed (%s: %s) for text: %s",
            type(exc).__name__,
            exc,
            text[:80],
        )
        return None


def init_lance(lance_dir: str) -> lancedb.DBConnection:
    """Connect to LanceDB at lance_dir, creating directory if needed."""
    return lancedb.connect(lance_dir)


def upsert_lesson(db: lancedb.DBConnection, data: dict) -> bool:
    """Upsert a lesson record into LanceDB.

    Gets embedding for data["text"], then inserts or replaces the record
    matching data["lesson_id"]. Returns True on success, False if
    embedding fails.
    """
    vector = get_embedding(data["text"])
    if vector is None:
        return False

    record = {
        "lesson_id": data["lesson_id"],
        "text": data["text"],
        "vector": vector,
        "cluster": data.get("cluster", ""),
        "tier": data.get("tier", ""),
        "scope": data.get("scope", ""),
        "enforcement": data.get("enforcement", ""),
        "recurrence_count": data.get("recurrence_count", 0),
    }

    existing = db.list_tables().tables
    if TABLE_NAME in existing:
        table = db.open_table(TABLE_NAME)
        table.delete(f"lesson_id = {int(data['lesson_id'])}")
        table.add([record])
    else:
        db.create_table(TABLE_NAME, [record], schema=SCHEMA)

    return True


def semantic_search(db: lancedb.DBConnection, query: str, top_k: int = 5) -> list[dict]:
    """Search lessons by semantic similarity.

    Returns list of dicts with lesson_id, text, cluster, score.
    Returns empty list if table doesn't exist or embedding fails.
    """
    if TABLE_NAME not in db.list_tables().tables:
        return []

    vector = get_embedding(query)
    if vector is None:
        return []

    table = db.open_table(TABLE_NAME)
    arrow_table = table.search(vector).limit(top_k).to_arrow()

    return [
        {
            "lesson_id": int(arrow_table.column("lesson_id")[i].as_py()),
            "text": str(arrow_table.column("text")[i].as_py()),
            "cluster": str(arrow_table.column("cluster")[i].as_py()),
            "score": float(arrow_table.column("_distance")[i].as_py()),
        }
        for i in range(arrow_table.num_rows)
    ]
