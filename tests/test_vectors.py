"""Tests for LanceDB vector search and Ollama embeddings."""

from unittest.mock import MagicMock, patch

FAKE_VECTOR = [0.1] * 768


class TestEmbedding:
    """Tests for get_embedding function."""

    @patch("lessons_db.vectors.requests.post")
    def test_get_embedding_returns_vector(self, mock_post):
        from lessons_db.vectors import get_embedding

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [FAKE_VECTOR]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = get_embedding("test text")

        assert result is not None
        assert len(result) == 768
        assert result == FAKE_VECTOR
        mock_post.assert_called_once()

    @patch("lessons_db.vectors.requests.post")
    def test_get_embedding_returns_none_on_failure(self, mock_post):
        from lessons_db.vectors import get_embedding

        mock_post.side_effect = Exception("Connection refused")

        result = get_embedding("test text")

        assert result is None

    @patch("lessons_db.vectors.requests.post")
    def test_get_embedding_uses_300s_timeout(self, mock_post):
        """Embedding timeout must be 300s — matches ollama-queue PROXY_WAIT_TIMEOUT."""
        from lessons_db.vectors import get_embedding

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [FAKE_VECTOR]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        get_embedding("test text")

        _, kwargs = mock_post.call_args
        assert kwargs.get("timeout") == 300, f"Expected timeout=300, got {kwargs.get('timeout')}"

    @patch("lessons_db.vectors.requests.post")
    def test_get_embedding_routes_to_queue_port(self, mock_post):
        """Default embed URL must point to ollama-queue (port 7683), not direct Ollama (11434)."""
        from lessons_db.vectors import OLLAMA_EMBED_URL, get_embedding

        mock_response = MagicMock()
        mock_response.json.return_value = {"embeddings": [FAKE_VECTOR]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        get_embedding("test text")

        called_url = mock_post.call_args[0][0]
        assert (
            "7683" in called_url or "7683" in OLLAMA_EMBED_URL
        ), f"Embed URL should route through ollama-queue (7683), got: {called_url}"


class TestLanceDB:
    """Tests for LanceDB init, upsert, and search."""

    def test_init_creates_connection(self, lance_dir):
        from lessons_db.vectors import init_lance

        db = init_lance(str(lance_dir))
        assert db is not None

    @patch("lessons_db.vectors.get_embedding")
    def test_upsert_and_search(self, mock_embed, lance_dir):
        from lessons_db.vectors import init_lance, semantic_search, upsert_lesson

        mock_embed.return_value = FAKE_VECTOR

        db = init_lance(str(lance_dir))
        data = {
            "lesson_id": 1,
            "text": "Always log before returning fallback",
            "cluster": "A",
            "tier": "lesson_learned",
            "scope": "language:python",
            "enforcement": "semgrep_error",
            "recurrence_count": 3,
        }
        result = upsert_lesson(db, data)
        assert result is True

        results = semantic_search(db, "log fallback", top_k=5)
        assert len(results) == 1
        assert results[0]["lesson_id"] == 1
        assert results[0]["text"] == "Always log before returning fallback"
        assert results[0]["cluster"] == "A"
        assert "score" in results[0]

    @patch("lessons_db.vectors.get_embedding")
    def test_search_empty_table_returns_empty(self, mock_embed, lance_dir):
        from lessons_db.vectors import init_lance, semantic_search

        mock_embed.return_value = FAKE_VECTOR

        db = init_lance(str(lance_dir))
        results = semantic_search(db, "anything", top_k=5)
        assert results == []

    @patch("lessons_db.vectors.get_embedding")
    def test_upsert_is_idempotent(self, mock_embed, lance_dir):
        from lessons_db.vectors import init_lance, semantic_search, upsert_lesson

        mock_embed.return_value = FAKE_VECTOR

        db = init_lance(str(lance_dir))
        data = {
            "lesson_id": 42,
            "text": "Original text",
            "cluster": "B",
            "tier": "insight",
            "scope": "domain:testing",
            "enforcement": "documentation",
            "recurrence_count": 1,
        }
        upsert_lesson(db, data)

        data["text"] = "Updated text"
        data["recurrence_count"] = 2
        upsert_lesson(db, data)

        results = semantic_search(db, "text", top_k=10)
        matching = [r for r in results if r["lesson_id"] == 42]
        assert len(matching) == 1
        assert matching[0]["text"] == "Updated text"
