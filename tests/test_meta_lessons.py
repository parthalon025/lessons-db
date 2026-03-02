"""Tests for loop_level column, meta command group, cluster detection, and meta-lesson generation."""

import json
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from lessons_db.cli import find_meta_lesson_clusters, main
from lessons_db.db import get_lesson, init_db, insert_lesson

# ---------------------------------------------------------------------------
# Schema / column tests
# ---------------------------------------------------------------------------


class TestLoopLevelColumn:
    """Verify the loop_level column exists and behaves correctly."""

    def test_loop_level_column_exists(self, db_path):
        conn = init_db(db_path)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
        assert "loop_level" in cols
        conn.close()

    def test_loop_level_defaults_to_single(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "test lesson",
                "one_liner": "a test",
            },
        )
        lesson = get_lesson(conn, lid)
        assert lesson["loop_level"] == "single"
        conn.close()

    def test_loop_level_can_be_set_to_double(self, db_path):
        conn = init_db(db_path)
        lid = insert_lesson(
            conn,
            {
                "title": "meta-lesson",
                "one_liner": "a meta test",
                "loop_level": "double",
            },
        )
        lesson = get_lesson(conn, lid)
        assert lesson["loop_level"] == "double"
        conn.close()

    def test_loop_level_migration_idempotent(self, db_path):
        """_add_extension_columns must not raise on a second call for loop_level."""
        from lessons_db.db import _add_extension_columns

        conn = init_db(db_path)
        # init_db already called _add_extension_columns once; second call must be no-op
        _add_extension_columns(conn)
        cols = {row[1] for row in conn.execute("PRAGMA table_info(lessons)")}
        assert "loop_level" in cols
        conn.close()


# ---------------------------------------------------------------------------
# CLI --help tests
# ---------------------------------------------------------------------------


class TestMetaCliHelp:
    """Verify meta group and generate-meta-lessons command --help."""

    def test_meta_help(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(main, ["--db", str(tmp_path / "test.db"), "meta", "--help"])
        assert result.exit_code == 0
        assert "meta" in result.output.lower()

    def test_generate_meta_lessons_help(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(tmp_path / "test.db"), "meta", "generate-meta-lessons", "--help"],
        )
        assert result.exit_code == 0
        assert "--min-cluster-size" in result.output
        assert "--dry-run" in result.output
        assert "--model" in result.output

    def test_generate_meta_lessons_help_exit_code(self):
        """Acceptance criterion: help exits 0."""
        runner = CliRunner()
        result = runner.invoke(main, ["meta", "generate-meta-lessons", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Cluster detection tests
# ---------------------------------------------------------------------------


class TestFindMetaLessonClusters:
    """Test the cluster detection logic."""

    def test_finds_clusters_above_threshold(self, db_path):
        conn = init_db(db_path)
        # Create 3 lessons with cluster_seed 'async-lifecycle'
        for i in range(3):
            insert_lesson(
                conn,
                {
                    "title": f"async lesson {i}",
                    "one_liner": f"async issue {i}",
                    "cluster_seed": "async-lifecycle",
                },
            )
        # Create 2 lessons with cluster_seed 'config-drift' (below threshold of 3)
        for i in range(2):
            insert_lesson(
                conn,
                {
                    "title": f"config lesson {i}",
                    "one_liner": f"config issue {i}",
                    "cluster_seed": "config-drift",
                },
            )

        clusters = find_meta_lesson_clusters(conn, min_cluster_size=3)
        assert "async-lifecycle" in clusters
        assert len(clusters["async-lifecycle"]) == 3
        assert "config-drift" not in clusters
        conn.close()

    def test_excludes_null_cluster_seed(self, db_path):
        conn = init_db(db_path)
        # Lessons without cluster_seed should not form clusters
        for i in range(5):
            insert_lesson(
                conn,
                {
                    "title": f"untagged {i}",
                    "one_liner": f"untagged {i}",
                },
            )
        clusters = find_meta_lesson_clusters(conn, min_cluster_size=3)
        assert len(clusters) == 0
        conn.close()

    def test_excludes_existing_double_loop(self, db_path):
        conn = init_db(db_path)
        # Create 3 single-loop + 1 double-loop in same cluster
        for i in range(3):
            insert_lesson(
                conn,
                {
                    "title": f"lesson {i}",
                    "one_liner": f"issue {i}",
                    "cluster_seed": "test-cluster",
                },
            )
        insert_lesson(
            conn,
            {
                "title": "existing meta",
                "one_liner": "already generated",
                "cluster_seed": "test-cluster",
                "loop_level": "double",
            },
        )

        clusters = find_meta_lesson_clusters(conn, min_cluster_size=3)
        # The double-loop lesson should NOT be counted
        assert "test-cluster" in clusters
        assert len(clusters["test-cluster"]) == 3
        # Verify none of the returned lessons are double-loop
        for lesson in clusters["test-cluster"]:
            assert lesson["title"] != "existing meta"
        conn.close()

    def test_empty_db_returns_empty(self, db_path):
        conn = init_db(db_path)
        clusters = find_meta_lesson_clusters(conn, min_cluster_size=3)
        assert clusters == {}
        conn.close()

    def test_custom_min_cluster_size(self, db_path):
        conn = init_db(db_path)
        for i in range(5):
            insert_lesson(
                conn,
                {
                    "title": f"big cluster {i}",
                    "one_liner": f"issue {i}",
                    "cluster_seed": "big",
                },
            )
        # With min_cluster_size=5, should match
        clusters = find_meta_lesson_clusters(conn, min_cluster_size=5)
        assert "big" in clusters
        # With min_cluster_size=6, should not match
        clusters = find_meta_lesson_clusters(conn, min_cluster_size=6)
        assert "big" not in clusters
        conn.close()


# ---------------------------------------------------------------------------
# Meta-lesson generation with mocked Ollama
# ---------------------------------------------------------------------------


class TestGenerateMetaLessons:
    """Test generate-meta-lessons command with mocked Ollama."""

    def _seed_cluster(self, conn, seed="async-lifecycle", count=3):
        """Insert a cluster of lessons for testing."""
        ids = []
        for i in range(count):
            lid = insert_lesson(
                conn,
                {
                    "title": f"async issue {i}",
                    "one_liner": f"Missing await on coroutine variant {i}",
                    "cluster_seed": seed,
                },
            )
            ids.append(lid)
        return ids

    def test_dry_run_no_db_writes(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        self._seed_cluster(conn)
        conn.close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "meta", "generate-meta-lessons", "--dry-run"],
        )
        assert result.exit_code == 0
        assert "would generate" in result.output.lower() or "Would generate" in result.output

        # Verify no double-loop lessons were created
        conn = init_db(db_path)
        row = conn.execute("SELECT COUNT(*) as cnt FROM lessons WHERE loop_level = 'double'").fetchone()
        assert row["cnt"] == 0
        conn.close()

    def test_generates_meta_lesson_with_mocked_ollama(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        lesson_ids = self._seed_cluster(conn)
        conn.close()

        mock_response = json.dumps(
            {
                "title": "Async lifecycle governance gap",
                "one_liner": "No systematic resource-cleanup protocol causes recurring async errors",
                "description": "Async lifecycle errors recur because there is no enforced protocol "
                "for resource cleanup. The governing variable is the absence of a "
                "systematic teardown discipline.",
            }
        )

        # Mock urllib.request.urlopen to return our fake Ollama response
        mock_resp_obj = MagicMock()
        mock_resp_obj.read.return_value = json.dumps(
            {
                "response": mock_response,
            }
        ).encode("utf-8")
        mock_resp_obj.__enter__ = MagicMock(return_value=mock_resp_obj)
        mock_resp_obj.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp_obj):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["--db", str(db_path), "meta", "generate-meta-lessons"],
            )

        assert result.exit_code == 0
        assert "Generated: 1" in result.output

        # Verify the meta-lesson was stored correctly
        conn = init_db(db_path)
        meta_rows = conn.execute("SELECT * FROM lessons WHERE loop_level = 'double'").fetchall()
        assert len(meta_rows) == 1
        meta = dict(meta_rows[0])
        assert meta["title"] == "Async lifecycle governance gap"
        assert "resource-cleanup" in meta["one_liner"]
        assert meta["cluster_seed"] == "async-lifecycle"
        assert meta["loop_level"] == "double"
        assert meta["parent_lesson_id"] == lesson_ids[0]
        assert meta["source"] == "auto_meta"
        assert meta["entry_type"] == "lesson"
        assert meta["tier"] == "insight"
        conn.close()

    def test_skips_cluster_with_existing_meta_lesson(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        self._seed_cluster(conn)
        # Create an existing double-loop meta-lesson for this cluster
        insert_lesson(
            conn,
            {
                "title": "existing meta",
                "one_liner": "already exists",
                "cluster_seed": "async-lifecycle",
                "loop_level": "double",
            },
        )
        conn.close()

        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "meta", "generate-meta-lessons"],
        )
        assert result.exit_code == 0
        assert "SKIP" in result.output
        assert "Generated: 0" in result.output

    def test_no_clusters_reports_nothing(self, tmp_path):
        db_path = tmp_path / "test.db"
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["--db", str(db_path), "meta", "generate-meta-lessons"],
        )
        assert result.exit_code == 0
        assert "No clusters" in result.output

    def test_min_cluster_size_option(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        self._seed_cluster(conn, count=3)
        conn.close()

        runner = CliRunner()
        # With --min-cluster-size 5, the cluster of 3 should be excluded
        result = runner.invoke(
            main,
            ["--db", str(db_path), "meta", "generate-meta-lessons", "--min-cluster-size", "5"],
        )
        assert result.exit_code == 0
        assert "No clusters" in result.output

    def test_ollama_error_handled_gracefully(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        self._seed_cluster(conn)
        conn.close()

        import urllib.error

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["--db", str(db_path), "meta", "generate-meta-lessons"],
            )

        assert result.exit_code == 0
        assert "ERROR" in result.output
        assert "Errors: 1" in result.output

    def test_ollama_invalid_json_handled(self, tmp_path):
        db_path = tmp_path / "test.db"
        conn = init_db(db_path)
        self._seed_cluster(conn)
        conn.close()

        mock_resp_obj = MagicMock()
        mock_resp_obj.read.return_value = json.dumps(
            {
                "response": "not valid json at all",
            }
        ).encode("utf-8")
        mock_resp_obj.__enter__ = MagicMock(return_value=mock_resp_obj)
        mock_resp_obj.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_resp_obj):
            runner = CliRunner()
            result = runner.invoke(
                main,
                ["--db", str(db_path), "meta", "generate-meta-lessons"],
            )

        assert result.exit_code == 0
        assert "ERROR" in result.output
