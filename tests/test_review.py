"""Tests for draft triage review pipeline."""

import json
from unittest.mock import MagicMock, patch

from lessons_db.db import init_db
from lessons_db.review import (
    DraftReview,
    ReviewBatch,
    claude_review_batch,
    execute_verdicts,
    filter_noise,
    jaccard_similarity,
)


class TestJaccardSimilarity:
    def test_identical_strings_score_one(self):
        assert jaccard_similarity("never swallow exceptions silently", "never swallow exceptions silently") == 1.0

    def test_completely_different_strings_score_zero(self):
        assert jaccard_similarity("apple orange", "banana grape") == 0.0

    def test_partial_overlap_between_zero_and_one(self):
        score = jaccard_similarity("never swallow exceptions", "always log exceptions first")
        assert 0.0 < score < 1.0

    def test_empty_strings_score_one(self):
        assert jaccard_similarity("", "") == 1.0

    def test_one_empty_string_scores_zero(self):
        assert jaccard_similarity("something", "") == 0.0


class TestFilterNoise:
    def _draft(self, id_, one_liner, source="auto_transcript"):
        return {"id": id_, "extracted_data": json.dumps({"one_liner": one_liner}), "source": source}

    def test_dismisses_no_mistakes_pattern(self):
        drafts = [self._draft(1, "No coding mistakes were discovered in this session.")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(kept) == 0
        assert len(dismissed) == 1

    def test_dismisses_no_bugs_pattern(self):
        drafts = [self._draft(1, "No bugs were found in the reviewed code.")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(dismissed) == 1

    def test_dismisses_repeated_content_pattern(self):
        drafts = [self._draft(1, "Repeated content in the transcript was found.")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(dismissed) == 1

    def test_dismisses_too_short_one_liner(self):
        drafts = [self._draft(1, "Write tests.")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(dismissed) == 1

    def test_dismisses_empty_one_liner(self):
        drafts = [self._draft(1, "")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(dismissed) == 1

    def test_keeps_good_one_liner(self):
        drafts = [self._draft(1, "Never call close() on sqlite3 connections inside a context manager — use closing().")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(kept) == 1
        assert len(dismissed) == 0

    def test_dismisses_near_duplicate_of_existing_lesson(self):
        existing = ["Never call close() on sqlite3 connections inside a context manager"]
        drafts = [self._draft(1, "Never call close on sqlite3 connections inside context manager")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=existing)
        assert len(dismissed) == 1

    def test_keeps_dissimilar_to_existing(self):
        existing = ["Never call close() on sqlite3 connections inside a context manager"]
        drafts = [self._draft(1, "Always await coroutines — bare async def without await is a logic error")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=existing)
        assert len(kept) == 1

    def test_dismisses_duplicate_within_batch(self):
        drafts = [
            self._draft(1, "Always log exceptions before swallowing them silently"),
            self._draft(2, "Always log exceptions before swallowing them"),
        ]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(kept) == 1
        assert len(dismissed) == 1

    def test_dismiss_reason_recorded(self):
        drafts = [self._draft(1, "No bugs found.")]
        _, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert "_dismiss_reason" in dismissed[0]

    def test_dismisses_no_anti_patterns(self):
        drafts = [self._draft(1, "No anti-patterns were found in the code reviewed.")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(dismissed) == 1

    def test_dismisses_transcript_does_not_include(self):
        drafts = [self._draft(1, "The transcript does not include any coding mistakes.")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(dismissed) == 1

    def test_dismisses_same_questions_presented_twice(self):
        drafts = [self._draft(1, "The same questions were presented twice in the session.")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(dismissed) == 1

    def test_malformed_extracted_data_dismissed(self):
        draft = {"id": 1, "extracted_data": "not-valid-json", "source": "auto_transcript"}
        kept, dismissed = filter_noise([draft], existing_one_liners=[])
        assert len(dismissed) == 1

    def test_empty_existing_one_liners_keeps_good_draft(self):
        drafts = [self._draft(1, "Never use bare except clauses without logging the error first")]
        kept, dismissed = filter_noise(drafts, existing_one_liners=[])
        assert len(kept) == 1
        assert len(dismissed) == 0


class TestClaudeReviewBatch:
    def _draft(self, id_, one_liner):
        return {"id": id_, "extracted_data": json.dumps({"one_liner": one_liner}), "source": "auto_transcript"}

    @staticmethod
    def _mock_parse_response(reviews: list[DraftReview]) -> MagicMock:
        """Build a mock structured-output parse response."""
        mock_choice = MagicMock()
        mock_choice.finish_reason = "stop"
        mock_choice.message.parsed = ReviewBatch(reviews=reviews)
        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        return mock_resp

    def test_returns_promote_verdict_for_specific_lesson(self):
        drafts = [self._draft(42, "Never use bare except: without logging the error first")]
        review = DraftReview(
            id=42,
            verdict="PROMOTE",
            confidence=5,
            reason="Specific, actionable, prevents silent failures",
            improved_one_liner="Never use bare `except:` — always log before swallowing",
            detection_pattern=r"except\s*:",
            semgrep_rule="",
        )

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.beta.chat.completions.parse.return_value = self._mock_parse_response([review])
            verdicts = claude_review_batch(drafts, existing_titles=[], api_key="test-key")

        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "PROMOTE"
        assert verdicts[0]["id"] == 42
        assert verdicts[0]["detection_pattern"] == r"except\s*:"

    def test_confidence_gate_demotes_low_confidence_promote(self):
        """A PROMOTE verdict with confidence < threshold is downgraded to DISMISS."""
        drafts = [self._draft(10, "Always validate inputs carefully before processing")]
        review = DraftReview(
            id=10,
            verdict="PROMOTE",
            confidence=2,  # below _CONFIDENCE_THRESHOLD=4
            reason="Borderline",
            improved_one_liner="Validate inputs",
            detection_pattern="",
            semgrep_rule="",
        )

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.beta.chat.completions.parse.return_value = self._mock_parse_response([review])
            verdicts = claude_review_batch(drafts, existing_titles=[], api_key="test-key")

        assert verdicts[0]["verdict"] == "DISMISS"

    def test_returns_dismiss_verdict_for_vague_lesson(self):
        drafts = [self._draft(99, "Write cleaner code and test more thoroughly")]
        review = DraftReview(
            id=99,
            verdict="DISMISS",
            confidence=1,
            reason="Too vague",
            improved_one_liner="",
            detection_pattern="",
            semgrep_rule="",
        )

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.beta.chat.completions.parse.return_value = self._mock_parse_response([review])
            verdicts = claude_review_batch(drafts, existing_titles=[], api_key="test-key")

        assert verdicts[0]["verdict"] == "DISMISS"

    def test_handles_api_error_falls_back_to_sub_batches(self):
        """On primary batch failure, should retry with sub-batches."""
        drafts = [self._draft(7, "Always log exceptions before swallowing them silently")]
        review = DraftReview(
            id=7,
            verdict="DISMISS",
            confidence=1,
            reason="retry succeeded",
            improved_one_liner="",
            detection_pattern="",
            semgrep_rule="",
        )
        with patch("openai.OpenAI") as MockClient:
            # First call (batch) fails; second call (sub-batch retry) succeeds
            MockClient.return_value.beta.chat.completions.parse.side_effect = [
                Exception("API timeout"),
                self._mock_parse_response([review]),
            ]
            verdicts = claude_review_batch(drafts, existing_titles=[], api_key="test-key")

        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "DISMISS"

    def test_handles_api_error_gracefully_when_retry_also_fails(self):
        """When both batch and sub-batch retry fail, marks verdicts as ERROR."""
        drafts = [self._draft(7, "Always log exceptions before swallowing them silently")]
        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.beta.chat.completions.parse.side_effect = Exception("API timeout")
            verdicts = claude_review_batch(drafts, existing_titles=[], api_key="test-key")

        assert len(verdicts) == 1
        assert verdicts[0]["verdict"] == "ERROR"
        assert "error" in verdicts[0]["reason"].lower()

    def test_processes_multiple_drafts_in_batches(self):
        """Verify batching: 25 drafts with batch_size=20 should trigger 2 API calls."""
        drafts = [self._draft(i, f"Always validate input at boundary {i} before processing") for i in range(25)]
        batch1 = [
            DraftReview(
                id=i,
                verdict="DISMISS",
                confidence=1,
                reason="test",
                improved_one_liner="",
                detection_pattern="",
                semgrep_rule="",
            )
            for i in range(20)
        ]
        batch2 = [
            DraftReview(
                id=i,
                verdict="DISMISS",
                confidence=1,
                reason="test",
                improved_one_liner="",
                detection_pattern="",
                semgrep_rule="",
            )
            for i in range(20, 25)
        ]

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.beta.chat.completions.parse.side_effect = [
                self._mock_parse_response(batch1),
                self._mock_parse_response(batch2),
            ]
            verdicts = claude_review_batch(drafts, existing_titles=[], api_key="test-key")

        assert MockClient.return_value.beta.chat.completions.parse.call_count == 2
        assert len(verdicts) == 25

    def test_existing_titles_included_in_prompt(self):
        """Verify existing lesson titles are passed in the prompt for duplicate detection."""
        drafts = [self._draft(1, "Never swallow exceptions without logging")]
        review = DraftReview(
            id=1,
            verdict="DISMISS",
            confidence=1,
            reason="dup",
            improved_one_liner="",
            detection_pattern="",
            semgrep_rule="",
        )

        with patch("openai.OpenAI") as MockClient:
            MockClient.return_value.beta.chat.completions.parse.return_value = self._mock_parse_response([review])
            claude_review_batch(drafts, existing_titles=["Never swallow exceptions"], api_key="test-key")

        call_kwargs = MockClient.return_value.beta.chat.completions.parse.call_args
        prompt_text = call_kwargs[1]["messages"][0]["content"]
        assert "Never swallow exceptions" in prompt_text


class TestExecuteVerdicts:
    def _insert_draft(self, conn, one_liner, source="auto_transcript"):
        data = {"one_liner": one_liner, "improved_one_liner": one_liner + " (improved)"}
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES ('raw', ?, 'pending', '2026-02-27', ?)",
            [json.dumps(data), source],
        )
        conn.commit()
        return conn.execute("SELECT last_insert_rowid()").fetchone()[0]

    def test_promote_verdict_inserts_lesson(self, db_path, tmp_path):
        conn = init_db(db_path)
        draft_id = self._insert_draft(conn, "Never swallow exceptions silently")
        verdicts = [
            {
                "id": draft_id,
                "verdict": "PROMOTE",
                "reason": "Specific and actionable",
                "improved_one_liner": "Never swallow exceptions — log first",
                "detection_pattern": r"except\s*:",
                "semgrep_rule": "",
            }
        ]

        result = execute_verdicts(conn, verdicts, log_dir=tmp_path)

        assert result["promoted"] == 1
        assert result["dismissed"] == 0
        lesson = conn.execute("SELECT * FROM lessons WHERE one_liner LIKE '%swallow%'").fetchone()
        assert lesson is not None
        assert lesson["polarity"] == "negative"

    def test_promote_verdict_inserts_detection_pattern(self, db_path, tmp_path):
        conn = init_db(db_path)
        draft_id = self._insert_draft(conn, "Never swallow exceptions silently")
        verdicts = [
            {
                "id": draft_id,
                "verdict": "PROMOTE",
                "reason": "Good",
                "improved_one_liner": "Never swallow exceptions",
                "detection_pattern": r"except\s*:",
                "semgrep_rule": "",
            }
        ]

        execute_verdicts(conn, verdicts, log_dir=tmp_path)

        pattern = conn.execute("SELECT * FROM detection_patterns").fetchone()
        assert pattern is not None
        assert pattern["regex"] == r"except\s*:"

    def test_dismiss_verdict_marks_draft_dismissed(self, db_path, tmp_path):
        conn = init_db(db_path)
        draft_id = self._insert_draft(conn, "Write better code generally")
        verdicts = [
            {
                "id": draft_id,
                "verdict": "DISMISS",
                "reason": "Too vague",
                "improved_one_liner": "",
                "detection_pattern": "",
                "semgrep_rule": "",
            }
        ]

        result = execute_verdicts(conn, verdicts, log_dir=tmp_path)

        assert result["dismissed"] == 1
        row = conn.execute("SELECT status FROM capture_drafts WHERE id = ?", [draft_id]).fetchone()
        assert row["status"] == "dismissed"

    def test_writes_triage_jsonl_log(self, db_path, tmp_path):
        conn = init_db(db_path)
        draft_id = self._insert_draft(conn, "Write better code generally")
        verdicts = [
            {
                "id": draft_id,
                "verdict": "DISMISS",
                "reason": "Too vague",
                "improved_one_liner": "",
                "detection_pattern": "",
                "semgrep_rule": "",
            }
        ]

        execute_verdicts(conn, verdicts, log_dir=tmp_path)

        log_files = list(tmp_path.glob("triage-*.jsonl"))
        assert len(log_files) == 1
        lines = log_files[0].read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["verdict"] == "DISMISS"
        assert entry["draft_id"] == draft_id

    def test_promote_with_missing_draft_returns_error_in_summary(self, db_path, tmp_path):
        conn = init_db(db_path)
        # Use a draft_id that doesn't exist
        verdicts = [
            {
                "id": 9999,
                "verdict": "PROMOTE",
                "reason": "Good",
                "improved_one_liner": "Some lesson",
                "detection_pattern": "",
                "semgrep_rule": "",
            }
        ]

        result = execute_verdicts(conn, verdicts, log_dir=tmp_path)

        assert result["promoted"] == 0
        assert result["errors"] == 1

    def test_promote_failed_sets_draft_status(self, db_path, tmp_path):
        """A draft that exists but fails promote_draft (status != 'pending') is set to promote_failed."""
        conn = init_db(db_path)
        # Insert a draft with status='approved' — promote_draft will find the row but
        # its WHERE status='pending' guard will return None, triggering the PROMOTE_FAILED path.
        data = json.dumps({"one_liner": "Always validate inputs at boundaries"})
        conn.execute(
            "INSERT INTO capture_drafts (raw_content, extracted_data, status, created_date, source) "
            "VALUES ('raw', ?, 'approved', '2026-02-27', 'auto_transcript')",
            [data],
        )
        conn.commit()
        draft_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        verdicts = [
            {
                "id": draft_id,
                "verdict": "PROMOTE",
                "reason": "Good",
                "improved_one_liner": "Always validate inputs at boundaries",
                "detection_pattern": "",
                "semgrep_rule": "",
            }
        ]
        result = execute_verdicts(conn, verdicts, log_dir=tmp_path)

        assert result["errors"] == 1
        assert result["promoted"] == 0
        row = conn.execute("SELECT status FROM capture_drafts WHERE id=?", [draft_id]).fetchone()
        assert row["status"] == "promote_failed"
