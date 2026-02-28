"""Tests for draft triage review pipeline."""

import json

from lessons_db.review import filter_noise, jaccard_similarity


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
