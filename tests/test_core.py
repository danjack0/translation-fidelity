"""Tests for core scoring engine."""
from __future__ import annotations

import pytest

from translation_fidelity import score
from translation_fidelity.core import (
    _calibrate_score,
    _cosine_similarity,
    _detect_language,
)
import numpy as np


# ---------- Unit tests for helpers ----------

class TestCosineSimilarity:
    def test_identical_vectors_score_one(self):
        v = np.array([1.0, 2.0, 3.0])
        assert _cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_score_zero(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors_score_negative_one(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)


class TestCalibration:
    def test_perfect_similarity_maps_to_100(self):
        assert _calibrate_score(1.0, language_match=True) == 100

    def test_zero_similarity_maps_to_zero(self):
        assert _calibrate_score(0.0, language_match=True) == 0

    def test_negative_similarity_clamped_to_zero(self):
        assert _calibrate_score(-0.5, language_match=True) == 0

    def test_high_similarity_with_lang_match_scores_high(self):
        assert _calibrate_score(0.98, language_match=True) >= 95

    def test_language_mismatch_penalty_applied(self):
        with_match = _calibrate_score(0.98, language_match=True)
        without = _calibrate_score(0.98, language_match=False)
        assert without < with_match
        assert without < 50  # penalty should be substantial

    def test_mid_range_calibration_monotonic(self):
        """Higher similarity should always produce higher score."""
        scores = [_calibrate_score(s, language_match=True) for s in [0.3, 0.5, 0.7, 0.9]]
        assert scores == sorted(scores)


class TestLanguageDetection:
    def test_short_text_returns_none(self):
        """Short text is unreliable; should skip detection."""
        assert _detect_language("Ciao") is None
        assert _detect_language("Hi there") is None

    def test_longer_text_detects_language(self):
        """Longer text should detect correctly."""
        result = _detect_language("Bonjour, comment allez-vous aujourd'hui?")
        assert result == "fr"

    def test_longer_italian_detected(self):
        result = _detect_language("Mi piace molto la pizza italiana napoletana.")
        assert result == "it"


# ---------- Integration tests for score() ----------

class TestScore:
    def test_good_translation_scores_high(self):
        result = score("Hello, how are you?", "Ciao, come stai?", "it")
        assert result.score >= 90
        assert result.language_match is True

    def test_garbage_translation_scores_low(self):
        result = score(
            "Hello, how are you?",
            "Asdf qwerty zxcv banana telefono giallo.",
            "it",
        )
        assert result.score < 30

    def test_antonym_translation_has_low_confidence(self):
        """Subtle meaning errors should be flagged with low confidence."""
        result = score("I love pizza.", "Odio la pizza.", "it")
        assert result.confidence == "low"

    def test_empty_translation_returns_zero(self):
        result = score("Hello", "", "it")
        assert result.score == 0
        assert "Translation is empty" in result.warnings

    def test_wrong_language_penalized(self):
        """French translation when Italian expected."""
        result = score(
            "Hello, how are you doing today?",
            "Bonjour, comment allez-vous aujourd'hui?",
            "it",
        )
        assert result.language_match is False
        assert result.score < 50
        assert any("detected" in w for w in result.warnings)

    def test_result_to_dict_has_expected_keys(self):
        result = score("Hello", "Ciao", "it")
        d = result.to_dict()
        expected_keys = {
            "score", "semantic_similarity", "detected_language",
            "target_language", "language_match", "confidence", "warnings",
        }
        assert set(d.keys()) == expected_keys

    def test_score_is_in_valid_range(self):
        """Score should always be 0-100 regardless of inputs."""
        result = score("Hello world", "Ciao mondo", "it")
        assert 0 <= result.score <= 100