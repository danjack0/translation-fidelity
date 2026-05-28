"""Tests for core scoring engine."""

from __future__ import annotations

import numpy as np
import pytest

from translation_fidelity import score
from translation_fidelity.core import (
    _calibrate_score,
    _cosine_similarity,
    _detect_language,
)

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
    def test_perfect_similarity_romance_scores_high(self):
        # Romance curve: 164.53 * 1.0 - 84.80 = ~80, clamped path
        score_val = _calibrate_score(1.0, language_match=True, family="romance")
        assert score_val >= 75

    def test_zero_similarity_maps_to_zero(self):
        # Negative predicted BLEU clamps to 0
        assert _calibrate_score(0.0, language_match=True, family="romance") == 0

    def test_negative_similarity_clamped_to_zero(self):
        assert _calibrate_score(-0.5, language_match=True, family="romance") == 0

    def test_high_similarity_with_lang_match_scores_well(self):
        # ~0.98 similarity in romance should land in the high range
        assert _calibrate_score(0.98, language_match=True, family="romance") >= 70

    def test_language_mismatch_penalty_applied(self):
        with_match = _calibrate_score(0.98, language_match=True, family="romance")
        without = _calibrate_score(0.98, language_match=False, family="romance")
        assert without < with_match

    def test_mid_range_calibration_monotonic(self):
        """Higher similarity should always produce higher (or equal) score."""
        scores = [
            _calibrate_score(s, language_match=True, family="romance") for s in [0.3, 0.5, 0.7, 0.9]
        ]
        assert scores == sorted(scores)

    def test_unknown_family_uses_global_curve(self):
        """Unknown family should still produce a valid score via global fallback."""
        score_val = _calibrate_score(0.9, language_match=True, family=None)
        assert 0 <= score_val <= 100

    def test_families_differ(self):
        """Different families should calibrate the same similarity differently."""
        sim = 0.9
        east = _calibrate_score(sim, language_match=True, family="east_asian")
        germanic = _calibrate_score(sim, language_match=True, family="germanic")
        assert east != germanic  # curves genuinely differ


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
    # --- Relative property tests (robust to recalibration) ---

    def test_good_beats_garbage(self):
        """A correct translation must score well above a nonsense one."""
        good = score("I love pizza.", "Amo la pizza.", "it").score
        garbage = score("I love pizza.", "Il treno parte alle otto.", "it").score
        assert good > garbage + 30

    def test_good_beats_wrong_language(self):
        """Correct-language translation should beat a wrong-language one."""
        correct = score(
            "Hello, how are you doing today?",
            "Ciao, come stai oggi amico mio?",
            "it",
        ).score
        wrong_lang = score(
            "Hello, how are you doing today?",
            "Bonjour, comment allez-vous aujourd'hui?",
            "it",
        ).score
        assert correct > wrong_lang

    # --- Absolute regression guards (ranges + documented observed values) ---

    def test_good_translation_in_expected_range(self):
        # Romance good translation observed at ~77 (2026-05). Wide bounds guard
        # against gross regression without being brittle to recalibration.
        result = score("Hello, how are you?", "Ciao, come stai?", "it")
        assert 60 <= result.score <= 90
        assert result.language_match is True

    def test_garbage_translation_scores_low(self):
        result = score(
            "Hello, how are you?",
            "Asdf qwerty zxcv banana telefono giallo.",
            "it",
        )
        assert result.score < 30

    # --- Behavior / contract tests (not about score magnitude) ---

    def test_antonym_translation_has_low_confidence(self):
        """Subtle meaning errors should be flagged with low confidence."""
        result = score("I love pizza.", "Odio la pizza.", "it")
        assert result.confidence == "low"

    def test_empty_translation_returns_zero(self):
        result = score("Hello", "", "it")
        assert result.score == 0
        assert "Translation is empty" in result.warnings

    def test_wrong_language_flagged(self):
        """French translation when Italian expected should be detected."""
        result = score(
            "Hello, how are you doing today?",
            "Bonjour, comment allez-vous aujourd'hui?",
            "it",
        )
        assert result.language_match is False
        assert any("detected" in w for w in result.warnings)

    def test_unknown_language_uses_global_curve(self):
        """A language with no calibration data still returns a valid score."""
        result = score("Hello", "Sawubona unjani namhlanje", "zu")  # Zulu, not in benchmark
        assert 0 <= result.score <= 100
        assert any("global curve" in w for w in result.warnings)

    def test_result_to_dict_has_expected_keys(self):
        result = score("Hello", "Ciao", "it")
        expected_keys = {
            "score",
            "semantic_similarity",
            "detected_language",
            "target_language",
            "language_match",
            "confidence",
            "warnings",
        }
        assert set(result.to_dict().keys()) == expected_keys

    def test_score_always_in_valid_range(self):
        """Score must be 0-100 for any input."""
        result = score("Hello world", "Ciao mondo", "it")
        assert 0 <= result.score <= 100
