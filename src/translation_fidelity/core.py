"""Core scoring engine: reference-free translation quality estimation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from lingua import LanguageDetectorBuilder
from sentence_transformers import SentenceTransformer

from translation_fidelity.calibration_data import (
    FAMILY_CALIBRATION,
    get_family,
)

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_LANGUAGE_DETECTOR = LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()


@dataclass
class ScoreResult:
    """Result of scoring a translation."""

    score: int  # 0-100, user-facing
    semantic_similarity: float  # raw cosine, 0.0-1.0
    detected_language: str | None
    target_language: str
    language_match: bool
    confidence: str  # "high", "medium", "low"
    warnings: list[str]

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "semantic_similarity": round(self.semantic_similarity, 4),
            "detected_language": self.detected_language,
            "target_language": self.target_language,
            "language_match": self.language_match,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """Load model once, cache forever. Lazy so import is cheap."""
    return SentenceTransformer(_MODEL_NAME)


def _detect_language(text: str) -> str | None:
    """
    Detect language, return ISO 639-1 code or None if uncertain.

    Returns None for very short text — detection is unreliable below ~4 words
    or ~20 characters regardless of detector.
    """
    if len(text.split()) < 4 and len(text) < 20:
        return None
    detected = _LANGUAGE_DETECTOR.detect_language_of(text)
    if detected is None:
        return None
    return detected.iso_code_639_1.name.lower()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Fallback curve when language family is unknown: average of all family curves.
_GLOBAL_SLOPE = sum(c.slope for c in FAMILY_CALIBRATION.values()) / len(FAMILY_CALIBRATION)
_GLOBAL_INTERCEPT = sum(c.intercept for c in FAMILY_CALIBRATION.values()) / len(FAMILY_CALIBRATION)


def _calibrate_score(
    similarity: float,
    language_match: bool,
    family: str | None,
) -> int:
    """
    Map raw embedding similarity (0.0-1.0) to a user-facing quality score (0-100),
    using per-language-family calibration curves fit from the benchmark.

    The curve predicts expected BLEU from embedding similarity. We treat that
    predicted BLEU as the 0-100 quality score directly.
    """
    if not language_match:
        # Wrong-language translations: heavy soft penalty (see design note in score()).
        similarity *= 0.4

    if family and family in FAMILY_CALIBRATION:
        curve = FAMILY_CALIBRATION[family]
        slope, intercept = curve.slope, curve.intercept
    else:
        slope, intercept = _GLOBAL_SLOPE, _GLOBAL_INTERCEPT

    predicted_bleu = slope * similarity + intercept
    return max(0, min(100, int(round(predicted_bleu))))


def score(source: str, translation: str, target_language: str) -> ScoreResult:
    """
    Score the quality of a translation against its source.

    Args:
        source: Original text
        translation: Translated text to score
        target_language: ISO 639-1 code of expected language (e.g. "it", "es", "fr")

    Returns:
        ScoreResult with 0-100 score and breakdown.
    """
    warnings: list[str] = []

    if not source.strip():
        warnings.append("Source text is empty")
    if not translation.strip():
        warnings.append("Translation is empty")
        return ScoreResult(
            score=0,
            semantic_similarity=0.0,
            detected_language=None,
            target_language=target_language,
            language_match=False,
            confidence="high",
            warnings=warnings,
        )

    detected = _detect_language(translation)
    language_match = detected == target_language.lower() if detected else True

    if detected and not language_match:
        warnings.append(f"Expected {target_language!r}, detected {detected!r}")

    model = _get_model()
    embeddings = model.encode([source, translation])
    similarity = _cosine_similarity(embeddings[0], embeddings[1])

    family = get_family(target_language)
    if family is None:
        warnings.append(f"No calibration data for {target_language!r}; using global curve")

    calibrated = _calibrate_score(similarity, language_match, family)

    # Confidence: high if extremes, medium in ambiguous middle zone
    if similarity > 0.9 or similarity < 0.3:
        confidence = "high"
    elif 0.5 <= similarity <= 0.8:
        confidence = "low"
    else:
        confidence = "medium"

    return ScoreResult(
        score=calibrated,
        semantic_similarity=similarity,
        detected_language=detected,
        target_language=target_language,
        language_match=language_match,
        confidence=confidence,
        warnings=warnings,
    )
