"""Reference-free translation quality estimation."""

from translation_fidelity.core import ScoreResult, score, score_batch

__version__ = "0.1.0"
__all__ = ["score", "score_batch", "ScoreResult"]
