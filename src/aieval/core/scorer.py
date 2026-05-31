"""Scorer decorator + protocol."""
from collections.abc import Callable


def scorer(fn: Callable[[str, str], float]) -> Callable[[str, str], float]:
    """Mark a function as a scorer. Currently a no-op decorator; reserved for future
    metadata (rubric labels, calibration sets, etc.)."""
    fn.is_scorer = True  # type: ignore[attr-defined]
    return fn
