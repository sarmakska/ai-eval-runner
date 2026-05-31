"""Scorer decorator, protocol and invocation helper.

A scorer is any callable that returns a float in the range 0.0 to 1.0. The
simplest form takes ``(prediction, expected)``. Scorers that need more context
(the full example, the model, the provider) can declare extra keyword-only
parameters and the runner will supply them. This keeps the common case a two
argument function while letting judge scorers reach the richer signal they need.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

Scorer = Callable[..., float]


def scorer(fn: Scorer | None = None, *, name: str | None = None) -> Scorer | Callable[[Scorer], Scorer]:
    """Mark a function as a scorer.

    Usage::

        @scorer
        def my_metric(prediction, expected):
            ...

        @scorer(name="length_ok")
        def _length(prediction, _expected):
            ...

    The optional ``name`` overrides the metric label stored in results. Without
    it the function's ``__name__`` is used.
    """

    def wrap(f: Scorer) -> Scorer:
        f.is_scorer = True  # type: ignore[attr-defined]
        if name is not None:
            f.scorer_name = name  # type: ignore[attr-defined]
        return f

    if fn is not None:
        return wrap(fn)
    return wrap


def scorer_name(fn: Scorer) -> str:
    """Return the metric label for a scorer."""
    return getattr(fn, "scorer_name", fn.__name__)


def invoke_scorer(fn: Scorer, prediction: str, expected: str, context: dict[str, Any]) -> float:
    """Call a scorer, supplying only the keyword arguments it actually declares.

    The two positional arguments ``prediction`` and ``expected`` are always
    passed. Any of ``example``, ``model``, ``provider`` and ``prompt`` declared
    by the scorer are filled from ``context``.
    """
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return float(fn(prediction, expected))

    kwargs: dict[str, Any] = {}
    params = sig.parameters
    for key in ("example", "model", "provider", "prompt"):
        if key in params:
            kwargs[key] = context.get(key)
    return float(fn(prediction, expected, **kwargs))
