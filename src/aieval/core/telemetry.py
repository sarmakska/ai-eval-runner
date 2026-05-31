"""OpenTelemetry attribute capture for eval runs.

Every run and every example can be wrapped in a span carrying semantic
attributes (model, provider, dataset version, scores, latency). When the
``opentelemetry-api`` package is installed and a tracer provider is configured
the spans flow to your collector. When it is not installed this module degrades
to a no-op so the runner has no hard dependency on a telemetry stack.

Attribute names follow the GenAI semantic conventions where they exist
(``gen_ai.request.model``, ``gen_ai.system``) and use the ``aieval.*`` namespace
for runner-specific signal.
"""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

try:  # pragma: no cover - exercised indirectly depending on environment
    from opentelemetry import trace as _otel_trace

    _HAVE_OTEL = True
except ImportError:  # pragma: no cover
    _otel_trace = None  # type: ignore[assignment]
    _HAVE_OTEL = False


def is_enabled() -> bool:
    """True when the OpenTelemetry API is importable."""
    return _HAVE_OTEL


def _coerce(value: Any) -> Any:
    if isinstance(value, bool | int | float | str):
        return value
    return str(value)


@contextlib.contextmanager
def span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[SpanHandle]:
    """Start a span carrying ``attributes``.

    Yields a :class:`SpanHandle` whose ``set`` method records further
    attributes once they are known (for example scores computed inside the
    span). With OpenTelemetry absent the handle is a no-op recorder, which keeps
    call sites identical in both modes and lets tests assert on captured
    attributes without a collector.
    """
    flat = {k: _coerce(v) for k, v in (attributes or {}).items()}
    if not _HAVE_OTEL:
        handle = SpanHandle(flat, _otel_span=None)
        yield handle
        return

    tracer = _otel_trace.get_tracer("aieval")
    with tracer.start_as_current_span(name) as otel_span:
        for key, value in flat.items():
            otel_span.set_attribute(key, value)
        handle = SpanHandle(flat, _otel_span=otel_span)
        yield handle


class SpanHandle:
    """Records attributes for a span and mirrors them onto the live OTel span.

    The captured attribute dict is always available via :attr:`attributes`,
    regardless of whether OpenTelemetry is installed, so it doubles as the unit
    of record the runner persists and tests inspect.
    """

    def __init__(self, attributes: dict[str, Any], _otel_span: Any) -> None:
        self.attributes = attributes
        self._otel_span = _otel_span

    def set(self, key: str, value: Any) -> None:
        coerced = _coerce(value)
        self.attributes[key] = coerced
        if self._otel_span is not None:
            self._otel_span.set_attribute(key, coerced)
