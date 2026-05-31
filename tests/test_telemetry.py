"""OpenTelemetry attribute capture. Works with or without the API installed."""
from aieval.core import telemetry


def test_span_captures_attributes():
    with telemetry.span("aieval.test", {"a": 1, "b": "x"}) as sp:
        sp.set("c", 3.5)
        assert sp.attributes["a"] == 1
        assert sp.attributes["b"] == "x"
        assert sp.attributes["c"] == 3.5


def test_span_coerces_non_primitive_attributes():
    with telemetry.span("aieval.test", {"obj": {"nested": True}}) as sp:
        assert isinstance(sp.attributes["obj"], str)


def test_is_enabled_returns_bool():
    assert isinstance(telemetry.is_enabled(), bool)
