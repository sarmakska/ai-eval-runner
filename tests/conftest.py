"""Shared fixtures: an isolated backend per test and a stub provider."""

import pytest


@pytest.fixture
def backend(tmp_path, monkeypatch):
    """Return a fresh SQLite backend bound to a temp database."""
    monkeypatch.setenv("AIEVAL_BACKEND", "sqlite")
    monkeypatch.setenv("AIEVAL_DB_PATH", str(tmp_path / "test.db"))
    import aieval.backends as backends

    backends._singleton = None
    yield backends.get_backend()
    backends._singleton = None


@pytest.fixture
def stub_sarmalink(monkeypatch):
    """Replace the SarmaLink provider with a deterministic echo for the prompt."""
    async def _stub(prompt: str, model: str = "smart") -> str:
        return f"echo: {prompt}"

    import aieval.providers as providers

    monkeypatch.setattr(providers, "sarmalink_completion", _stub)
    return _stub
