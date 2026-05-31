"""Smoke tests. Prove the main entry points boot and a run completes end to end."""
import os

from typer.testing import CliRunner

import aieval
from aieval import dataset, run, scorer
from aieval.cli import app

runner = CliRunner()


def test_package_exports():
    assert aieval.__version__
    assert callable(run)
    assert callable(scorer)
    assert hasattr(dataset, "jsonl")


def test_cli_boots():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "CLI" in result.output
    for command in ("run", "list", "view", "diff", "ci"):
        assert command in result.output


def test_run_completes_end_to_end(tmp_path):
    os.environ["AIEVAL_BACKEND"] = "sqlite"
    os.environ["AIEVAL_DB_PATH"] = str(tmp_path / "smoke.db")

    # Force a fresh backend singleton so the temp database path is honoured.
    import aieval.backends as backends

    backends._singleton = None

    @scorer
    def always_pass(prediction: str, _expected: str) -> float:
        return 1.0

    run_id = run(
        name="smoke",
        dataset=dataset.from_list([{"prompt": "hello", "expected": "hi"}]),
        scorers=[always_pass],
        provider="sarmalink",
        model="smart",
        concurrency=2,
        retries=0,
    )

    assert run_id
    results = backends.get_backend().get_results(run_id)
    assert len(results) == 1
    assert results[0]["ok"] == 1
