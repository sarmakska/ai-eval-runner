"""End-to-end: run two evals over a fixture dataset, then diff, gate and A/B them."""
from pathlib import Path

from typer.testing import CliRunner

from aieval import dataset, run, scorer
from aieval.cli import app
from aieval.core.pairwise import pairwise
from aieval.core.regression import compare
from aieval.scorers import exact_match

FIXTURE = Path(__file__).parent / "fixtures" / "dataset.jsonl"
cli = CliRunner()


@scorer(name="match")
def _match(prediction: str, expected: str) -> float:
    return exact_match(prediction, expected)


def _run(name: str, predict) -> str:
    """Run an eval with an injected deterministic provider."""
    import aieval.core.runner as runner_mod

    original = runner_mod.get_provider
    runner_mod.get_provider = lambda _name: predict
    try:
        return run(
            name=name,
            dataset=dataset.jsonl(FIXTURE),
            scorers=[_match],
            provider="sarmalink",
            model="stub",
            concurrency=2,
            retries=0,
        )
    finally:
        runner_mod.get_provider = original


def test_full_flow_run_persist_diff_pairwise_ci(backend):
    async def good(prompt: str, model: str = "stub") -> str:
        # Echo the expected answer for the first example only.
        return "the cat sat on the mat" if "cat" in prompt else "wrong"

    async def better(prompt: str, model: str = "stub") -> str:
        # Match two examples, so it scores strictly higher than `good`.
        if "cat" in prompt:
            return "the cat sat on the mat"
        if "json" in prompt:
            return "{}"
        return "still wrong"

    baseline_id = _run("e2e", good)
    candidate_id = _run("e2e", better)

    base_results = backend.get_results(baseline_id)
    cand_results = backend.get_results(candidate_id)
    assert len(base_results) == 3
    assert len(cand_results) == 3

    # The candidate matched more examples, so the diff is an improvement.
    report = compare(base_results, cand_results)
    match_delta = next(d for d in report.scorer_deltas if d.scorer == "match")
    assert match_delta.delta > 0

    # Pairwise should not declare a regression for the candidate.
    pw = pairwise(base_results, cand_results, iterations=500)[0]
    assert pw.winner in ("b", "tie")

    # The CI gate passes when comparing the improving candidate to its baseline.
    result = cli.invoke(app, ["ci", candidate_id, "--baseline", baseline_id, "--threshold", "0.05"])
    assert result.exit_code == 0

    # The CI gate fails when the baseline is the better run (candidate regressed).
    result = cli.invoke(app, ["ci", baseline_id, "--baseline", candidate_id, "--threshold", "0.05"])
    assert result.exit_code == 1


def test_cli_diff_and_pairwise_render(backend):
    async def predict(prompt: str, model: str = "stub") -> str:
        return "the cat sat on the mat" if "cat" in prompt else "x"

    a = _run("render", predict)
    b = _run("render", predict)

    diff_out = cli.invoke(app, ["diff", a, b])
    assert diff_out.exit_code == 0
    assert "match" in diff_out.output

    pw_out = cli.invoke(app, ["pairwise", a, b, "--iterations", "200"])
    assert pw_out.exit_code == 0
    assert "Pairwise" in pw_out.output
