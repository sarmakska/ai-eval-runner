"""Pairwise A/B with bootstrapped confidence intervals."""
from aieval.core.pairwise import pairwise


def _results(scores):
    return [{"idx": i, "ok": True, "scores": {"acc": s}} for i, s in enumerate(scores)]


def test_clear_win_for_b():
    a = _results([0.0] * 30)
    b = _results([1.0] * 30)
    result = pairwise(a, b, iterations=500)[0]
    assert result.mean_diff == 1.0
    assert result.ci_low > 0
    assert result.winner == "b"


def test_clear_win_for_a():
    a = _results([1.0] * 30)
    b = _results([0.0] * 30)
    result = pairwise(a, b, iterations=500)[0]
    assert result.ci_high < 0
    assert result.winner == "a"


def test_tie_when_identical():
    a = _results([0.5] * 20)
    b = _results([0.5] * 20)
    result = pairwise(a, b, iterations=500)[0]
    assert result.mean_diff == 0.0
    assert result.winner == "tie"


def test_is_reproducible_with_seed():
    a = _results([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    b = _results([1.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    r1 = pairwise(a, b, iterations=500, seed=7)[0]
    r2 = pairwise(a, b, iterations=500, seed=7)[0]
    assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)


def test_only_pairs_shared_examples():
    a = [{"idx": 0, "ok": True, "scores": {"acc": 1.0}}, {"idx": 1, "ok": True, "scores": {"acc": 0.0}}]
    b = [{"idx": 0, "ok": True, "scores": {"acc": 1.0}}]
    result = pairwise(a, b, iterations=200)[0]
    assert result.n == 1
