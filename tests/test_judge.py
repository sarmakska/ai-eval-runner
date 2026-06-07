"""LLM-as-judge scorer."""
import pytest

import aieval.scorers.judge as judge_mod
from aieval.core.scorer import invoke_scorer, scorer_name
from aieval.scorers import llm_judge, parse_verdict


def test_parse_verdict_normalises_to_unit_range():
    assert parse_verdict('{"score": 10, "reason": "perfect"}') == 1.0
    assert parse_verdict('{"score": 5, "reason": "ok"}') == 0.5
    assert parse_verdict('here is my verdict {"score": 0, "reason": "wrong"} done') == 0.0


def test_parse_verdict_clamps_and_handles_malformed():
    assert parse_verdict('{"score": 20}') == 1.0
    assert parse_verdict('{"score": -3}') == 0.0
    assert parse_verdict("not json at all") == 0.0
    assert parse_verdict("") == 0.0


def test_llm_judge_uses_grading_model(monkeypatch):
    captured = {}

    async def fake_provider(prompt: str, model: str = "smart") -> str:
        captured["prompt"] = prompt
        captured["model"] = model
        return '{"score": 8, "reason": "good"}'

    monkeypatch.setattr(judge_mod, "get_provider", lambda name: fake_provider)

    j = llm_judge(rubric="grade correctness", model="judge-pro", name="correctness")
    assert scorer_name(j) == "correctness"

    score = invoke_scorer(
        j,
        "the answer is 42",
        "42",
        {"prompt": "what is the answer?", "model": "smart", "provider": "sarmalink"},
    )
    assert score == 0.8
    assert captured["model"] == "judge-pro"
    assert "what is the answer?" in captured["prompt"]
    assert "grade correctness" in captured["prompt"]


def test_llm_judge_self_consistency_takes_median(monkeypatch):
    # Three verdicts: 0.2, 0.4, 1.0 -> median 0.4. A single noisy 1.0 cannot
    # drag the score up once the median of an odd sample is taken.
    replies = iter(
        ['{"score": 2}', '{"score": 4}', '{"score": 10}']
    )
    calls = {"n": 0}

    async def fake_provider(prompt: str, model: str = "smart") -> str:
        calls["n"] += 1
        return next(replies)

    monkeypatch.setattr(judge_mod, "get_provider", lambda name: fake_provider)

    j = llm_judge(rubric="grade", samples=3, name="consistent")
    score = invoke_scorer(j, "x", "y", {"prompt": "q"})
    assert score == 0.4
    assert calls["n"] == 3


def test_llm_judge_rejects_non_positive_samples():
    with pytest.raises(ValueError, match="positive integer"):
        llm_judge(samples=0)
