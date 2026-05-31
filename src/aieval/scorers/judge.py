"""LLM-as-judge scorers.

A judge scorer asks a model to grade a prediction against a rubric and returns
a normalised score in the range 0.0 to 1.0. The model is prompted to answer
with a single JSON object ``{"score": <0-10>, "reason": "..."}`` which is parsed
strictly; a malformed verdict scores 0.0 rather than crashing the run.

``llm_judge`` builds a scorer bound to a rubric and a grading provider. The
resulting callable declares ``example``, ``model`` and ``provider`` keyword
parameters, which the runner fills in via ``invoke_scorer``, so the judge sees
the original prompt as well as the prediction and the expected answer.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from ..core.scorer import scorer
from ..providers import get_provider

DEFAULT_RUBRIC = (
    "Grade how well the response answers the request. Reward correctness, "
    "completeness and relevance. Penalise hallucination and missing detail."
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def _build_prompt(rubric: str, prompt: str, prediction: str, expected: str) -> str:
    expected_block = f"\nReference answer:\n{expected}\n" if expected else "\n"
    return (
        "You are a strict evaluation judge. Grade the response on a scale of 0 to 10.\n"
        f"Rubric: {rubric}\n\n"
        f"Request:\n{prompt}\n"
        f"{expected_block}"
        f"Response to grade:\n{prediction}\n\n"
        'Reply with one JSON object only, no prose: {"score": <integer 0-10>, "reason": "<one sentence>"}'
    )


def parse_verdict(text: str) -> float:
    """Parse a judge reply into a 0.0-1.0 score. Malformed replies score 0.0."""
    match = _JSON_OBJECT.search(text or "")
    if not match:
        return 0.0
    try:
        verdict = json.loads(match.group(0))
        raw = float(verdict["score"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, raw / 10.0))


def llm_judge(
    *,
    rubric: str = DEFAULT_RUBRIC,
    provider: str = "sarmalink",
    model: str = "smart",
    name: str = "llm_judge",
) -> Any:
    """Build a judge scorer bound to ``rubric`` and a grading model.

    The grading provider and model are fixed at build time and are independent
    of the model under evaluation, so you can grade a cheap model's output with
    a stronger judge.
    """
    judge_fn = get_provider(provider)

    @scorer(name=name)
    def _judge(prediction: str, expected: str, *, prompt: str | None = None, **_: Any) -> float:
        grading_prompt = _build_prompt(rubric, prompt or "", prediction, expected)
        reply = asyncio.run(judge_fn(grading_prompt, model))
        return parse_verdict(reply)

    return _judge
