"""Eval runner. Parallel execution, retry, scoring, storage and telemetry."""
from __future__ import annotations

import asyncio
import subprocess
import time
from collections.abc import Callable, Iterable

from ..backends import get_backend
from ..providers import get_provider
from . import dataset as dataset_mod
from . import telemetry
from .scorer import invoke_scorer, scorer_name


def git_sha() -> str:
    """Return the short git SHA of the working tree, or ``"unknown"``."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def run(
    *,
    name: str,
    dataset: Iterable[dict],
    scorers: list[Callable[..., float]],
    provider: str = "sarmalink",
    model: str = "smart",
    concurrency: int = 8,
    retries: int = 1,
) -> str:
    """Run an eval. Returns the run id.

    Each example is sent to the provider concurrently (bounded by
    ``concurrency``), scored by every scorer, and persisted. The run is tagged
    with the git SHA and the dataset content version so later comparisons only
    pair genuinely comparable runs. When OpenTelemetry is configured each run
    and example emits a span with semantic attributes.
    """
    backend = get_backend()
    provider_fn = get_provider(provider)

    examples = list(dataset)
    dataset_version = dataset_mod.version(examples)
    sha = git_sha()
    run_id = backend.create_run(
        name=name,
        model=model,
        dataset_hash=dataset_version,
        total=len(examples),
        git_sha=sha,
        provider=provider,
    )

    print(f"Run {run_id}: {name} on {len(examples)} examples with model={model} (sha={sha}, dataset={dataset_version})")

    async def run_one(idx: int, example: dict) -> dict:
        prompt = example["prompt"]
        expected = example.get("expected", "")
        context = {"example": example, "model": model, "provider": provider, "prompt": prompt}
        with telemetry.span(
            "aieval.example",
            {
                "aieval.example.index": idx,
                "gen_ai.request.model": model,
                "gen_ai.system": provider,
            },
        ) as sp:
            for attempt in range(retries + 1):
                try:
                    t0 = time.time()
                    prediction = await provider_fn(prompt, model)
                    latency_ms = int((time.time() - t0) * 1000)
                    scores = {
                        scorer_name(s): invoke_scorer(s, prediction, expected, context) for s in scorers
                    }
                    sp.set("aieval.example.latency_ms", latency_ms)
                    for metric, value in scores.items():
                        sp.set(f"aieval.score.{metric}", value)
                    return {
                        "idx": idx,
                        "prompt": prompt,
                        "expected": expected,
                        "prediction": prediction,
                        "latency_ms": latency_ms,
                        "scores": scores,
                        "ok": True,
                    }
                except Exception as e:  # noqa: BLE001 - surfaced as a failed example
                    if attempt == retries:
                        sp.set("aieval.example.error", str(e))
                        return {"idx": idx, "ok": False, "error": str(e)}
            return {"idx": idx, "ok": False, "error": "unreachable"}

    async def run_all():
        sem = asyncio.Semaphore(concurrency)

        async def guarded(i: int, ex: dict):
            async with sem:
                return await run_one(i, ex)

        return await asyncio.gather(*(guarded(i, ex) for i, ex in enumerate(examples)))

    with telemetry.span(
        "aieval.run",
        {
            "aieval.run.name": name,
            "aieval.run.dataset_version": dataset_version,
            "aieval.run.git_sha": sha,
            "gen_ai.request.model": model,
            "gen_ai.system": provider,
        },
    ) as run_span:
        results = asyncio.run(run_all())
        backend.persist_results(run_id, results)
        backend.close_run(run_id)
        summary = aggregate(results, scorers)
        run_span.set("aieval.run.pass_rate", summary["pass_rate"])
        run_span.set("aieval.run.avg_latency_ms", summary["avg_latency_ms"])

    print(f"Done. Pass rate: {summary['pass_rate']:.1%}, avg latency: {summary['avg_latency_ms']}ms")
    return run_id


def aggregate(results: list[dict], scorers: list) -> dict:
    """Summarise results: pass rate, average latency and per-scorer means."""
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return {"pass_rate": 0.0, "avg_latency_ms": 0, "scorers": {}}
    avg = {scorer_name(s): sum(r["scores"][scorer_name(s)] for r in ok) / len(ok) for s in scorers}
    pass_rate = sum(1 for r in ok if all(v >= 0.5 for v in r["scores"].values())) / len(results)
    avg_latency = sum(r["latency_ms"] for r in ok) // len(ok)
    return {"pass_rate": pass_rate, "avg_latency_ms": avg_latency, "scorers": avg}
