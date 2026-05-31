"""Eval runner. Parallel execution, retry, partial resumption."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Callable, Iterable

from ..backends import get_backend
from ..providers import get_provider


def run(
    *,
    name: str,
    dataset: Iterable[dict],
    scorers: list[Callable[[str, str], float]],
    provider: str = "sarmalink",
    model: str = "smart",
    concurrency: int = 8,
    retries: int = 1,
) -> str:
    """Run an eval. Returns the run id."""
    backend = get_backend()
    provider_fn = get_provider(provider)

    examples = list(dataset)
    dataset_hash = hashlib.sha256(json.dumps(examples, sort_keys=True).encode()).hexdigest()[:12]
    run_id = backend.create_run(name=name, model=model, dataset_hash=dataset_hash, total=len(examples))

    print(f"Run {run_id}: {name} on {len(examples)} examples with model={model}")

    async def run_one(idx: int, example: dict) -> dict:
        prompt = example["prompt"]
        expected = example.get("expected", "")
        for attempt in range(retries + 1):
            try:
                t0 = time.time()
                prediction = await provider_fn(prompt, model)
                latency_ms = int((time.time() - t0) * 1000)
                scores = {s.__name__: float(s(prediction, expected)) for s in scorers}
                return {
                    "idx": idx,
                    "prompt": prompt,
                    "expected": expected,
                    "prediction": prediction,
                    "latency_ms": latency_ms,
                    "scores": scores,
                    "ok": True,
                }
            except Exception as e:
                if attempt == retries:
                    return {"idx": idx, "ok": False, "error": str(e)}
        return {"idx": idx, "ok": False, "error": "unreachable"}

    async def run_all():
        sem = asyncio.Semaphore(concurrency)

        async def guarded(i: int, ex: dict):
            async with sem:
                return await run_one(i, ex)

        results = await asyncio.gather(*(guarded(i, ex) for i, ex in enumerate(examples)))
        return results

    results = asyncio.run(run_all())
    backend.persist_results(run_id, results)
    backend.close_run(run_id)

    summary = aggregate(results, scorers)
    print(f"Done. Pass rate: {summary['pass_rate']:.1%}, avg latency: {summary['avg_latency_ms']}ms")
    return run_id


def aggregate(results: list[dict], scorers: list) -> dict:
    ok = [r for r in results if r.get("ok")]
    if not ok:
        return {"pass_rate": 0.0, "avg_latency_ms": 0}
    avg = {s.__name__: sum(r["scores"][s.__name__] for r in ok) / len(ok) for s in scorers}
    pass_rate = sum(1 for r in ok if all(v >= 0.5 for v in r["scores"].values())) / len(results)
    avg_latency = sum(r["latency_ms"] for r in ok) // len(ok)
    return {"pass_rate": pass_rate, "avg_latency_ms": avg_latency, "scorers": avg}
