"""Pairwise A/B comparison with bootstrapped confidence intervals.

When you have two runs over the same examples (run A and run B), this module
answers "is B genuinely better than A, and by how much?" with a paired
bootstrap rather than a single point estimate. It resamples example indices
with replacement many times, recomputes the per-scorer mean difference on each
resample, and reports the observed difference plus a percentile confidence
interval. A win is declared only when the whole interval sits on one side of
zero, which guards against calling a difference that the data cannot support.

The bootstrap is implemented in pure Python with a seedable ``random.Random``
so results are reproducible and the package carries no numeric dependency.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass


def _scores(result: dict) -> dict[str, float]:
    raw = result.get("scores")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = {}
    if not isinstance(raw, dict):
        return {}
    return {k: float(v) for k, v in raw.items()}


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = pos - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


@dataclass
class PairwiseResult:
    """Bootstrapped A/B result for one scorer.

    ``mean_diff`` is mean(B) minus mean(A) on the paired examples. ``ci_low``
    and ``ci_high`` bound that difference at the requested confidence level.
    ``winner`` is ``"b"`` when the interval is entirely positive, ``"a"`` when
    entirely negative, and ``"tie"`` when it straddles zero.
    """

    scorer: str
    n: int
    mean_a: float
    mean_b: float
    mean_diff: float
    ci_low: float
    ci_high: float
    confidence: float

    @property
    def winner(self) -> str:
        if self.ci_low > 0:
            return "b"
        if self.ci_high < 0:
            return "a"
        return "tie"


def pairwise(
    run_a: list[dict],
    run_b: list[dict],
    *,
    confidence: float = 0.95,
    iterations: int = 2000,
    seed: int = 1234,
) -> list[PairwiseResult]:
    """Compare two runs example-by-example with a paired bootstrap.

    Only examples that succeeded in both runs and carry the same scorer are
    paired. Returns one :class:`PairwiseResult` per scorer present in both runs.
    """
    a_by_idx = {r["idx"]: _scores(r) for r in run_a if r.get("ok")}
    b_by_idx = {r["idx"]: _scores(r) for r in run_b if r.get("ok")}
    shared_idx = sorted(set(a_by_idx) & set(b_by_idx))

    scorers: set[str] = set()
    for idx in shared_idx:
        scorers.update(set(a_by_idx[idx]) & set(b_by_idx[idx]))

    rng = random.Random(seed)
    results: list[PairwiseResult] = []
    alpha = (1 - confidence) / 2

    for scorer in sorted(scorers):
        diffs = [
            b_by_idx[idx][scorer] - a_by_idx[idx][scorer]
            for idx in shared_idx
            if scorer in a_by_idx[idx] and scorer in b_by_idx[idx]
        ]
        n = len(diffs)
        if n == 0:
            continue
        mean_diff = sum(diffs) / n
        mean_a = sum(a_by_idx[idx][scorer] for idx in shared_idx if scorer in a_by_idx[idx]) / n
        mean_b = sum(b_by_idx[idx][scorer] for idx in shared_idx if scorer in b_by_idx[idx]) / n

        boot_means: list[float] = []
        for _ in range(iterations):
            resample = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
            boot_means.append(resample)
        boot_means.sort()
        ci_low = _percentile(boot_means, alpha)
        ci_high = _percentile(boot_means, 1 - alpha)

        results.append(
            PairwiseResult(
                scorer=scorer,
                n=n,
                mean_a=mean_a,
                mean_b=mean_b,
                mean_diff=mean_diff,
                ci_low=ci_low,
                ci_high=ci_high,
                confidence=confidence,
            )
        )
    return results
