"""Regression comparison between two runs.

Given the persisted results of a baseline run and a candidate run this module
computes per-scorer mean deltas and a per-example diff (which examples improved,
regressed or stayed level on each scorer). The CI gate uses the same comparison
to decide whether a candidate may merge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


@dataclass
class ScorerDelta:
    """Mean score change for one scorer between baseline and candidate."""

    scorer: str
    baseline_mean: float
    candidate_mean: float

    @property
    def delta(self) -> float:
        return self.candidate_mean - self.baseline_mean


@dataclass
class ExampleDiff:
    """Per-example score change on a single scorer."""

    idx: int
    scorer: str
    baseline_score: float
    candidate_score: float

    @property
    def delta(self) -> float:
        return self.candidate_score - self.baseline_score


@dataclass
class RegressionReport:
    """The full comparison of a candidate run against a baseline run."""

    scorer_deltas: list[ScorerDelta] = field(default_factory=list)
    example_diffs: list[ExampleDiff] = field(default_factory=list)

    def regressions(self, threshold: float) -> list[ScorerDelta]:
        """Scorers whose mean dropped by more than ``threshold``."""
        return [d for d in self.scorer_deltas if d.delta < -threshold]

    def worst_examples(self, scorer: str, limit: int = 10) -> list[ExampleDiff]:
        ranked = sorted(
            (e for e in self.example_diffs if e.scorer == scorer),
            key=lambda e: e.delta,
        )
        return ranked[:limit]


def compare(baseline: list[dict], candidate: list[dict]) -> RegressionReport:
    """Compare two sets of run results and return a :class:`RegressionReport`."""
    base_by_idx = {r["idx"]: _scores(r) for r in baseline if r.get("ok")}
    cand_by_idx = {r["idx"]: _scores(r) for r in candidate if r.get("ok")}

    scorers: set[str] = set()
    for s in base_by_idx.values():
        scorers.update(s)
    for s in cand_by_idx.values():
        scorers.update(s)

    report = RegressionReport()
    for scorer in sorted(scorers):
        base_vals = [s[scorer] for s in base_by_idx.values() if scorer in s]
        cand_vals = [s[scorer] for s in cand_by_idx.values() if scorer in s]
        report.scorer_deltas.append(
            ScorerDelta(
                scorer=scorer,
                baseline_mean=_mean(base_vals),
                candidate_mean=_mean(cand_vals),
            )
        )
        shared = sorted(set(base_by_idx) & set(cand_by_idx))
        for idx in shared:
            b = base_by_idx[idx].get(scorer)
            c = cand_by_idx[idx].get(scorer)
            if b is None or c is None or b == c:
                continue
            report.example_diffs.append(
                ExampleDiff(idx=idx, scorer=scorer, baseline_score=b, candidate_score=c)
            )
    return report
