"""Regression comparison and CI gate logic."""
from aieval.core.regression import compare


def _results(scores_by_idx):
    return [{"idx": i, "ok": True, "scores": s} for i, s in scores_by_idx.items()]


def test_compare_computes_scorer_deltas():
    baseline = _results({0: {"acc": 1.0}, 1: {"acc": 1.0}})
    candidate = _results({0: {"acc": 1.0}, 1: {"acc": 0.0}})
    report = compare(baseline, candidate)
    delta = next(d for d in report.scorer_deltas if d.scorer == "acc")
    assert delta.baseline_mean == 1.0
    assert delta.candidate_mean == 0.5
    assert delta.delta == -0.5


def test_regressions_respects_threshold():
    baseline = _results({0: {"acc": 1.0}, 1: {"acc": 1.0}})
    candidate = _results({0: {"acc": 1.0}, 1: {"acc": 0.0}})
    report = compare(baseline, candidate)
    assert report.regressions(threshold=0.6) == []
    assert len(report.regressions(threshold=0.4)) == 1


def test_example_diffs_capture_movers():
    baseline = _results({0: {"acc": 1.0}, 1: {"acc": 1.0}})
    candidate = _results({0: {"acc": 0.0}, 1: {"acc": 1.0}})
    report = compare(baseline, candidate)
    worst = report.worst_examples("acc")
    assert worst[0].idx == 0
    assert worst[0].delta == -1.0


def test_compare_handles_json_string_scores():
    baseline = [{"idx": 0, "ok": True, "scores": '{"acc": 1.0}'}]
    candidate = [{"idx": 0, "ok": True, "scores": '{"acc": 0.0}'}]
    report = compare(baseline, candidate)
    assert report.scorer_deltas[0].delta == -1.0
