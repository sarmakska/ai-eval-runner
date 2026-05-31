# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- LLM-as-judge scorers via `llm_judge`, which grade a prediction against a rubric using a configurable grading model and normalise the verdict to the 0.0 to 1.0 range.
- Regression diff: a real `aieval diff` command and a `/diff/{run_a}/{run_b}` viewer route that show per-scorer mean deltas and the examples that moved most.
- Pairwise A/B comparison with a paired bootstrap (`aieval pairwise` and `aieval.pairwise`), reporting a per-scorer confidence interval and declaring a winner only when the interval clears zero.
- Dataset versioning: an order-independent content version for every dataset and a `DatasetRegistry` that records versions over time so a run is tied to the exact data it scored.
- OpenTelemetry attribute capture for runs and examples, behind an optional `otel` extra, with a no-op fallback when the API is absent. Attributes follow the GenAI semantic conventions where they exist.
- A working CI regression gate: `aieval ci <run_id>` compares a candidate against a baseline run and exits non-zero when any scorer drops past the threshold.
- Scorer context: scorers may now declare `example`, `model`, `provider` and `prompt` keyword parameters, which the runner supplies, so judge scorers can see the full request.
- Runs now record their git SHA and provider alongside the dataset version.
- End-to-end tests with fixtures covering the run, persist, diff, pairwise and CI gate flow, plus unit tests for every new module.

### Changed

- `@scorer` now accepts an optional `name` to override the metric label.
- Bumped dependency floors for typer, pydantic, httpx, duckdb, uvicorn, rich and ruff to current stable releases. All non-breaking; verified against the locked versions.
- CI now also runs on Python 3.14.

### Fixed

- `aieval run <eval.py>` now executes the eval file as `__main__`, so the standard `if __name__ == "__main__":` guard fires and the run actually persists. Previously the command loaded the module under a different name and did nothing.
- Removed dead `if False` branch from the DuckDB backend and made its result `ok` column return a real boolean.

## [1.0.0]

### Added

- Parallel async runner with retry.
- JSONL and in-process dataset loaders.
- Built-in scorers: exact match, JSON validity, ROUGE-L.
- SQLite and DuckDB backends.
- SarmaLink and OpenAI-compatible providers.
- FastAPI plus HTMX viewer.
- Typer CLI with `run`, `list`, `view`, `diff` and `ci` commands.
