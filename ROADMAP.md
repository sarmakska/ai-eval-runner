# Roadmap

## Shipped

- Parallel async runner with retry, git SHA and dataset version tagging
- JSONL and in-process dataset loaders
- Dataset versioning with an order-independent content version and a registry
- Scorer decorator with optional naming and context injection
- Built-in scorers: exact match, JSON validity, ROUGE-L
- LLM-as-judge scorers graded against a rubric
- SQLite and DuckDB backends
- SarmaLink and OpenAI-compatible providers
- Regression diff: `aieval diff` and the viewer diff route
- CI regression gate: `aieval ci` with a per-scorer threshold and a non-zero exit on regression
- Pairwise A/B with bootstrapped confidence intervals: `aieval pairwise`
- OpenTelemetry attribute capture behind an optional extra
- FastAPI plus HTMX viewer
- Typer CLI

## Planned

- BLEU and BERTScore scorers
- Calibration sets for judge scorers, with agreement metrics against human labels
- HuggingFace dataset hub integration
- Postgres backend
- Streaming view of in-progress runs
- Cost tracking per run

## Will not ship

- A vendor-locked, hosted eval platform. This is open source on purpose.
- Real-time online evaluation. Use APM tools for that.
- Auto-fix-the-prompt. Humans should review prompt changes.

## Contributing

PRs welcome. Pick from "Planned", open an issue, fork, branch, push, open a PR.
Keep them small and focused.

I will not merge:

- Framework swaps. Typer and FastAPI stay.
- Sync runners. Everything is async.
- Adapters for providers with no free tier path.

Releases: see [GitHub Releases](https://github.com/sarmakska/ai-eval-runner/releases).
