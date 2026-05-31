"""aieval CLI."""
import importlib.util
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .backends import get_backend
from .core.pairwise import pairwise as run_pairwise
from .core.regression import compare

app = typer.Typer(help="ai-eval-runner CLI")
console = Console()


@app.command()
def run(eval_path: str = typer.Argument(..., help="Path to eval Python file")):
    """Execute an eval Python file."""
    path = Path(eval_path).resolve()
    if not path.exists():
        console.print(f"[red]Not found:[/] {eval_path}")
        raise typer.Exit(1)
    # Execute the eval as if it were the main module so the standard
    # `if __name__ == "__main__":` guard in the eval file fires.
    spec = importlib.util.spec_from_file_location("__main__", path)
    if spec is None or spec.loader is None:
        console.print(f"[red]Could not load:[/] {eval_path}")
        raise typer.Exit(1)
    module = importlib.util.module_from_spec(spec)
    sys.modules["__main__"] = module
    spec.loader.exec_module(module)


@app.command()
def list():
    """List all runs."""
    backend = get_backend()
    runs = backend.list_runs()
    if not runs:
        console.print("[yellow]No runs yet.[/]")
        return
    table = Table(title="Eval Runs")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Model")
    table.add_column("SHA")
    table.add_column("Total")
    table.add_column("Started")
    for r in runs:
        rid = r.get("id", "")[:8] if isinstance(r.get("id"), str) else str(r.get("id"))
        table.add_row(
            rid,
            str(r.get("name")),
            str(r.get("model")),
            str(r.get("git_sha", "")),
            str(r.get("total")),
            str(r.get("started_at")),
        )
    console.print(table)


@app.command()
def view(port: int = 8000):
    """Start the HTMX viewer."""
    import uvicorn

    from .viewer.app import app as viewer_app
    uvicorn.run(viewer_app, host="0.0.0.0", port=port)


@app.command()
def diff(run_a: str, run_b: str):
    """Compare two runs: per-scorer mean deltas and the examples that moved most."""
    backend = get_backend()
    a = backend.get_results(run_a)
    b = backend.get_results(run_b)
    if not a or not b:
        console.print("[red]One or both runs have no results.[/]")
        raise typer.Exit(1)

    report = compare(a, b)
    table = Table(title=f"Scorer deltas: {run_a[:8]} -> {run_b[:8]}")
    table.add_column("Scorer")
    table.add_column("Baseline", justify="right")
    table.add_column("Candidate", justify="right")
    table.add_column("Delta", justify="right")
    for d in report.scorer_deltas:
        colour = "green" if d.delta >= 0 else "red"
        table.add_row(
            d.scorer,
            f"{d.baseline_mean:.3f}",
            f"{d.candidate_mean:.3f}",
            f"[{colour}]{d.delta:+.3f}[/]",
        )
    console.print(table)

    for d in report.scorer_deltas:
        worst = report.worst_examples(d.scorer, limit=3)
        movers = [e for e in worst if e.delta < 0]
        if movers:
            console.print(f"\n[bold]{d.scorer}[/] biggest regressions:")
            for e in movers:
                console.print(f"  example {e.idx}: {e.baseline_score:.2f} -> {e.candidate_score:.2f} ({e.delta:+.2f})")


@app.command()
def ci(
    run_id: str = typer.Argument(..., help="Candidate run id to gate"),
    baseline: str = typer.Option("", help="Baseline run id; defaults to the previous run of the same name"),
    threshold: float = typer.Option(0.05, help="Max allowed mean drop per scorer"),
):
    """Gate a candidate run against a baseline. Exit non-zero on regression."""
    backend = get_backend()
    candidate = backend.get_results(run_id)
    if not candidate:
        console.print(f"[red]No results for run {run_id}.[/]")
        raise typer.Exit(2)

    if not baseline:
        run = backend.get_run(run_id)
        prior = backend.find_baseline(run["name"], exclude_run_id=run_id) if run else None
        if prior is None:
            console.print("[yellow]No baseline run found. Nothing to compare; passing.[/]")
            return
        baseline = prior["id"]

    base_results = backend.get_results(baseline)
    report = compare(base_results, candidate)
    regressions = report.regressions(threshold)

    table = Table(title=f"CI gate: baseline {baseline[:8]} vs candidate {run_id[:8]} (threshold {threshold})")
    table.add_column("Scorer")
    table.add_column("Delta", justify="right")
    table.add_column("Status")
    for d in report.scorer_deltas:
        regressed = d.delta < -threshold
        status = "[red]REGRESSED[/]" if regressed else "[green]ok[/]"
        table.add_row(d.scorer, f"{d.delta:+.3f}", status)
    console.print(table)

    if regressions:
        names = ", ".join(d.scorer for d in regressions)
        console.print(f"[red]Regression gate failed on: {names}[/]")
        raise typer.Exit(1)
    console.print("[green]No regressions past threshold.[/]")


@app.command()
def pairwise(
    run_a: str,
    run_b: str,
    confidence: float = typer.Option(0.95, help="Confidence level for the interval"),
    iterations: int = typer.Option(2000, help="Bootstrap resamples"),
):
    """Paired A/B with a bootstrapped confidence interval per scorer."""
    backend = get_backend()
    a = backend.get_results(run_a)
    b = backend.get_results(run_b)
    if not a or not b:
        console.print("[red]One or both runs have no results.[/]")
        raise typer.Exit(1)

    results = run_pairwise(a, b, confidence=confidence, iterations=iterations)
    if not results:
        console.print("[yellow]No paired, scored examples shared between the runs.[/]")
        return

    table = Table(title=f"Pairwise A/B: A={run_a[:8]} B={run_b[:8]} ({int(confidence * 100)}% CI)")
    table.add_column("Scorer")
    table.add_column("n", justify="right")
    table.add_column("mean A", justify="right")
    table.add_column("mean B", justify="right")
    table.add_column("diff (B-A)", justify="right")
    table.add_column("CI")
    table.add_column("Winner")
    for r in results:
        winner = {"a": "[cyan]A[/]", "b": "[magenta]B[/]", "tie": "tie"}[r.winner]
        table.add_row(
            r.scorer,
            str(r.n),
            f"{r.mean_a:.3f}",
            f"{r.mean_b:.3f}",
            f"{r.mean_diff:+.3f}",
            f"[{r.ci_low:+.3f}, {r.ci_high:+.3f}]",
            winner,
        )
    console.print(table)


if __name__ == "__main__":
    app()
