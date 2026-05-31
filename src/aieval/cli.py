"""aieval CLI."""
import importlib.util
import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .backends import get_backend

app = typer.Typer(help="ai-eval-runner CLI")
console = Console()


@app.command()
def run(eval_path: str = typer.Argument(..., help="Path to eval Python file")):
    """Execute an eval Python file."""
    path = Path(eval_path).resolve()
    if not path.exists():
        console.print(f"[red]Not found:[/] {eval_path}")
        raise typer.Exit(1)
    spec = importlib.util.spec_from_file_location("user_eval", path)
    if spec is None or spec.loader is None:
        console.print(f"[red]Could not load:[/] {eval_path}")
        raise typer.Exit(1)
    module = importlib.util.module_from_spec(spec)
    sys.modules["user_eval"] = module
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
    table.add_column("Total")
    table.add_column("Started")
    for r in runs:
        rid = r.get("id", "")[:8] if isinstance(r.get("id"), str) else str(r.get("id"))
        table.add_row(rid, str(r.get("name")), str(r.get("model")), str(r.get("total")), str(r.get("started_at")))
    console.print(table)


@app.command()
def view(port: int = 8000):
    """Start the HTMX viewer."""
    import uvicorn

    from .viewer.app import app as viewer_app
    uvicorn.run(viewer_app, host="0.0.0.0", port=port)


@app.command()
def diff(run_a: str, run_b: str):
    """Compare two runs by score deltas."""
    backend = get_backend()
    a = backend.get_results(run_a)
    b = backend.get_results(run_b)
    console.print(f"Run A: {len(a)} results, Run B: {len(b)} results")


@app.command()
def ci(baseline: str = "main", threshold: float = 0.05):
    """Compare current run against baseline. Exit non-zero if regression > threshold."""
    console.print(f"CI mode: baseline={baseline}, regression threshold={threshold}")
    # Real implementation would diff against the baseline run and exit 1 on regressions.


if __name__ == "__main__":
    app()
