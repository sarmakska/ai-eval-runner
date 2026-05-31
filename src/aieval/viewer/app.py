"""FastAPI + HTMX viewer for runs, with a regression diff view."""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from ..backends import get_backend
from ..core.regression import compare

app = FastAPI(title="aieval viewer")

_STYLE = (
    "body{font-family:ui-sans-serif,system-ui;background:#0a0a14;color:#f5f5f7;"
    "padding:2rem;max-width:1300px;margin:0 auto}"
    "table{border-collapse:collapse;width:100%}"
    "th,td{padding:.6rem .9rem;text-align:left;border-top:1px solid rgba(255,255,255,.1)}"
    "th{font-size:.72rem;text-transform:uppercase;opacity:.6}"
    "a{color:#a78bfa;text-decoration:none}a:hover{text-decoration:underline}"
    "h1{font-size:1.5rem}.pos{color:#34d399}.neg{color:#f87171}code{color:#c4b5fd}"
)


def _page(title: str, body: str) -> str:
    return f"<!doctype html><html><head><title>{title}</title><style>{_STYLE}</style></head><body>{body}</body></html>"


@app.get("/", response_class=HTMLResponse)
async def index(_req: Request):
    backend = get_backend()
    runs = backend.list_runs()
    rows = "".join(
        f"<tr><td><a href='/runs/{r['id']}'>{r['id'][:8] if isinstance(r['id'], str) else r['id']}</a></td>"
        f"<td>{r['name']}</td><td>{r['model']}</td><td><code>{r.get('git_sha', '')}</code></td>"
        f"<td>{r['total']}</td></tr>"
        for r in runs
    )
    body = (
        "<h1>ai-eval-runner</h1><p style='opacity:.6'>Eval runs. "
        "Compare two with <code>/diff/&lt;run_a&gt;/&lt;run_b&gt;</code>.</p>"
        "<table><thead><tr><th>ID</th><th>Name</th><th>Model</th><th>SHA</th><th>Examples</th></tr></thead>"
        f"<tbody>{rows or '<tr><td colspan=5 style=\"opacity:.5;text-align:center;padding:2rem\">No runs yet</td></tr>'}</tbody></table>"
    )
    return _page("aieval", body)


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(run_id: str):
    backend = get_backend()
    run = backend.get_run(run_id)
    results = backend.get_results(run_id)
    if not run:
        return HTMLResponse(_page("Not found", "<h1>Run not found</h1>"), status_code=404)
    rows = "".join(
        f"<tr><td>{r['idx']}</td><td>{'pass' if r['ok'] else 'fail'}</td>"
        f"<td><pre style='margin:0;font-size:12px;max-width:380px;overflow:hidden'>{(r.get('prompt') or '')[:120]}</pre></td>"
        f"<td><pre style='margin:0;font-size:12px;max-width:380px;overflow:hidden'>{(r.get('prediction') or '')[:120]}</pre></td>"
        f"<td>{r.get('latency_ms', '')}ms</td></tr>"
        for r in results
    )
    body = (
        "<a href='/'>&larr; All runs</a>"
        f"<h1>{run['name']}</h1>"
        f"<p>Model: <code>{run['model']}</code> &middot; SHA: <code>{run.get('git_sha', '')}</code> &middot; "
        f"Dataset version: <code>{run.get('dataset_hash', '')[:12]}</code> &middot; {run['total']} examples</p>"
        "<table><thead><tr><th>#</th><th>Result</th><th>Prompt</th><th>Prediction</th><th>Latency</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )
    return _page(run["name"], body)


@app.get("/diff/{run_a}/{run_b}", response_class=HTMLResponse)
async def diff_view(run_a: str, run_b: str):
    """Regression diff: per-scorer deltas and the examples that moved most."""
    backend = get_backend()
    a = backend.get_results(run_a)
    b = backend.get_results(run_b)
    if not a or not b:
        return HTMLResponse(_page("Diff", "<h1>One or both runs have no results.</h1>"), status_code=404)

    report = compare(a, b)
    delta_rows = "".join(
        f"<tr><td>{d.scorer}</td><td>{d.baseline_mean:.3f}</td><td>{d.candidate_mean:.3f}</td>"
        f"<td class='{'pos' if d.delta >= 0 else 'neg'}'>{d.delta:+.3f}</td></tr>"
        for d in report.scorer_deltas
    )

    example_blocks = ""
    for d in report.scorer_deltas:
        movers = [e for e in report.worst_examples(d.scorer, limit=5) if e.delta != 0]
        if not movers:
            continue
        mrows = "".join(
            f"<tr><td>{e.idx}</td><td>{e.baseline_score:.2f}</td><td>{e.candidate_score:.2f}</td>"
            f"<td class='{'pos' if e.delta >= 0 else 'neg'}'>{e.delta:+.2f}</td></tr>"
            for e in movers
        )
        example_blocks += (
            f"<h2 style='font-size:1.1rem;margin-top:1.5rem'>{d.scorer}: examples that moved</h2>"
            "<table><thead><tr><th>#</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead>"
            f"<tbody>{mrows}</tbody></table>"
        )

    body = (
        "<a href='/'>&larr; All runs</a>"
        f"<h1>Diff {run_a[:8]} &rarr; {run_b[:8]}</h1>"
        "<table><thead><tr><th>Scorer</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead>"
        f"<tbody>{delta_rows}</tbody></table>{example_blocks}"
    )
    return _page("Diff", body)
