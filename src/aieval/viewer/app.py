"""FastAPI + HTMX viewer for runs."""
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from ..backends import get_backend

app = FastAPI(title="aieval viewer")


@app.get("/", response_class=HTMLResponse)
async def index(_req: Request):
    backend = get_backend()
    runs = backend.list_runs()
    rows = "".join(
        f"<tr><td><a href='/runs/{r['id']}'>{r['id'][:8] if isinstance(r['id'], str) else r['id']}</a></td>"
        f"<td>{r['name']}</td><td>{r['model']}</td><td>{r['total']}</td></tr>"
        for r in runs
    )
    return f"""
<!doctype html>
<html><head><title>aieval</title>
<style>
body {{ font-family: ui-sans-serif, system-ui; background: #0a0a14; color: #f5f5f7; padding: 2rem; max-width: 1100px; margin: 0 auto; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ padding: .75rem 1rem; text-align: left; border-top: 1px solid rgba(255,255,255,.1); }}
th {{ font-size: .75rem; text-transform: uppercase; opacity: .6; }}
a {{ color: #a78bfa; text-decoration: none; }}
a:hover {{ text-decoration: underline; }}
h1 {{ font-size: 1.5rem; }}
</style></head>
<body>
<h1>ai-eval-runner</h1>
<p style="opacity:.6">Eval runs</p>
<table>
<thead><tr><th>ID</th><th>Name</th><th>Model</th><th>Examples</th></tr></thead>
<tbody>{rows or '<tr><td colspan=4 style="opacity:.5;text-align:center;padding:2rem;">No runs yet</td></tr>'}</tbody>
</table>
</body></html>
"""


@app.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(run_id: str):
    backend = get_backend()
    run = backend.get_run(run_id)
    results = backend.get_results(run_id)
    if not run:
        return HTMLResponse("<h1>Run not found</h1>", status_code=404)
    rows = "".join(
        f"<tr><td>{r['idx']}</td><td>{'✓' if r['ok'] else '✗'}</td>"
        f"<td><pre style='margin:0;font-size:12px;max-width:380px;overflow:hidden;text-overflow:ellipsis'>{(r.get('prompt') or '')[:120]}</pre></td>"
        f"<td><pre style='margin:0;font-size:12px;max-width:380px;overflow:hidden;text-overflow:ellipsis'>{(r.get('prediction') or '')[:120]}</pre></td>"
        f"<td>{r.get('latency_ms', '')}ms</td></tr>"
        for r in results
    )
    return f"""
<!doctype html><html><head><title>{run['name']}</title>
<style>body{{font-family:ui-sans-serif,system-ui;background:#0a0a14;color:#f5f5f7;padding:2rem;max-width:1300px;margin:0 auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:.6rem .8rem;border-top:1px solid rgba(255,255,255,.08)}}a{{color:#a78bfa}}</style></head>
<body>
<a href="/">&larr; All runs</a>
<h1>{run['name']}</h1>
<p>Model: <code>{run['model']}</code> · Dataset hash: <code>{run.get('dataset_hash','')[:12]}</code> · {run['total']} examples</p>
<table><thead><tr><th>#</th><th>OK</th><th>Prompt</th><th>Prediction</th><th>Latency</th></tr></thead>
<tbody>{rows}</tbody></table></body></html>
"""
