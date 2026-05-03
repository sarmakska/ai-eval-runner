import sqlite3
import json
import uuid
from pathlib import Path


class SqliteBackend:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                model TEXT NOT NULL,
                dataset_hash TEXT NOT NULL,
                total INTEGER NOT NULL,
                started_at INTEGER DEFAULT (strftime('%s', 'now') * 1000),
                finished_at INTEGER
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                idx INTEGER NOT NULL,
                ok INTEGER NOT NULL,
                prompt TEXT,
                expected TEXT,
                prediction TEXT,
                latency_ms INTEGER,
                scores TEXT,
                error TEXT,
                FOREIGN KEY (run_id) REFERENCES runs (id)
            )
        """)
        self.conn.commit()

    def create_run(self, *, name: str, model: str, dataset_hash: str, total: int) -> str:
        run_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO runs (id, name, model, dataset_hash, total) VALUES (?, ?, ?, ?, ?)",
            (run_id, name, model, dataset_hash, total),
        )
        self.conn.commit()
        return run_id

    def persist_results(self, run_id: str, results: list[dict]):
        for r in results:
            self.conn.execute(
                "INSERT INTO results (run_id, idx, ok, prompt, expected, prediction, latency_ms, scores, error) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    r["idx"],
                    1 if r.get("ok") else 0,
                    r.get("prompt"),
                    r.get("expected"),
                    r.get("prediction"),
                    r.get("latency_ms"),
                    json.dumps(r.get("scores", {})),
                    r.get("error"),
                ),
            )
        self.conn.commit()

    def close_run(self, run_id: str):
        self.conn.execute(
            "UPDATE runs SET finished_at = strftime('%s', 'now') * 1000 WHERE id = ?",
            (run_id,),
        )
        self.conn.commit()

    def list_runs(self, limit: int = 50):
        cur = self.conn.execute(
            "SELECT id, name, model, total, started_at, finished_at FROM runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]

    def get_run(self, run_id: str):
        cur = self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        return dict(zip([d[0] for d in cur.description], row))

    def get_results(self, run_id: str):
        cur = self.conn.execute("SELECT * FROM results WHERE run_id = ?", (run_id,))
        return [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]
