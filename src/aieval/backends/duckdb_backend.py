"""DuckDB backend. Same shape as SQLite, faster for large analytical queries."""
import json
import uuid

import duckdb


class DuckdbBackend:
    def __init__(self, path: str):
        self.conn = duckdb.connect(path)
        self.conn.execute("CREATE SEQUENCE IF NOT EXISTS seq_results START 1")
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS runs (
                id VARCHAR PRIMARY KEY,
                name VARCHAR NOT NULL,
                model VARCHAR NOT NULL,
                dataset_hash VARCHAR NOT NULL,
                total INTEGER NOT NULL,
                git_sha VARCHAR,
                provider VARCHAR,
                started_at TIMESTAMP DEFAULT current_timestamp,
                finished_at TIMESTAMP
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY,
                run_id VARCHAR NOT NULL,
                idx INTEGER NOT NULL,
                ok BOOLEAN NOT NULL,
                prompt VARCHAR, expected VARCHAR, prediction VARCHAR,
                latency_ms INTEGER, scores VARCHAR, error VARCHAR
            )
        """)

    def create_run(
        self,
        *,
        name: str,
        model: str,
        dataset_hash: str,
        total: int,
        git_sha: str = "unknown",
        provider: str = "",
    ) -> str:
        run_id = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO runs (id, name, model, dataset_hash, total, git_sha, provider) VALUES (?, ?, ?, ?, ?, ?, ?)",
            [run_id, name, model, dataset_hash, total, git_sha, provider],
        )
        return run_id

    def persist_results(self, run_id: str, results: list[dict]):
        for r in results:
            self.conn.execute(
                "INSERT INTO results (id, run_id, idx, ok, prompt, expected, prediction, latency_ms, scores, error) "
                "VALUES (nextval('seq_results'), ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    run_id,
                    r["idx"],
                    bool(r.get("ok")),
                    r.get("prompt"),
                    r.get("expected"),
                    r.get("prediction"),
                    r.get("latency_ms"),
                    json.dumps(r.get("scores", {})),
                    r.get("error"),
                ],
            )

    def close_run(self, run_id: str):
        self.conn.execute("UPDATE runs SET finished_at = current_timestamp WHERE id = ?", [run_id])

    def list_runs(self, limit: int = 50):
        return self.conn.execute(
            f"SELECT * FROM runs ORDER BY started_at DESC LIMIT {limit}"
        ).fetchdf().to_dict(orient="records")

    def get_run(self, run_id: str):
        rows = self.conn.execute("SELECT * FROM runs WHERE id = ?", [run_id]).fetchdf().to_dict(orient="records")
        return rows[0] if rows else None

    def get_results(self, run_id: str):
        rows = self.conn.execute("SELECT * FROM results WHERE run_id = ?", [run_id]).fetchdf().to_dict(orient="records")
        for r in rows:
            r["ok"] = bool(r["ok"])
        return rows

    def find_baseline(self, name: str, exclude_run_id: str | None = None) -> dict | None:
        rows = self.conn.execute(
            "SELECT * FROM runs WHERE name = ? AND id != ? ORDER BY started_at DESC LIMIT 1",
            [name, exclude_run_id or ""],
        ).fetchdf().to_dict(orient="records")
        return rows[0] if rows else None
