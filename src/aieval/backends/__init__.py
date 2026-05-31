import os

from .duckdb_backend import DuckdbBackend
from .sqlite_backend import SqliteBackend

_singleton = None


def get_backend():
    global _singleton
    if _singleton is None:
        kind = os.getenv("AIEVAL_BACKEND", "sqlite").lower()
        path = os.getenv("AIEVAL_DB_PATH", "./aieval.db")
        _singleton = DuckdbBackend(path) if kind == "duckdb" else SqliteBackend(path)
    return _singleton
