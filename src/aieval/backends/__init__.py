import os
from .sqlite_backend import SqliteBackend
from .duckdb_backend import DuckdbBackend

_singleton = None


def get_backend():
    global _singleton
    if _singleton is None:
        kind = os.getenv("AIEVAL_BACKEND", "sqlite").lower()
        path = os.getenv("AIEVAL_DB_PATH", "./aieval.db")
        if kind == "duckdb":
            _singleton = DuckdbBackend(path)
        else:
            _singleton = SqliteBackend(path)
    return _singleton
