"""Dataset loaders and versioning.

Datasets are sequences of dicts, each with at least a ``prompt`` key and an
optional ``expected`` key. Beyond loading, this module gives every dataset a
stable content version so a run can be tied to the exact data it scored. The
version is a short SHA-256 over the canonical JSON of every row, order
independent, so reordering rows does not change the version but editing,
adding or removing a row does.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path


def jsonl(path: str | Path) -> Iterator[dict]:
    """Yield rows from a JSONL file, skipping blank lines."""
    p = Path(path)
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def from_list(rows: list[dict]) -> Iterator[dict]:
    """Yield rows from an in-process list."""
    yield from rows


def version(rows: Iterable[dict]) -> str:
    """Return the stable 12-character content version for a dataset.

    The hash is order independent: each row is canonicalised to sorted-key
    JSON, the row hashes are sorted, then hashed together. Two datasets with
    the same rows in a different order share a version.
    """
    row_hashes = sorted(
        hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        for row in rows
    )
    digest = hashlib.sha256("\n".join(row_hashes).encode())
    return digest.hexdigest()[:12]


@dataclass(frozen=True)
class DatasetVersion:
    """A recorded version of a named dataset."""

    name: str
    version: str
    row_count: int


class DatasetRegistry:
    """Records dataset versions so changes between runs are auditable.

    The registry is backed by a JSON file. Registering a dataset returns its
    :class:`DatasetVersion` and appends a new entry only when the content
    version is one it has not seen for that name.
    """

    def __init__(self, path: str | Path = "./aieval-datasets.json"):
        self.path = Path(path)
        self._entries: list[dict] = []
        if self.path.exists():
            self._entries = json.loads(self.path.read_text() or "[]")

    def register(self, name: str, rows: Iterable[dict]) -> DatasetVersion:
        materialised = list(rows)
        ver = version(materialised)
        dv = DatasetVersion(name=name, version=ver, row_count=len(materialised))
        seen = any(e["name"] == name and e["version"] == ver for e in self._entries)
        if not seen:
            self._entries.append({"name": name, "version": ver, "row_count": len(materialised)})
            self._flush()
        return dv

    def versions(self, name: str) -> list[DatasetVersion]:
        return [
            DatasetVersion(name=e["name"], version=e["version"], row_count=e["row_count"])
            for e in self._entries
            if e["name"] == name
        ]

    def _flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._entries, indent=2))
