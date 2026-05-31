"""Dataset loaders."""
import json
from collections.abc import Iterator
from pathlib import Path


def jsonl(path: str | Path) -> Iterator[dict]:
    p = Path(path)
    with p.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def from_list(rows: list[dict]) -> Iterator[dict]:
    yield from rows
