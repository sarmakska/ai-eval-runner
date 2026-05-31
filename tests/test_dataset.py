"""Dataset loading and versioning."""
from pathlib import Path

from aieval.core import dataset
from aieval.core.dataset import DatasetRegistry

FIXTURE = Path(__file__).parent / "fixtures" / "dataset.jsonl"


def test_jsonl_loads_rows():
    rows = list(dataset.jsonl(FIXTURE))
    assert len(rows) == 3
    assert rows[0]["prompt"] == "the cat sat on the mat"


def test_version_is_stable_and_order_independent():
    rows = list(dataset.jsonl(FIXTURE))
    v1 = dataset.version(rows)
    v2 = dataset.version(list(reversed(rows)))
    assert v1 == v2
    assert len(v1) == 12


def test_version_changes_when_a_row_changes():
    rows = list(dataset.jsonl(FIXTURE))
    base = dataset.version(rows)
    edited = [*rows[:-1], {"prompt": "different", "expected": "x"}]
    assert dataset.version(edited) != base


def test_registry_records_new_versions(tmp_path):
    reg = DatasetRegistry(tmp_path / "ds.json")
    rows = list(dataset.jsonl(FIXTURE))
    first = reg.register("eval-set", rows)
    assert first.row_count == 3

    # Registering the same content again does not add a duplicate entry.
    reg.register("eval-set", rows)
    assert len(reg.versions("eval-set")) == 1

    # A changed dataset registers a new version under the same name.
    reg.register("eval-set", [*rows, {"prompt": "new", "expected": "y"}])
    versions = reg.versions("eval-set")
    assert len(versions) == 2
    assert versions[0].version != versions[1].version
