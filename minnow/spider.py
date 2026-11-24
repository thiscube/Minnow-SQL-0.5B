"""Read the Spider dataset from its unpacked folder."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from minnow.schema import create_statements


@dataclass(frozen=True)
class SpiderExample:
    """One Spider question with its reference query and database."""

    question: str
    query: str
    db_id: str


def load_split(folder: Path, filename: str) -> list[SpiderExample]:
    """Load a Spider split, such as dev.json, from the dataset folder."""
    items = json.loads((folder / filename).read_text(encoding="utf-8"))
    return [SpiderExample(item["question"], item["query"], item["db_id"]) for item in items]


def database_path(folder: Path, db_id: str) -> Path:
    """Return the SQLite file of a Spider database, which for the test split is in test_database.

    The test databases share no names with the others, so a name is looked up in database
    first and in test_database only when it is not there.
    """
    path = folder / "database" / db_id / f"{db_id}.sqlite"
    test = folder / "test_database" / db_id / f"{db_id}.sqlite"
    return test if test.exists() and not path.exists() else path


def schemas(
    folder: Path, examples: Iterable[SpiderExample], sample_rows: int = 0
) -> dict[str, str]:
    """Return the schema text of every database the examples use, by database id."""
    db_ids = sorted({example.db_id for example in examples})
    return {db_id: create_statements(database_path(folder, db_id), sample_rows) for db_id in db_ids}
