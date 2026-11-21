"""Read the Spider dataset from its unpacked folder."""

import json
from dataclasses import dataclass
from pathlib import Path


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
    """Return the SQLite file of a Spider database."""
    return folder / "database" / db_id / f"{db_id}.sqlite"
