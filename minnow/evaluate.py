"""Save predictions as they are made, and score them by execution accuracy."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from minnow.execution import execution_match
from minnow.spider import SpiderExample, database_path


@dataclass(frozen=True)
class Score:
    """How many predictions returned the same rows as the reference queries."""

    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        """Execution accuracy as a percentage."""
        return 100 * self.correct / self.total


def load_predictions(path: Path) -> dict[int, str]:
    """Read saved predictions, keyed by the question's position in the split."""
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {record["index"]: record["sql"] for record in records}


def append_prediction(path: Path, index: int, sql: str) -> None:
    """Add one prediction to the file, so a stopped run can continue where it left off."""
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps({"index": index, "sql": sql}) + "\n")


def score(examples: Sequence[SpiderExample], predictions: Mapping[int, str], spider: Path) -> Score:
    """Score every example; a missing prediction counts as wrong."""
    correct = sum(
        index in predictions
        and execution_match(database_path(spider, example.db_id), predictions[index], example.query)
        for index, example in enumerate(examples)
    )
    return Score(correct, len(examples))
