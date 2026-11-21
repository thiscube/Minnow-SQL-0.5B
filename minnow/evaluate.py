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


def claim_run_folder(folder: Path, settings: Mapping[str, object]) -> None:
    """Tie a folder to one set of run settings, so a resumed run can't mix two setups."""
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "settings.json"
    if not file.exists():
        file.write_text(json.dumps(dict(settings), indent=2) + "\n", encoding="utf-8")
        return
    saved = json.loads(file.read_text(encoding="utf-8"))
    if saved != dict(settings):
        raise ValueError(f"{folder} holds a run with other settings. Use another folder.")


def drop_unfinished_line(path: Path) -> None:
    """Remove a last line that was cut off when a run was stopped in the middle of writing."""
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


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
