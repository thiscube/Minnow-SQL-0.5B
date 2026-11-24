"""Save predictions as they are made, so a stopped run can continue where it left off."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

CANDIDATES_FILE = "candidates.jsonl"


def claim_run_folder(folder: Path, settings: Mapping[str, object]) -> None:
    """Tie a folder to one set of run settings, so a resumed run can't mix two setups."""
    folder.mkdir(parents=True, exist_ok=True)
    file = folder / "settings.json"
    if not file.exists():
        text = json.dumps(dict(settings), indent=2) + "\n"
        file.write_text(text, encoding="utf-8", newline="\n")
        return
    saved = json.loads(file.read_text(encoding="utf-8"))
    if saved != dict(settings):
        raise ValueError(f"{folder} holds a run with other settings. Use another folder.")


def add_generation_seconds(folder: Path, seconds: float) -> int | None:
    """Add the time one invocation spent generating to the run's total, and return the total.

    A run can be stopped and resumed, so the time is summed across invocations. The total is
    unknown, and None, when nothing was generated now and no earlier time was recorded.
    """
    file = folder / "timing.json"
    if not file.exists() and not seconds:
        return None
    earlier = json.loads(file.read_text(encoding="utf-8"))["seconds"] if file.exists() else 0.0
    total = earlier + seconds
    file.write_text(json.dumps({"seconds": total}) + "\n", encoding="utf-8", newline="\n")
    return round(total)


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
    """Add one prediction to the end of the file."""
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps({"index": index, "sql": sql}) + "\n")


def append_candidates(path: Path, index: int, candidates: Sequence[str]) -> None:
    """Add every candidate query of one question to the end of the file, greedy one first."""
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps({"index": index, "candidates": list(candidates)}) + "\n")


def load_candidates(path: Path) -> dict[int, list[str]]:
    """Read saved candidates, keyed by the question's position in the split."""
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {record["index"]: record["candidates"] for record in records}
