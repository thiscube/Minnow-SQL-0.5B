"""Read a run's files and make sure they agree with each other and with the results log.

A run folder holds predictions.jsonl, outcomes.jsonl and summary.json. Nothing here writes.
"""

import hashlib
import json
from pathlib import Path

from minnow.evaluate import load_predictions
from minnow.outcomes import (
    MISSING_PREDICTION,
    OUTCOMES_FILE,
    PREDICTIONS_FILE,
    Breakdown,
    Outcome,
    breakdown,
    load_outcomes,
)
from minnow.report import Run

SUMMARY_FILE = "summary.json"
RUN_FILES = (PREDICTIONS_FILE, SUMMARY_FILE, OUTCOMES_FILE)
_SUMMARY_KEYS = ("model", "correct", "total", "execution_accuracy", "generation_seconds")
_JUDGE_AGAIN = "Judge the run again with python -m minnow.outcomes."


class RunLogError(ValueError):
    """The results log cannot be written from the files as they are."""


def read_text(path: Path) -> str:
    """Return a file's text with plain line endings, or nothing when the file does not exist."""
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def accuracy(correct: int, total: int) -> float:
    """Return the share of correct predictions as a percentage with two decimals."""
    return round(100 * correct / total, 2)


def fingerprint(folder: Path) -> str:
    """Return a hash of a run's files that is the same on every operating system."""
    digest = hashlib.sha256()
    for name in RUN_FILES:
        digest.update(f"{name}\n{read_text(folder / name)}".encode())
    return digest.hexdigest()


def _outcomes(folder: Path) -> list[Outcome]:
    for name in RUN_FILES:
        if not (folder / name).is_file():
            hint = " Run python -m minnow.outcomes on it first." if name == OUTCOMES_FILE else ""
            raise RunLogError(f"{(folder / name).as_posix()} does not exist.{hint}")
    try:
        return load_outcomes(folder / OUTCOMES_FILE)
    except ValueError as error:
        raise RunLogError(str(error)) from error


def _summary(folder: Path) -> dict[str, object]:
    return json.loads(read_text(folder / SUMMARY_FILE))


def judged_run(folder: Path, change: str, day: str) -> Run:
    """Build the log record of a run folder, refusing files that disagree with each other."""
    outcomes = _outcomes(folder)
    counts = breakdown(outcomes)
    summary = _summary(folder)
    if (summary.get("correct"), summary.get("total")) != (counts.correct, counts.total):
        raise RunLogError(
            f"{SUMMARY_FILE} says {summary.get('correct')} of {summary.get('total')} correct, "
            f"but {OUTCOMES_FILE} has {counts.correct} of {counts.total}. {_JUDGE_AGAIN}"
        )
    judged = {outcome.index for outcome in outcomes if outcome.error != MISSING_PREDICTION}
    if set(load_predictions(folder / PREDICTIONS_FILE)) != judged:
        raise RunLogError(
            f"{PREDICTIONS_FILE} and {OUTCOMES_FILE} do not cover the same questions. "
            f"{_JUDGE_AGAIN}"
        )
    return Run(
        name=folder.resolve().name,
        date=day,
        model=str(summary["model"]),
        change=" ".join(change.split()),
        correct=counts.correct,
        total=counts.total,
        execution_accuracy=accuracy(counts.correct, counts.total),
        generation_seconds=summary.get("generation_seconds"),
        fingerprint=fingerprint(folder),
    )


def verified_counts(folder: Path, run: Run) -> Breakdown:
    """Count a logged run's outcomes, refusing files that no longer say what was logged."""
    counts = breakdown(_outcomes(folder))
    judged = (counts.correct, counts.total, accuracy(counts.correct, counts.total))
    if (run.correct, run.total, run.execution_accuracy) != judged:
        raise RunLogError(
            f"The log gives {run.name} {run.correct} of {run.total} correct "
            f"({run.execution_accuracy}%), but its {OUTCOMES_FILE} has {judged[0]} of "
            f"{judged[1]} ({judged[2]}%)."
        )
    logged = (run.model, run.correct, run.total, run.execution_accuracy, run.generation_seconds)
    summary = _summary(folder)
    if tuple(summary.get(key) for key in _SUMMARY_KEYS) != logged:
        raise RunLogError(f"The log and {SUMMARY_FILE} disagree about {run.name}.")
    if fingerprint(folder) != run.fingerprint:
        raise RunLogError(
            f"The files of {run.name} changed after it was logged. Log it again with "
            "python -m minnow.runlog add --replace."
        )
    return counts
