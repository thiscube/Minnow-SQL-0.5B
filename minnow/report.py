"""Write the text of the results log from the logged runs and their counted outcomes.

Nothing here reads a file or the clock: the same runs always give the same text.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from minnow.outcomes import Breakdown

CHART_FILE = "results/chart.svg"
_INTRO = (
    "Execution accuracy (EX) on the Spider development set{size}. A prediction is "
    "correct when it runs and returns the same rows as the reference query. Scoring is done by "
    "[`minnow/execution.py`](minnow/execution.py) on the original Spider databases."
)
_LOG_HEADER = (
    "| Date | Model | Change | Dev EX (%) | Correct | Files |",
    "| --- | --- | --- | ---: | ---: | --- |",
)
_README_HEADER = (
    "| Run | Change | Spider dev EX (%) | Correct |",
    "| --- | --- | ---: | ---: |",
)
_EMPTY_RESULT_CAVEAT = (
    "With a single database per question, such a match cannot be told apart from a wrong query "
    "that also returns nothing."
)
_ERROR_KINDS_SHOWN = 3


@dataclass(frozen=True)
class Run:
    """One logged run, as it is stored in results/runs.json."""

    name: str
    date: str
    model: str
    change: str
    correct: int
    total: int
    execution_accuracy: float
    generation_seconds: int | None
    fingerprint: str = ""


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


def _model_and_change(run: Run) -> list[str]:
    return [run.model.rsplit("/", 1)[-1], run.change]


def _score(run: Run) -> list[str]:
    return [f"{run.execution_accuracy:.2f}", f"{run.correct} / {run.total}"]


def _counted(count: int, noun: str) -> str:
    return f"{count:,} {noun}" + ("" if count == 1 else "s")


def _notes(run: Run, counts: Breakdown) -> list[str]:
    did_not_run = f"{_counted(counts.did_not_run, 'prediction')} did not run."
    if counts.error_classes:
        kinds = counts.error_classes[:_ERROR_KINDS_SHOWN]
        listed = ", ".join(f"`{kind}` ({count:,})" for kind, count in kinds)
        did_not_run += f" The most common errors: {listed}."
    notes = [
        did_not_run,
        f"{_counted(counts.wrong_rows, 'prediction')} ran and returned different rows.",
    ]
    if counts.correct_on_empty:
        notes.append(
            f"{counts.correct_on_empty:,} of the {counts.correct:,} correct predictions matched "
            f"a reference query that returns no rows. {_EMPTY_RESULT_CAVEAT}"
        )
    if run.generation_seconds is not None:
        predictions = _counted(counts.total, "prediction")
        seconds = _counted(run.generation_seconds, "second")
        notes.append(f"Generating all {predictions} took {seconds}.")
    return notes


def results_markdown(runs: Sequence[Run], counts: Mapping[str, Breakdown]) -> str:
    """Return results.md: a table row per run in the order given, then notes on each run."""
    size = f", {runs[0].total:,} questions" if runs else ""
    lines = ["# Results", "", _INTRO.format(size=size), "", *_LOG_HEADER]
    for run in runs:
        files = f"[results/{run.name}](results/{run.name})"
        lines.append(_row([run.date, *_model_and_change(run), *_score(run), files]))
    lines += ["", "## Notes"]
    for run in runs:
        notes = [f"- {note}" for note in _notes(run, counts[run.name])]
        lines += ["", f"### {run.date}, {run.name}", "", *notes]
    return "\n".join(lines) + "\n"


def readme_block(runs: Sequence[Run]) -> str:
    """Return the README's results block: the chart, then a folded table of the runs."""
    rows = [_row([run.name, run.change, *_score(run)]) for run in runs]
    image = f"![Execution accuracy by run]({CHART_FILE})"
    opened = f"<details>\n<summary>What each of the {len(runs)} runs changed</summary>\n"
    return "\n".join([image, "", opened, *_README_HEADER, *rows, "", "</details>"]) + "\n"
