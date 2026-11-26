"""Write the README's score table from the files of the runs it names.

heldout/standing.json describes the table: for every column a title and the folder that holds
the runs of one question set, for every row a label and one run name per column (null leaves
the cell empty). Every score is read from that run's files, so none is typed.
"""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from minnow.runfiles import RunLogError, judged_run, read_text

STANDING_FILE = "heldout/standing.json"
STANDING_START = "<!-- standing:start -->"
STANDING_END = "<!-- standing:end -->"


def _line(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _column(root: Path, column: Mapping[str, str], names: Sequence[str | None]) -> list[str]:
    """Return a column's header, then one cell per row."""
    folder = root / column["folder"]
    runs = [judged_run(folder / name, "", "") if name else None for name in names]
    totals = {run.total for run in runs if run}
    if len(totals) != 1:
        raise RunLogError(
            f"The runs of {column['folder']} named in {STANDING_FILE} have to cover the same "
            "questions."
        )
    header = f"{column['title']} ({totals.pop():,} questions)"
    return [header, *(f"{run.execution_accuracy:.2f}%" if run else "" for run in runs)]


def _highlights(root: Path, table: Mapping) -> list[str]:
    """Return the headline list: first row against second row, and the second row's last cell."""
    first, last = table["columns"][0]["folder"], table["columns"][-1]["folder"]
    before = judged_run(root / first / table["rows"][0]["runs"][0], "", "")
    after = judged_run(root / first / table["rows"][1]["runs"][0], "", "")
    unseen = judged_run(root / last / table["rows"][1]["runs"][-1], "", "")
    published = table["published"]
    wrong_before, wrong_after = before.total - before.correct, after.total - after.correct
    fewer = wrong_before / wrong_after
    mistakes = "Half the mistakes" if abs(fewer - 2) < 0.25 else f"{fewer:.1f} times fewer mistakes"
    return [
        f"- **{mistakes}:** {round(100 * wrong_before / before.total)} of 100 questions wrong "
        f"before training, now {round(100 * wrong_after / after.total)}.",
        f"- **{after.correct / before.correct:.1f} times as many right:** {before.correct:,} of "
        f"{before.total:,} before training, now {after.correct:,}.",
        f"- **{unseen.execution_accuracy:.0f}% on questions it never saw:** the published "
        f"{published['name']}, same size, reports {published['score']:g}% on Spider test.",
    ]


def standing_table(root: Path) -> str:
    """Return the headline list and the score table that heldout/standing.json describes."""
    table = json.loads(read_text(root / STANDING_FILE))
    rows = table["rows"]
    columns = [
        _column(root, column, [row["runs"][position] for row in rows])
        for position, column in enumerate(table["columns"])
    ]
    lines = [
        *(_highlights(root, table) if "published" in table else []),
        *([""] if "published" in table else []),
        _line(["", *(column[0] for column in columns)]),
        _line(["---", *("---:" for _ in columns)]),
    ]
    for position, row in enumerate(rows, start=1):
        cells = [column[position] for column in columns]
        if row.get("bold"):
            cells = [f"**{cell}**" if cell else cell for cell in cells]
        lines.append(_line([row["label"], *cells]))
    return "\n".join(lines) + "\n"
