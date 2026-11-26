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


def standing_table(root: Path) -> str:
    """Return the score table that heldout/standing.json describes, as Markdown."""
    table = json.loads(read_text(root / STANDING_FILE))
    rows = table["rows"]
    columns = [
        _column(root, column, [row["runs"][position] for row in rows])
        for position, column in enumerate(table["columns"])
    ]
    lines = [
        _line(["", *(column[0] for column in columns)]),
        _line(["---", *("---:" for _ in columns)]),
    ]
    for position, row in enumerate(rows, start=1):
        cells = [column[position] for column in columns]
        if row.get("bold"):
            cells = [f"**{cell}**" if cell else cell for cell in cells]
        lines.append(_line([row["label"], *cells]))
    return "\n".join(lines) + "\n"
