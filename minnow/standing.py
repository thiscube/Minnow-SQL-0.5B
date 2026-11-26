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
    return [column["title"], *(f"{run.execution_accuracy:.2f}%" if run else "" for run in runs)]


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
    half = ", more than half" if 2 * wrong_before > before.total else ""
    return [
        f"- **{mistakes}.** It now gets {round(100 * wrong_after / after.total)} of every 100 "
        f"questions wrong. Before training it got {round(100 * wrong_before / before.total)} "
        f"wrong{half}.",
        f"- **{after.correct / before.correct:.1f} times as many right.** It answers "
        f"{after.correct:,} of the {after.total:,} questions correctly, up from "
        f"{before.correct:,}.",
        f"- **{unseen.execution_accuracy:.0f}% on questions it had never seen.** "
        f"{published['name']}, a published model of the same size, reports "
        f"{published['score']:g}%.",
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
