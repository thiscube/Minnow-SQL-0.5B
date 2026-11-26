"""Keep the results log: results/runs.json is the record and the rest is generated from it.

Run these from the repository root:

    python -m minnow.runlog add outputs/sft --change "Fine-tuned on the Spider training set"
    python -m minnow.runlog render
    python -m minnow.runlog check

"add" logs a run folder that holds predictions, outcomes and a summary, and copies its files
into results/. "render" rewrites results.md, results/chart.svg and the results block of
README.md, and the README's score table when heldout/standing.json describes one. "check"
changes nothing: it exits with 0 when all of that is up to date and with 1 when it is not.
Exit code 2 means the command could not do its work.
"""

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import date
from pathlib import Path

from minnow.chart import bar_chart
from minnow.report import CHART_FILE, Run, readme_block, results_markdown
from minnow.runfiles import (
    RUN_FILES,
    SUMMARY_FILE,
    RunLogError,
    judged_run,
    read_text,
    verified_counts,
)
from minnow.standing import STANDING_END, STANDING_FILE, STANDING_START, standing_table

TARGET_ACCURACY = 70.0
PUBLISHED_ACCURACY = 72.08
RESULTS_FOLDER = "results"
RUNS_FILE = "results/runs.json"
RESULTS_FILE = "results.md"
README_FILE = "README.md"
BLOCK_START = "<!-- results:start -->"
BLOCK_END = "<!-- results:end -->"
STALE_EXIT_CODE = 1
FAILED_EXIT_CODE = 2
_CHART_TITLE = "Execution accuracy on Spider dev (%)"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _comparable(runs: Sequence[Run]) -> list[Run]:
    """Return the runs, refusing a log whose runs cannot be told apart or compared."""
    if len({run.name for run in runs}) != len(runs):
        raise RunLogError(f"{RUNS_FILE} lists the same run name twice.")
    if len({run.total for run in runs}) > 1:
        raise RunLogError(f"The runs in {RUNS_FILE} do not all cover the same questions.")
    return list(runs)


def load_runs(root: Path) -> list[Run]:
    """Read the logged runs in the order they were added; no file yet means no runs."""
    records = json.loads(read_text(root / RUNS_FILE) or "[]")
    return _comparable([Run(**record) for record in records])


def save_runs(root: Path, runs: Sequence[Run]) -> None:
    """Write the logged runs to results/runs.json."""
    _write(root / RUNS_FILE, json.dumps([asdict(run) for run in runs], indent=2) + "\n")


def chart_svg(runs: Sequence[Run]) -> str:
    """Draw every run's accuracy next to the target and the best published result."""
    markers = [
        (f"target {TARGET_ACCURACY:g}%", TARGET_ACCURACY),
        (f"SLM-SQL-0.5B, published {PUBLISHED_ACCURACY:g}%", PUBLISHED_ACCURACY),
    ]
    return bar_chart(_CHART_TITLE, [(run.name, run.execution_accuracy) for run in runs], markers)


def replace_block(readme: str, block: str, start: str = BLOCK_START, end: str = BLOCK_END) -> str:
    """Put the block between the two marker lines of the README text."""
    before, found_start, rest = readme.partition(start)
    _, found_end, after = rest.partition(end)
    if not (found_start and found_end):
        raise RunLogError(f"{README_FILE} needs the lines {start} and {end} around that part.")
    return f"{before}{start}\n\n{block}\n{end}{after}"


def _readme(root: Path, runs: Sequence[Run]) -> str:
    """Return the README with its generated blocks filled in."""
    readme = replace_block(read_text(root / README_FILE), readme_block(runs))
    if (root / STANDING_FILE).is_file():
        readme = replace_block(readme, standing_table(root), STANDING_START, STANDING_END)
    return readme


def rendered(root: Path, runs: Sequence[Run]) -> dict[str, str]:
    """Return what every generated file should hold, by its path from the repository root."""
    counts = {run.name: verified_counts(root / RESULTS_FOLDER / run.name, run) for run in runs}
    return {
        RESULTS_FILE: results_markdown(runs, counts),
        CHART_FILE: chart_svg(runs),
        README_FILE: _readme(root, runs),
    }


def render(root: Path) -> None:
    """Rewrite results.md, the chart and the README's results block from the logged runs."""
    for name, text in rendered(root, load_runs(root)).items():
        _write(root / name, text)


def _unlogged(root: Path, runs: Sequence[Run]) -> list[str]:
    logged = {run.name for run in runs}
    found = {path.parent.name for path in (root / RESULTS_FOLDER).glob(f"*/{SUMMARY_FILE}")}
    return [
        *(
            f"{RESULTS_FOLDER}/{name} has a {SUMMARY_FILE} but is not in {RUNS_FILE}: "
            "log it with python -m minnow.runlog add"
            for name in sorted(found - logged)
        ),
        *(
            f"{RUNS_FILE} lists {name}, but {RESULTS_FOLDER}/{name}/{SUMMARY_FILE} does not exist"
            for name in sorted(logged - found)
        ),
    ]


def check(root: Path) -> list[str]:
    """Return everything that is out of date in the results log; nothing means it is current."""
    try:
        runs = load_runs(root)
        problems = _unlogged(root, runs)
        if problems:
            return problems
        expected = rendered(root, runs)
    except RunLogError as error:
        return [str(error)]
    return [
        f"{name} is out of date: run python -m minnow.runlog render"
        for name, text in expected.items()
        if read_text(root / name) != text
    ]


def add(root: Path, folder: Path, change: str, day: str, replace: bool = False) -> Run:
    """Log a judged run: copy its files into results/, record it and render the log."""
    run = judged_run(folder, change, day)
    runs = load_runs(root)
    names = [logged.name for logged in runs]
    if run.name in names and not replace:
        raise RunLogError(
            f"A run named {run.name} is already logged. Give the new run's folder another "
            "name, or pass --replace to overwrite the logged one."
        )
    if run.name in names:
        runs[names.index(run.name)] = run
    else:
        runs.append(run)
    _comparable(runs)
    kept = root / RESULTS_FOLDER / run.name
    if folder.resolve() != kept.resolve():
        kept.mkdir(parents=True, exist_ok=True)
        for name in RUN_FILES:
            shutil.copyfile(folder / name, kept / name)
    files = rendered(root, runs)
    save_runs(root, runs)
    for name, text in files.items():
        _write(root / name, text)
    return run


def _parse(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Keep the results log up to date.")
    commands = parser.add_subparsers(dest="command", required=True)
    adding = commands.add_parser("add", help="log a judged run and render the log")
    adding.add_argument("run", type=Path, help="run folder with predictions, summary and outcomes")
    adding.add_argument("--change", required=True, help="one line saying what changed")
    adding.add_argument(
        "--date", type=date.fromisoformat, help="day of the run as YYYY-MM-DD (default: today)"
    )
    adding.add_argument(
        "--replace", action="store_true", help="overwrite the logged run with the same name"
    )
    commands.add_parser("render", help="rewrite results.md, the chart and the README block")
    commands.add_parser("check", help="exit with 1 when the results log is out of date")
    return parser.parse_args(argv)


def _run_command(args: argparse.Namespace, root: Path) -> int:
    if args.command == "add":
        day = (args.date or date.today()).isoformat()
        run = add(root, args.run, args.change, day, args.replace)
        print(f"Logged {run.name}: {run.execution_accuracy:.2f}% ({run.correct} of {run.total}).")
    elif args.command == "render":
        render(root)
        print(f"Rendered {RESULTS_FILE}, {CHART_FILE} and the results block of {README_FILE}.")
    else:
        problems = check(root)
        if problems:
            print("\n".join(problems))
            return STALE_EXIT_CODE
        print("The results log is up to date.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run add, render or check on the repository in the current folder."""
    args = _parse(argv)
    try:
        return _run_command(args, Path())
    except RunLogError as error:
        print(error, file=sys.stderr)
        return FAILED_EXIT_CODE


if __name__ == "__main__":
    sys.exit(main())
