"""Keep the results log: results/runs.json is the record and the rest is generated from it.

Run these from the repository root:

    python -m minnow.runlog add outputs/sft --change "Fine-tuned on the Spider training set"
    python -m minnow.runlog render
    python -m minnow.runlog check

"add" logs a run that minnow.outcomes has judged and copies its files into results/. "render"
rewrites results.md, results/chart.svg and the results block of README.md. "check" changes
nothing: it exits with 0 when all of that is up to date and with 1 when it is not. Exit code 2
means the command could not do its work.
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
from minnow.outcomes import OUTCOMES_FILE, PREDICTIONS_FILE, Breakdown, breakdown, load_outcomes
from minnow.report import CHART_FILE, Run, readme_block, results_markdown

TARGET_ACCURACY = 70.0
PUBLISHED_ACCURACY = 72.08
RESULTS_FOLDER = "results"
RUNS_FILE = "results/runs.json"
RESULTS_FILE = "results.md"
README_FILE = "README.md"
SUMMARY_FILE = "summary.json"
BLOCK_START = "<!-- results:start -->"
BLOCK_END = "<!-- results:end -->"
STALE_EXIT_CODE = 1
FAILED_EXIT_CODE = 2
_CHART_TITLE = "Execution accuracy on Spider dev (%)"
_RUN_FILES = (PREDICTIONS_FILE, SUMMARY_FILE, OUTCOMES_FILE)


class RunLogError(ValueError):
    """The results log cannot be written from the files as they are."""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _accuracy(correct: int, total: int) -> float:
    return round(100 * correct / total, 2) if total else 0.0


def load_runs(root: Path) -> list[Run]:
    """Read the logged runs in the order they were added; no file yet means no runs."""
    return [Run(**record) for record in json.loads(_read(root / RUNS_FILE) or "[]")]


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


def replace_block(readme: str, block: str) -> str:
    """Put the block between the two marker lines of the README text."""
    before, start, rest = readme.partition(BLOCK_START)
    _, end, after = rest.partition(BLOCK_END)
    if not (start and end):
        raise RunLogError(
            f"{README_FILE} needs the lines {BLOCK_START} and {BLOCK_END} around its results."
        )
    return f"{before}{BLOCK_START}\n\n{block}\n{BLOCK_END}{after}"


def _counts(root: Path, run: Run) -> Breakdown:
    path = root / RESULTS_FOLDER / run.name / OUTCOMES_FILE
    if not path.is_file():
        raise RunLogError(f"{path} does not exist, so {run.name} cannot be reported.")
    counts = breakdown(load_outcomes(path))
    logged = (run.correct, run.total, run.execution_accuracy)
    judged = (counts.correct, counts.total, _accuracy(counts.correct, counts.total))
    if logged != judged:
        raise RunLogError(
            f"{RUNS_FILE} gives {run.name} {logged[0]} of {logged[1]} correct ({logged[2]}%), "
            f"but its {OUTCOMES_FILE} has {judged[0]} of {judged[1]} ({judged[2]}%)."
        )
    return counts


def rendered(root: Path, runs: Sequence[Run]) -> dict[str, str]:
    """Return what every generated file should hold, by its path from the repository root."""
    counts = {run.name: _counts(root, run) for run in runs}
    return {
        RESULTS_FILE: results_markdown(runs, counts),
        CHART_FILE: chart_svg(runs),
        README_FILE: replace_block(_read(root / README_FILE), readme_block(runs)),
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
    runs = load_runs(root)
    problems = _unlogged(root, runs)
    if problems:
        return problems
    try:
        expected = rendered(root, runs)
    except RunLogError as error:
        return [str(error)]
    return [
        f"{name} is out of date: run python -m minnow.runlog render"
        for name, text in expected.items()
        if _read(root / name) != text
    ]


def _judged_run(folder: Path, change: str, day: str) -> Run:
    for name in _RUN_FILES:
        if not (folder / name).is_file():
            hint = " Run python -m minnow.outcomes on it first." if name == OUTCOMES_FILE else ""
            raise RunLogError(f"{folder / name} does not exist.{hint}")
    summary = json.loads(_read(folder / SUMMARY_FILE))
    counts = breakdown(load_outcomes(folder / OUTCOMES_FILE))
    if (summary["correct"], summary["total"]) != (counts.correct, counts.total):
        raise RunLogError(
            f"{SUMMARY_FILE} says {summary['correct']} of {summary['total']} correct, but "
            f"{OUTCOMES_FILE} has {counts.correct} of {counts.total}. Judge the run again with "
            "python -m minnow.outcomes."
        )
    return Run(
        name=folder.resolve().name,
        date=day,
        model=summary["model"],
        change=" ".join(change.split()),
        correct=counts.correct,
        total=counts.total,
        execution_accuracy=_accuracy(counts.correct, counts.total),
        generation_seconds=summary["generation_seconds_this_run"],
    )


def add(root: Path, folder: Path, change: str, day: str) -> Run:
    """Log a judged run: copy its files into results/, record it and render the log."""
    run = _judged_run(folder, change, day)
    kept = root / RESULTS_FOLDER / run.name
    if folder.resolve() != kept.resolve():
        kept.mkdir(parents=True, exist_ok=True)
        for name in _RUN_FILES:
            shutil.copyfile(folder / name, kept / name)
    runs = load_runs(root)
    names = [logged.name for logged in runs]
    if run.name in names:
        runs[names.index(run.name)] = run
    else:
        runs.append(run)
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
    commands.add_parser("render", help="rewrite results.md, the chart and the README block")
    commands.add_parser("check", help="exit with 1 when the results log is out of date")
    return parser.parse_args(argv)


def _run_command(args: argparse.Namespace, root: Path) -> int:
    if args.command == "add":
        run = add(root, args.run, args.change, (args.date or date.today()).isoformat())
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
