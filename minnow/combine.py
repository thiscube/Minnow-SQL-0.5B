"""Combine the predictions of several runs into one run by voting.

For each question the runs' predictions are run on the database and the result most of them
agree on is kept, the same rule that picks among one model's candidates. A tie goes to the run
that is named first. No reference query is used to choose.

    python -m minnow.combine outputs/combined results/a results/b results/c --spider data/spider

The new folder gets predictions.jsonl, outcomes.jsonl and summary.json, ready for
python -m minnow.runlog add.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from minnow.evaluate import append_prediction, load_predictions
from minnow.execution import run_query
from minnow.outcomes import (
    OUTCOMES_FILE,
    PREDICTIONS_FILE,
    breakdown,
    format_breakdown,
    judge_all,
    save_outcomes,
)
from minnow.select import pick
from minnow.spider import SpiderExample, database_path, load_split


def combine(
    examples: Sequence[SpiderExample], runs: Sequence[Mapping[int, str]], spider: Path
) -> dict[int, str]:
    """Return one prediction per question: the one most runs agree on when it is run."""
    combined = {}
    for index, example in enumerate(examples):
        candidates = [run[index] for run in runs if index in run]
        if candidates:
            database = database_path(spider, example.db_id)
            combined[index] = pick(candidates, lambda sql, db=database: run_query(db, sql))
    return combined


def main(argv: Sequence[str] | None = None) -> int:
    """Combine run folders into a new run folder and print its counts."""
    parser = argparse.ArgumentParser(description="Combine several runs into one by voting.")
    parser.add_argument("out", type=Path, help="new run folder to write")
    parser.add_argument("runs", type=Path, nargs="+", help="run folders to combine, best first")
    parser.add_argument("--spider", type=Path, required=True, help="unpacked Spider folder")
    args = parser.parse_args(argv)
    if (args.out / PREDICTIONS_FILE).exists():
        parser.error(f"{args.out} already holds a run.")
    examples = load_split(args.spider, "dev.json")
    runs = [load_predictions(folder / PREDICTIONS_FILE) for folder in args.runs]
    combined = combine(examples, runs, args.spider)
    args.out.mkdir(parents=True, exist_ok=True)
    for index in sorted(combined):
        append_prediction(args.out / PREDICTIONS_FILE, index, combined[index])
    outcomes = judge_all(examples, combined, args.spider)
    save_outcomes(args.out / OUTCOMES_FILE, outcomes)
    counts = breakdown(outcomes)
    summary = {
        "model": "vote of " + ", ".join(folder.name for folder in args.runs),
        "split": "dev",
        "correct": counts.correct,
        "total": counts.total,
        "execution_accuracy": round(100 * counts.correct / counts.total, 2),
        "generation_seconds": None,
        "decoding": f"vote over {len(runs)} runs",
    }
    text = json.dumps(summary, indent=2) + "\n"
    (args.out / "summary.json").write_text(text, encoding="utf-8", newline="\n")
    print(format_breakdown(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
