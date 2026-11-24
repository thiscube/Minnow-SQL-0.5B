"""Combine the candidates of several runs into one run by voting.

For each question every run's candidate queries are pooled and run on the database, and the
result most of them agree on is kept, the same rule that picks among one model's candidates.
A run folder with candidates.jsonl gives all its candidates; one without gives its chosen
prediction. The pool takes each run's first candidate, then each run's second, and so on, so
a tie goes to the greedy answers and to the run named first. No reference query is used.

    python -m minnow.combine outputs/combined results/a results/b --spider data/spider

The new folder gets predictions.jsonl, outcomes.jsonl and summary.json, ready for
python -m minnow.runlog add.
"""

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from functools import partial
from itertools import zip_longest
from pathlib import Path

from minnow.evaluate import CANDIDATES_FILE, append_prediction, load_candidates, load_predictions
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


def load_run(folder: Path) -> dict[int, list[str]]:
    """Return a run's candidates per question, or its predictions when it saved no candidates."""
    candidates = load_candidates(folder / CANDIDATES_FILE)
    if candidates:
        return candidates
    return {index: [sql] for index, sql in load_predictions(folder / PREDICTIONS_FILE).items()}


def pool(lists: Sequence[Sequence[str]]) -> list[str]:
    """Interleave candidate lists: every list's first item, then every list's second, ..."""
    return [sql for rank in zip_longest(*lists) for sql in rank if sql is not None]


def combine(
    examples: Sequence[SpiderExample],
    runs: Sequence[Mapping[int, Sequence[str]]],
    spider: Path,
    prefer_rows: bool = False,
) -> dict[int, str]:
    """Return one prediction per question: the one most pooled candidates agree on."""
    combined = {}
    for index, example in enumerate(examples):
        candidates = pool([run[index] for run in runs if index in run])
        if candidates:
            run = partial(run_query, database_path(spider, example.db_id))
            combined[index] = pick(candidates, run, prefer_rows)
    return combined


def main(argv: Sequence[str] | None = None) -> int:
    """Combine run folders into a new run folder and print its counts."""
    parser = argparse.ArgumentParser(description="Combine several runs into one by voting.")
    parser.add_argument("out", type=Path, help="new run folder to write")
    parser.add_argument("runs", type=Path, nargs="+", help="run folders to combine, best first")
    parser.add_argument("--spider", type=Path, required=True, help="unpacked Spider folder")
    parser.add_argument("--split", default="dev.json", help="question file the runs answered")
    parser.add_argument("--prefer-rows", action="store_true", help="put empty results last")
    args = parser.parse_args(argv)
    if (args.out / PREDICTIONS_FILE).exists():
        parser.error(f"{args.out} already holds a run.")
    examples = load_split(args.spider, args.split)
    runs = [load_run(folder) for folder in args.runs]
    combined = combine(examples, runs, args.spider, args.prefer_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    for index in sorted(combined):
        append_prediction(args.out / PREDICTIONS_FILE, index, combined[index])
    outcomes = judge_all(examples, combined, args.spider)
    save_outcomes(args.out / OUTCOMES_FILE, outcomes)
    counts = breakdown(outcomes)
    rule = ", empty results last" if args.prefer_rows else ""
    summary = {
        "model": "vote of " + ", ".join(folder.name for folder in args.runs),
        "split": args.split.removesuffix(".json"),
        "correct": counts.correct,
        "total": counts.total,
        "execution_accuracy": round(100 * counts.correct / counts.total, 2),
        "generation_seconds": None,
        "decoding": f"vote over the candidates of {len(runs)} runs{rule}",
    }
    text = json.dumps(summary, indent=2) + "\n"
    (args.out / "summary.json").write_text(text, encoding="utf-8", newline="\n")
    print(format_breakdown(counts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
