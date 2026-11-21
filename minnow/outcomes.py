"""Judge every prediction of a run once and save the verdicts.

A prediction is "correct" when it returns the same rows as the reference query, "wrong_rows"
when it runs and returns something else, and "error" when it does not run. Everything else
that is reported about a run is counted from these saved verdicts.

    python -m minnow.outcomes results/baseline --spider data/spider

This reads predictions.jsonl from the run folder, writes outcomes.jsonl next to it and prints
the counts.
"""

import argparse
import json
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from minnow.evaluate import load_predictions
from minnow.execution import QueryError, reference_is_ordered, results_match, run_query
from minnow.spider import SpiderExample, database_path, load_split

CORRECT = "correct"
WRONG_ROWS = "wrong_rows"
ERROR = "error"
MISSING_PREDICTION = "missing prediction"
PREDICTIONS_FILE = "predictions.jsonl"
OUTCOMES_FILE = "outcomes.jsonl"


@dataclass(frozen=True)
class Outcome:
    """What happened when one prediction was run against its reference query."""

    index: int
    verdict: str
    error: str
    gold_rows: int


@dataclass(frozen=True)
class Breakdown:
    """The outcomes of a run, counted."""

    total: int
    correct: int
    wrong_rows: int
    did_not_run: int
    error_classes: tuple[tuple[str, int], ...]
    correct_on_empty: int


def error_class(message: str) -> str:
    """Shorten a SQLite error message to its kind, without the names from the query."""
    if message == "interrupted":
        return "timeout"
    if "syntax error" in message:
        return "syntax error"
    return message.split(":", 1)[0]


def judge(database: Path, predicted_sql: str | None, gold_sql: str, index: int = 0) -> Outcome:
    """Run a prediction and its reference query and say how they compare."""
    gold = run_query(database, gold_sql)
    if predicted_sql is None:
        return Outcome(index, ERROR, MISSING_PREDICTION, len(gold))
    try:
        predicted = run_query(database, predicted_sql)
    except QueryError as error:
        return Outcome(index, ERROR, error_class(str(error)), len(gold))
    same = results_match(predicted, gold, ordered=reference_is_ordered(gold_sql))
    return Outcome(index, CORRECT if same else WRONG_ROWS, "", len(gold))


def judge_all(
    examples: Sequence[SpiderExample], predictions: Mapping[int, str], spider: Path
) -> list[Outcome]:
    """Judge the prediction for every example; a missing prediction counts as an error."""
    return [
        judge(database_path(spider, example.db_id), predictions.get(index), example.query, index)
        for index, example in enumerate(examples)
    ]


def save_outcomes(path: Path, outcomes: Iterable[Outcome]) -> None:
    """Write the outcomes as JSON lines, sorted by index."""
    ordered = sorted(outcomes, key=lambda outcome: outcome.index)
    text = "".join(json.dumps(asdict(outcome)) + "\n" for outcome in ordered)
    path.write_text(text, encoding="utf-8", newline="\n")


def load_outcomes(path: Path) -> list[Outcome]:
    """Read the outcomes that save_outcomes wrote."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Outcome(**json.loads(line)) for line in lines if line]


def breakdown(outcomes: Iterable[Outcome]) -> Breakdown:
    """Count the verdicts, the kinds of error and the correct answers to empty results."""
    judged = list(outcomes)
    verdicts = Counter(outcome.verdict for outcome in judged)
    errors = Counter(outcome.error for outcome in judged if outcome.verdict == ERROR)
    return Breakdown(
        total=len(judged),
        correct=verdicts[CORRECT],
        wrong_rows=verdicts[WRONG_ROWS],
        did_not_run=verdicts[ERROR],
        error_classes=tuple(sorted(errors.items(), key=lambda item: (-item[1], item[0]))),
        correct_on_empty=sum(o.verdict == CORRECT and o.gold_rows == 0 for o in judged),
    )


def format_breakdown(counts: Breakdown) -> str:
    """Lay the counts out as a small table."""
    rows = [
        ("correct", counts.correct),
        ("ran with different rows", counts.wrong_rows),
        ("did not run", counts.did_not_run),
        *((f"  {name}", count) for name, count in counts.error_classes),
        ("total", counts.total),
        ("correct with an empty reference result", counts.correct_on_empty),
    ]
    width = max(len(label) for label, _ in rows)
    return "\n".join(f"{label:<{width}}  {count:>5}" for label, count in rows)


def main(argv: Sequence[str] | None = None) -> int:
    """Judge the predictions of a run folder, save the outcomes and print the counts."""
    parser = argparse.ArgumentParser(description="Judge every prediction of a run and save it.")
    parser.add_argument("run", type=Path, help="run folder that holds predictions.jsonl")
    parser.add_argument("--spider", type=Path, required=True, help="unpacked Spider folder")
    parser.add_argument("--split", default="dev.json", help="split file inside the Spider folder")
    args = parser.parse_args(argv)
    predictions_file = args.run / PREDICTIONS_FILE
    if not predictions_file.is_file():
        parser.error(f"{predictions_file} does not exist, so there is nothing to judge.")
    examples = load_split(args.spider, args.split)
    outcomes = judge_all(examples, load_predictions(predictions_file), args.spider)
    save_outcomes(args.run / OUTCOMES_FILE, outcomes)
    print(format_breakdown(breakdown(outcomes)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
