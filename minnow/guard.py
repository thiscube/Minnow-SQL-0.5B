"""Guard against training on Spider dev data.

Spider dev is only for scoring. Training data leaks if it contains a dev
question, or any example that uses a dev database.

Check two JSON files, each a list of objects with "question" and "db_id":

    python -m minnow.guard train.json dev.json

The exit code is 0 when the data is clean and 1 when it leaks.
"""

import argparse
import json
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

_NOT_A_WORD = re.compile(r"[\W_]+")
_SHOWN_PER_KIND = 5


class LeakError(ValueError):
    """Training data overlaps with the dev set."""


@dataclass(frozen=True)
class Example:
    """One question and the database it is asked about."""

    question: str
    db_id: str


@dataclass(frozen=True)
class LeakReport:
    """Training questions and databases that also appear in the dev set."""

    questions: tuple[str, ...]
    databases: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """True when nothing from the dev set was found."""
        return not self.questions and not self.databases


def normalize(text: str) -> str:
    """Lowercase the text and drop punctuation and extra spaces, so they can't hide a match."""
    return _NOT_A_WORD.sub(" ", text.casefold()).strip()


def find_leaks(train: Iterable[Example], dev: Iterable[Example]) -> LeakReport:
    """Return every training question and database that the dev set also has."""
    dev_examples = list(dev)
    if not dev_examples:
        raise ValueError("The dev set is empty, so there is nothing to check against.")
    dev_questions = {normalize(example.question) for example in dev_examples}
    dev_databases = {normalize(example.db_id) for example in dev_examples}
    questions: set[str] = set()
    databases: set[str] = set()
    for example in train:
        if normalize(example.question) in dev_questions:
            questions.add(example.question)
        if normalize(example.db_id) in dev_databases:
            databases.add(example.db_id)
    return LeakReport(tuple(sorted(questions)), tuple(sorted(databases)))


def assert_no_leak(train: Iterable[Example], dev: Iterable[Example]) -> None:
    """Raise LeakError if the training data contains anything from the dev set."""
    report = find_leaks(train, dev)
    if not report.clean:
        raise LeakError(_describe(report))


def _describe(report: LeakReport) -> str:
    lines = [
        f"Training data leaks from the dev set: {len(report.questions)} question(s), "
        f"{len(report.databases)} database(s)."
    ]
    lines += [f"  question: {question}" for question in report.questions[:_SHOWN_PER_KIND]]
    lines += [f"  database: {database}" for database in report.databases[:_SHOWN_PER_KIND]]
    return "\n".join(lines)


def load_examples(path: Path) -> list[Example]:
    """Read examples from a JSON list of objects with "question" and "db_id"."""
    items = json.loads(path.read_text(encoding="utf-8"))
    return [Example(item["question"], item["db_id"]) for item in items]


def main(argv: Sequence[str] | None = None) -> int:
    """Check a training file against a dev file and print the result."""
    parser = argparse.ArgumentParser(description="Check training data for Spider dev leaks.")
    parser.add_argument("train", type=Path, help="JSON file with the training examples")
    parser.add_argument("dev", type=Path, help="JSON file with the dev examples")
    args = parser.parse_args(argv)
    train = load_examples(args.train)
    dev = load_examples(args.dev)
    try:
        assert_no_leak(train, dev)
    except LeakError as error:
        print(error)
        return 1
    print(f"No leak: {len(train)} training examples checked against {len(dev)} dev examples.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
