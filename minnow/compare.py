"""Compare two runs question by question, using only their saved outcomes.

    python -m minnow.compare results/baseline results/sft

The first folder is the run before a change and the second the run after it. Both need the
outcomes.jsonl that minnow.outcomes writes. "Fixed" questions were wrong before and are correct
after; "broken" questions were correct before and are wrong after.
"""

import argparse
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from minnow.outcomes import CORRECT, OUTCOMES_FILE, Outcome, load_outcomes

_SHOWN_PER_KIND = 20


@dataclass(frozen=True)
class Comparison:
    """How the answers of two runs differ, question by question."""

    both_correct: int
    both_wrong: int
    fixed: tuple[int, ...]
    broken: tuple[int, ...]

    @property
    def net_change(self) -> int:
        """How many more questions are correct after than before."""
        return len(self.fixed) - len(self.broken)


def _correct_by_index(outcomes: Iterable[Outcome]) -> dict[int, bool]:
    return {outcome.index: outcome.verdict == CORRECT for outcome in outcomes}


def compare(before: Iterable[Outcome], after: Iterable[Outcome]) -> Comparison:
    """Count the questions that stayed the same, were fixed or were broken between two runs."""
    was_correct = _correct_by_index(before)
    is_correct = _correct_by_index(after)
    if was_correct.keys() != is_correct.keys():
        different = len(was_correct.keys() ^ is_correct.keys())
        raise ValueError(f"The runs do not cover the same questions: {different} differ.")
    indexes = sorted(was_correct)
    return Comparison(
        both_correct=sum(was_correct[index] and is_correct[index] for index in indexes),
        both_wrong=sum(not was_correct[index] and not is_correct[index] for index in indexes),
        fixed=tuple(index for index in indexes if not was_correct[index] and is_correct[index]),
        broken=tuple(index for index in indexes if was_correct[index] and not is_correct[index]),
    )


def _index_list(kind: str, indexes: Sequence[int]) -> str:
    shown = ", ".join(str(index) for index in indexes[:_SHOWN_PER_KIND]) or "none"
    more = len(indexes) - _SHOWN_PER_KIND
    return f"{kind}: {shown}" + (f" and {more} more" if more > 0 else "")


def format_comparison(result: Comparison) -> str:
    """Lay the counts out as a small table, followed by the first fixed and broken questions."""
    rows = [
        ("both correct", str(result.both_correct)),
        ("both wrong", str(result.both_wrong)),
        ("fixed", str(len(result.fixed))),
        ("broken", str(len(result.broken))),
        ("net change", f"{result.net_change:+d}"),
    ]
    lines = [f"{label:<12}  {count:>5}" for label, count in rows]
    return "\n".join(
        [*lines, _index_list("fixed", result.fixed), _index_list("broken", result.broken)]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Compare the outcomes of two run folders and print the result."""
    parser = argparse.ArgumentParser(description="Compare two runs question by question.")
    parser.add_argument("before", type=Path, help="run folder from before the change")
    parser.add_argument("after", type=Path, help="run folder from after the change")
    args = parser.parse_args(argv)
    for folder in (args.before, args.after):
        if not (folder / OUTCOMES_FILE).is_file():
            parser.error(f"{folder / OUTCOMES_FILE} does not exist: run minnow.outcomes first.")
    try:
        result = compare(
            load_outcomes(args.before / OUTCOMES_FILE), load_outcomes(args.after / OUTCOMES_FILE)
        )
    except ValueError as error:
        parser.error(str(error))
    print(format_comparison(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
