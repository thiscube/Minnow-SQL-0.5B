"""Choose one query among several candidates by running them.

The model writes several candidate queries for a question. Each one is run on the question's
database, and the candidate whose result most others agree with is kept. No reference answer
is used, so this can be done for any question at any time.
"""

from collections import Counter
from collections.abc import Callable, Sequence

from minnow.execution import QueryError, Row


def _has_content(rows: frozenset) -> bool:
    """True when a result holds at least one value that is not NULL."""
    return any(value is not None for row, _ in rows for value in row)


def pick(
    candidates: Sequence[str], run: Callable[[str], Sequence[Row]], prefer_rows: bool = False
) -> str:
    """Return the candidate whose rows the largest number of candidates produce.

    Candidates that do not run are ignored. With prefer_rows, a result with at least one
    non-NULL value beats an empty or all-NULL result whatever the counts, because a wrong query
    often returns nothing. A tie goes to the candidate that was generated first, and when
    nothing runs the first candidate is returned.
    """
    groups: dict[frozenset, list[int]] = {}
    for position, sql in enumerate(candidates):
        try:
            rows = run(sql)
        except QueryError:
            continue
        groups.setdefault(frozenset(Counter(rows).items()), []).append(position)
    if not groups:
        return candidates[0]
    largest = max(
        groups.items(),
        key=lambda item: (prefer_rows and _has_content(item[0]), len(item[1]), -item[1][0]),
    )
    return candidates[largest[1][0]]
