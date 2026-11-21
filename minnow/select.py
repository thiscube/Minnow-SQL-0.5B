"""Choose one query among several candidates by running them.

The model writes several candidate queries for a question. Each one is run on the question's
database, and the candidate whose result most others agree with is kept. No reference answer
is used, so this can be done for any question at any time.
"""

from collections import Counter
from collections.abc import Callable, Sequence

from minnow.execution import QueryError, Row


def pick(candidates: Sequence[str], run: Callable[[str], Sequence[Row]]) -> str:
    """Return the candidate whose rows the largest number of candidates produce.

    Candidates that do not run are ignored. A tie goes to the candidate that was generated
    first, and when nothing runs the first candidate is returned.
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
    largest = max(groups.values(), key=lambda positions: (len(positions), -positions[0]))
    return candidates[largest[0]]
