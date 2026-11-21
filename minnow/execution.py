"""Execution accuracy: run a predicted and a reference query and compare their rows.

A prediction counts as correct when it runs and returns the same rows as the reference query.
Rows are compared as a multiset, or in order when the reference query has an ORDER BY. Column
order does not matter: SELECT a, b and SELECT b, a give the same result.

This follows the idea of the Spider test-suite evaluation, but it is a simpler reimplementation
and it uses only the original Spider databases.
"""

import re
import sqlite3
import time
from collections import Counter
from collections.abc import Sequence
from itertools import permutations
from pathlib import Path

from minnow.schema import connect_read_only

Row = tuple[object, ...]
_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)
_PROGRESS_STEPS = 1000
_MAX_PERMUTED_COLUMNS = 6


class QueryError(RuntimeError):
    """The query failed, or ran longer than allowed."""


def run_query(database: Path, sql: str, timeout: float = 10.0) -> list[Row]:
    """Run a query on a read-only copy of the database and return all rows."""
    connection = connect_read_only(database)
    connection.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), _PROGRESS_STEPS)
    try:
        return [tuple(row) for row in connection.execute(sql).fetchall()]
    except sqlite3.Error as error:
        raise QueryError(str(error)) from error
    finally:
        connection.close()


def _columns_reordered(rows: Sequence[Row], order: Sequence[int]) -> list[Row]:
    return [tuple(row[index] for index in order) for row in rows]


def _same_rows(predicted: Sequence[Row], gold: Sequence[Row], ordered: bool) -> bool:
    if ordered:
        return list(predicted) == list(gold)
    return Counter(predicted) == Counter(gold)


def results_match(predicted: Sequence[Row], gold: Sequence[Row], ordered: bool) -> bool:
    """True when both results hold the same rows, allowing the columns in any order."""
    if len(predicted) != len(gold):
        return False
    if not gold:
        return True
    width = len(gold[0])
    if any(len(row) != width for row in predicted):
        return False
    if width > _MAX_PERMUTED_COLUMNS:
        return _same_rows(predicted, gold, ordered)
    return any(
        _same_rows(_columns_reordered(predicted, order), gold, ordered)
        for order in permutations(range(width))
    )


def execution_match(database: Path, predicted_sql: str, gold_sql: str) -> bool:
    """True when the predicted query runs and returns the same rows as the reference query."""
    gold = run_query(database, gold_sql)
    try:
        predicted = run_query(database, predicted_sql)
    except QueryError:
        return False
    return results_match(predicted, gold, ordered=bool(_ORDER_BY.search(gold_sql)))
