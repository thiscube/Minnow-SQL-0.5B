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


def run_query(
    database: Path, sql: str, timeout: float = 10.0, max_rows: int = 1_000_000
) -> list[Row]:
    """Run a query on a read-only copy of the database and return all rows.

    A query that fails, runs longer than the timeout or returns more than max_rows rows
    raises QueryError, so a runaway query can't hang the run or fill the memory.
    """
    connection = connect_read_only(database)
    connection.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), _PROGRESS_STEPS)
    try:
        rows = connection.execute(sql).fetchmany(max_rows + 1)
    except sqlite3.Error as error:
        raise QueryError(str(error)) from error
    finally:
        connection.close()
    if len(rows) > max_rows:
        raise QueryError(f"more than {max_rows} rows")
    return rows


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


def reference_is_ordered(sql: str) -> bool:
    """True when the reference query sorts its rows, so a prediction must match their order."""
    return bool(_ORDER_BY.search(sql))


def execution_match(database: Path, predicted_sql: str, gold_sql: str) -> bool:
    """True when the predicted query runs and returns the same rows as the reference query."""
    gold = run_query(database, gold_sql)
    try:
        predicted = run_query(database, predicted_sql)
    except QueryError:
        return False
    return results_match(predicted, gold, ordered=reference_is_ordered(gold_sql))
